# sectvid v4: Full Build Specification

**Read this first.** This document is the only context a builder gets. It was written after a long design discussion, and the reasoning is included so you can judge edge cases. If anything here conflicts with what seems convenient, follow the document or ask. Values marked *(guess)* are untested starting points, not requirements.

## 1. Goal

Generate **people (including faces) video, frame by frame, over long horizons, with correct motion and stable identity.**

- **Deliverable A (now):** predict what happens next from recent frames. No behavior control.
- **B (later):** label-conditioned prediction. **C (later):** produce a requested action. Do not build B/C yet, but keep the label input slot.
- Start narrow: a few hundred clips, then scale only after Gate 4 (Section 12).

## 2. Existing repo (inherited code)

The prior project "sectvid" is a bio-inspired model: input vector → Kenyon Cell layer (sparse coding) → MBON layer (Hebbian) → conv decoder producing one 32×32 grayscale frame, trained with single-frame MSE. Files present: `brain.py`, `layers/{kenyon_cell_layer,mbon_layer,decoder}.py`, tests. Ten modules exist only as `.pyc` under Python 3.12 (`config, utils, main, data_generator, workspace, evaluation, logger, marketplace, parasite, visualization`); source is lost.

**Disposition:** the old pipeline is **not** the foundation. Do not train the old decoder or reuse its loss. Quarantine it in `legacy/`. Only the KC/MBON layers may return, as an optional module (Section 7). Known defects, so nobody reuses them: `dream()` uses a hardcoded reward `error=0.1` (self-reward with no quality signal); `TemporalMemory` mutates buffers inside `forward` and only at batch size 1; the 1-D branch of `MBONLayer.update_weights` broadcasts to every column; KC adaptive-threshold smoothing leaks state across inputs; `prune_and_grow` can leave connectivity below the pruned count; `torch.load(weights_only=False)` must not load untrusted files.

## 3. Lessons that drive the design (from the owner's earlier video-model work)

1. A sequence of plausible images is not an animation. Repeated "next frame" prediction gives a **morph**: smooth-looking, with motion and identity quietly failing.
2. The easiest answer to a motion objective is **"don't move."** Most footage has little motion, and small changes are barely visible when rendered. Speeding up afterward does not make motion correct.
3. A state describing appearance may **not contain what is needed to move it** (direction, object structure).
4. **Train and judge on the rollouts you will use.** One-step training collapses when outputs become inputs.
5. **Separate what moves from how it looks.** Explicit motion/structure keeps identity and makes movement controllable; an image model renders appearance.
6. **Data and sampling matter more than architecture:** static clips, unclear subjects, mixed actions, ignored timing teach the wrong thing. Broad unlabeled mixes average distinct actions into generic motion.
7. **Tests must be able to prove us wrong.** One score or one striking frame hides frozen motion, mush, drift, accidental sharpness.

Recurring lesson: video fails when appearance modeling is asked to stand in for temporal understanding.

## 4. Design: how hand-drawn animation works, and what we copy

Animators draw **key poses**, then **breakdowns** (path and character of the move), then **inbetweens**. Inbetweeners **redraw the form** at each intermediate pose; they never average two drawings. **Where** (arcs) and **when** (spacing, holds) are planned separately. Identity comes from a **model sheet**, so each drawing is checked against the character, not the previous drawing. This is why errors do not compound.

So: a **motion planner** predicts the next pose in structure space (autoregressive); a **drawer** generates every frame in full from that pose plus reference frames. Pixel errors never feed the next prediction.

**Banned approaches (do not build these, even as a baseline for the model):**

- predicting frame t+1 by editing/altering frame t's pixels
- warping or blending frames, optical-flow-and-warp pipelines as the generator
- interpolating between two images or latents to "get from A to B"
- full-frame pixel regression (L1/L2/SSIM) to a target frame as a training signal
- any self-assigned reward as a training signal

(The warp/flow baseline may exist **only as an evaluation reference** to measure how much better we are.)

## 5. Definitions

- **Structure / pose sheet:** compact per-frame description of the subject (Section 6.2).
- **Planner:** predicts next structure from recent structure. **Drawer:** renders a frame from structure + reference.
- **Reference sheet:** a few real frames of the same person.
- **Window:** a fixed-length segment (*guess:* 32 frames) cut from a clip; the unit of sampling.
- **Bin:** still / slow / fast, by motion statistics. **Rollout:** the model feeds on its own outputs for K steps. **Horizon:** rollout length.

## 6. Specification

### 6.1 Data

```
data/clips/<clip_id>.mp4
data/meta/<clip_id>.json   # identity_id, behavior (token|"unlabeled"), direction (left|right|toward|away|none|null),
                           # fps, camera_motion_stats, bins per window
data/structure/<clip_id>.npz   # cached structure per frame + confidences
```

- Cut **windows**, not whole clips; bin each by camera and subject motion. **Do not delete still windows** (teaches "everything must move") and **do not let them dominate** (teaches "don't move"). Sample bins with fixed quotas, start near one third each *(guess)*.
- **One behavior per window.** Split mixed clips; labels do not fix mixing.
- Record labels now; **they are not model inputs until Stage 6.**
- Splits: held-out by **clip and by identity** (some identities never seen in training), stratified by behavior. Leakage tests required (Section 10).
- Augmentation must transform structure consistently (horizontal flip mirrors coordinates and swaps left/right keypoints and direction labels). No augmentation that breaks structure.

### 6.2 Structure schema (per frame)

| Part | Content | Notes |
| --- | --- | --- |
| Body | 17 2D keypoints (x, y, confidence), image-normalized to \[0,1\] | add hands only if later needed |
| Head | yaw, pitch, roll + 2D position + scale |  |
| Face | landmarks or expression parameters from a face crop | face-crop coordinates, kept separate from body scale |
| Silhouette | person mask | used for conditioning and metrics |
| Camera | 6-parameter affine, estimated from background features | **subject motion = residual after removing camera**; supervise and score separately |
| Timing | dt (seconds) | holds and "twos" are valid |

**Known risk (lesson 3):** the schema omits hair, cloth, hands, lighting, mouth interior; the drawer must guess them. Log "unplanned detail" failures and extend the schema when a failure traces to a missing field. **Extractor noise becomes ground truth:** carry per-point confidence, weight losses by it, smooth lightly, and validate the extractor in Stage 0.

### 6.3 Planner

- **Input:** last N structures *(guess N=8-16)*, dt, label slot (**always dropped** in A). **Output:** next structure, next dt. Suggested: small causal transformer over per-frame structure tokens *(guess)*.
- **Losses:** robust coordinate loss on keypoints/head pose/landmarks weighted by confidence; velocity loss; timing loss; magnitude term as a **regularizer only**. Direction and endpoint accuracy must be supervised, because amount of motion alone can be right while direction is wrong.
- **Rollout training:** unroll K steps on its own predictions, grow K on a curriculum *(guess 2→8→32→target)*; with probability \~0.1 feed real structure instead, flagged by an input bit and excluded from the loss (the FAR clean-context idea).
- **Mean collapse:** deterministic regression averages ambiguous futures (e.g., a mouth). Only if you observe hedged plans, switch to a multimodal head (flow matching on structure).

### 6.4 Drawer

- **Do not train from scratch.** A few hundred clips cannot teach what people and faces look like. Fine-tune a **pretrained image generator** with structure conditioning (render structure as skeleton/landmark/mask maps) and **reference attention** over the reference sheet. Model choice requires owner approval. Starting resolution *(guess:* 256×256).
- **Reference rule:** references come from the same identity but **a different time than the target and never adjacent frames** (enforced by test); otherwise the model learns to copy.
- **Objective:** generative (noise/velocity prediction on latents) + perceptual + identity-feature similarity to the reference + **structure obedience** (re-detect structure on the output and compare to the input structure).
- **Flicker:** frames drawn independently can flicker. Start with reference-only conditioning and **measure**. Only if flicker fails, add the last 1-2 *generated* frames as context (never as a base to edit), and flag it because that is where drift can return.

## 7. Mushroom-body module (optional)

Claimed strength: fast few-shot association (retrieval, recognition), not image synthesis and not 3D form. Allowed jobs: **reference retrieval** from structure, **behavior recognition**, **keyframe memory**. Each must beat a plain baseline (nearest-neighbor retrieval, small learned classifier) on held-out data. Fail → remove. It is **not** a mechanism for the drawer to learn from less data; pretraining is.

## 8. Loss registry

| Name | Role | Applies to |
| --- | --- | --- |
| `structure` | coordinate error, confidence-weighted | planner |
| `timing`, `velocity` | dt and speed | planner |
| `magnitude` | regularizer only | planner |
| `generative` | latent diffusion/flow objective | drawer |
| `perceptual` | feature-space similarity | drawer |
| `identity` | cosine to reference embedding (frozen encoder) | drawer |
| `structure_obedience` | re-detected vs input structure | drawer |

Anything else must be registered with a role and approved. A lint test fails the build if an unregistered loss, or a full-frame pixel-regression loss, touches drawer output.

## 9. Evaluation (build before any model)

All metrics on held-out windows, reported **per step of rollout**, per behavior, and per motion bin, with fixed seeds, saved as JSON plus auto-rendered contact sheets/videos. **A human must look at the sequences.**

- **Structure error:** mean keypoint error normalized by torso length (face landmarks by inter-ocular distance); **angular error** of per-step displacement (magnitude-weighted); camera and subject scored **separately**; moving-region IoU.
- **Motion magnitude ratio:** predicted/real mean speed. **Freeze detector only**, never a pass criterion.
- **Frozen rate** (on bins with real motion: fraction of rollouts below the real 5th percentile of speed) and **hallucinated-motion rate** (on still bins: fraction above the real 95th percentile).
- **Identity:** cosine similarity (frozen vision encoder) to the reference, over steps. **Structure obedience:** re-detection error between generated and input structure.
- **Flicker:** high-frequency temporal energy vs real footage of the same window. **Sharpness ratio:** Laplacian variance vs real, to catch accidental sharpening/blur.
- **Ghosting/cross-fade score:** for generated frame t, compare against alpha-blends of frames t−k and t+k; flag cases a blend explains as well as real footage does. Flags "one image dissolving into another."
- **Perceptual distance (LPIPS) vs real** where a real future exists. Avoid FVD at this data scale.
- **Baselines (always run):** hold-last-structure, constant-velocity structure extrapolation (both rendered by the same drawer), copy-last-frame, and an evaluation-only flow-warp reference.

## 10. Required tests

1. **Loss lint** (fails on seeded pixel-regression loss).
2. **Extractor validation:** jitter, dropout rate, camera/subject separation on clips with known motion.
3. **Normalization/flip round trip** for structure.
4. **Reference-leak test:** no reference within N frames of its target; no identity or clip overlap across splits.
5. **Metric sanity:** a static video must be flagged frozen; a cross-fade of two real frames must score high on ghosting; a perturbed real video must score near real on identity.

## 11. Proposed layout

```
legacy/                    # quarantined old code
sectvid/{data,structure,planner,drawer,memory}/
sectvid/eval/{metrics,baselines,report}.py
configs/*.yaml             # every guess value lives here
tests/
reports/                   # JSON + contact sheets per run
```

## 12. Stages, tasks, gates

| Stage | Tasks | Gate | Kill / stop |
| --- | --- | --- | --- |
| **0** | Metrics, baselines, ghosting score, loss lint, extractor validation, tests 1-5 | Baselines run end to end; all tests pass | Extractor too noisy to serve as ground truth |
| **1** | Windowing, motion bins, labels recorded, structure cached, splits | Bins populated; no leakage | n/a |
| **2** | Planner ("pencil test"; render stick-figure animations) | Beats constant-velocity on endpoint and angle, camera and subject separately | No better than constant velocity after data/loss fixes |
| **3** | Drawer ceiling: **true** structure → frames | Identity and structure obedience pass; flicker acceptable | Drawer ignores structure or loses identity |
| **4** | Joint rollout to target horizon | Identity and obedience stable over horizon; ghosting clean; beats baselines | Drift outpaces baselines |
| **5** | Mushroom-body ablation | Beats plain baseline | Fails → delete |
| **6** | Label-conditioned prediction (label dropout), then more subjects | Gate-4 metrics hold | n/a |

## 13. Failure catalog

| Symptom | Likely cause | Check |
| --- | --- | --- |
| Subject barely moves | "Don't move" optimum, static-heavy sampling | frozen rate, bin quotas, direction error |
| Right amount of motion, wrong direction | Magnitude-only supervision | angular error |
| Dissolving/ghosted frames | Blending, or drawer copying a reference | ghosting score, reference-leak test |
| Identity drifts over horizon | Conditioned on generated frames, or weak reference | identity curve vs step |
| Flicker | Independent frames | flicker metric, then generated-frame context |
| Plan looks hedged/mushy | Mean collapse | switch to multimodal head |
| Drawer ignores pose | Weak conditioning | structure obedience |

## 14. Rules for builders

Do not change metrics, gates, or the loss registry without approval. Report negative results and **stop at a failed gate**. Put every guess value in config. Never add a loss outside the registry. Record seeds and versions. If the spec is silent, ask.

## 15. Open decisions

Number of distinct identities in the data; whether hands are tracked; pretrained drawer choice; whether flicker forces generated-frame context; window length and resolution.

## 16. Evidence base (what supports what)

- **FAR** (Gu et al.): long-sequence training beats test-time extrapolation (FVD 396 vs 34); stochastic clean context improved FVD 399→347; local context saturates near 8 frames.
- **NOVA** (Deng et al., ICLR 2025): removing temporal-causal modeling cut dynamic degree 23.27→11.38 while total score barely moved (75.84 vs 75.38), i.e., aggregate scores hide frozen motion; flow-magnitude motion score as conditioning.
- **MobileI2V** (Zhang et al.): same model at motion score 2 vs 5 moved dynamic degree 0.157→0.495 with other metrics near flat; flow-based data filtering; heavy compression blurs faces.
- **Not supported by these papers:** the planner/drawer split, structure-space autoregression, structure obedience, the mushroom-body roles. These are the owner's lessons and this design's hypotheses, to be validated by the gates.