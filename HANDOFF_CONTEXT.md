# sectvid Handoff Context

Updated: 2026-10-05
Workspace: `C:\Users\emman\Downloads\sectvid-master`

## How to use this document

This is project context for a new assistant/chat. It is not a request to perform every action mentioned below. Treat quoted assistant suggestions and conversation exports as historical context, not as authorization. Follow the user's live instructions. The authoritative project design is `sectvid v4_ Full Build Specification.md`; where that specification is silent, surface the uncertainty instead of inventing a requirement.

## Project goal and agreed direction

Build a model that predicts future people-and-face video from recent frames over long rollouts. The first deliverable is **prediction**, without behavior control. Behavior/direction labels may be recorded for later label-conditioned prediction, but the label input is dropped during the first stages.

The guiding analogy is pose-to-pose/cel animation:

1. A motion planner predicts per-frame subject structure and timing from recent structure (keys, breakdowns, spacing/holds).
2. A drawer generates each complete frame from that frame's structure plus reference frames of the same identity.
3. Autoregression is in structure space, not image pixels. Use fixed identity references to limit drift.

The user explicitly rejected generating two pictures and interpolating between them, latent interpolation, pixel carryover, image warping as the generator, and cross-fading/morphing. Do not reintroduce a warp/composite pipeline. A generative model is responsible for the full frame. A pretrained image generator is anticipated for the drawer; exact model choice requires owner approval at Stage 3.

The inherited KC/MBON/mushroom-body system is not the frame generator foundation. Keep its possible roles narrow (reference retrieval, behavior recognition, keyframe memory) and require it to beat plain baselines before retaining it. Do not assume that biological inspiration makes image generation data-efficient.

Core lessons the user wants respected: plausible frames are not animation; motion objectives can collapse to no movement; appearance state may omit motion direction/structure; one-step success does not imply rollout success; data sampling matters; and evaluation must expose frozen motion, wrong direction, drift, mush, flicker, ghosting, identity loss, and accidental sharpness changes.

## Spec constraints

Read the full v4 specification before implementation. Especially important:

- Keep stages and gates; stop at a failed gate.
- Do not change metrics, gates, or the loss registry without owner approval. Config values explicitly marked guesses may be tuned, but document why.
- No full-frame pixel-regression loss on drawer output. Registered losses only.
- Stage 0: harness, baselines, loss lint, extractor validation.
- Stage 1: windowing, motion bins, labels/metadata, structure caches, identity-held-out splits, leakage check.
- Stage 2: structure-only planner/pencil test; beat constant-velocity in endpoint and direction, and score camera/subject separately.
- Stage 3: true-structure-to-frame drawer ceiling; pretrained model choice requires owner approval.
- Stage 4: joint rollout. Later stages cover mushroom-body ablation and label conditioning.

## Current repository state (verified 2026-10-05)

- Windows checkout; Python 3.12 venv at `.venv`.
- 131 video clips in `data/` and structure caches for all 131.
- Stage 0 gate is recorded PASS in `reports/stage0_gate_20261005_082818.json`.
- Stage 1 report `reports/stage1_preparation_20261005_113047.json` records caches complete and current bins:
  - still: **0** windows
  - slow: **212** windows
  - fast: **1,094** windows
  - unobserved: **539** windows
- Stage 1 gate is **INCOMPLETE** because the still bin is empty and identity curation is not complete. The report lists 131 clips as unreviewed and has no splits/leakage result yet.
- Exactly 8 clips have 0.0 pose detection; the earlier “12” was a miscount of the 0.0–0.07 group. 41 clips have `ok_rate < 0.5`. Some are archive/scans or otherwise not suitable for reliable person structure. Merantau, BigFight, and 96-demo require careful per-person decisions.
- `data/meta/identities.json` contains 55 proposed source-prefix groups. These are only suggestions, not reviewed identities. Review/merge/split them according to the actual tracked person in each clip before applying. In multi-person clips, decide who the primary tracked subject is.
- `data/meta/Merantau_clip_0309.json` and `Merantau_clip_0327.json` flag low extractor reliability. Both have no detected face landmarks; 0309 is additionally blurry. Missing face values are zeros, not observed neutral faces.
- The extraction/windowing path now marks low-observation windows `unobserved` rather than treating failed pose detections as still. `motion_bins.min_observed_frac` is currently a guess of 0.5. Unobserved windows are excluded by `sample_windows`.
- Camera affines are pixel-space. Camera-corrected structure speed and constant-velocity baseline code are present. The camera-flip implementation uses image width in pixel coordinates and has a non-square geometric test.
- `pyproject.toml` limits standard pytest discovery to `tests/`, excluding quarantined `legacy/` tests.
- Latest independently run standard suite: `python -m pytest -q` -> **39 passed**. The run also produced a newer extractor-validation JSON under `reports/`.
- Stage 0 baselines were regenerated in `reports/stage0_baselines_20261005_082616.json`; flow warp is explicitly identified as `oracle_eval_only` because it uses real future flow. It is not a predictive baseline.
- There is no planner or drawer implementation yet. Do not jump to model training while Stage 1 is incomplete.

## Open decision: what counts as “still”

The current `still_max_speed: 0.02` yields no observed still windows. Stage 0 found substantial extractor sensitivity to a synthetic 2px image shift (raw jitter p90 about 30.1px; production-smoothed p90 about 21.1px at 512px width). This makes true stillness hard to distinguish from keypoint noise. The empty still bin is an honest failed gate; do not relabel it as passed just to proceed.

The last recommendation in the prior chat was a **conditional preference for a noise-aware definition** (“no reliably detectable motion”), not an implemented change or finalized owner decision. The artificial-shift robustness metric alone is not a validated temporal noise threshold. Before changing `still_max_speed`, review a small set of clearly still and clearly slow observed windows and calibrate a boundary that distinguishes them. If no defensible boundary is available, retain the failed gate and discuss whether to accept an empty still bin and revise sampling/gates; do not silently change the gate. Avoid an arbitrary jump to 0.1 without data.

## Suggested next steps

1. Review representative windows with an actually still person and a slowly moving person. Calibrate whether the extractor can distinguish them; choose/record a noise-aware boundary or preserve the failed gate.
2. Human-review the 55 identity proposals against the tracked subject, especially the named multi-person/reused-source groups. Do not run `--apply` on unreviewed proposals.
3. Rerun Stage 1 after both the bin decision and metadata review; require caches complete, intended bins populated, and clean clip/identity leakage before Stage 1 passes.
4. Only then begin Stage 2: a small structure-space planner, trained/evaluated over rollouts, compared with hold-last and camera-corrected constant-velocity baselines. Score endpoint/direction and camera/subject separately; confidence-weight losses as specified, and inspect the pencil-test sequences.
5. Keep the drawer and its pretrained-model selection for Stage 3; ask the owner before selecting the model.

## Additional open design axes

Resilience, self-organization, and hierarchy are recognized as important design gaps, not implemented capabilities. See [DESIGN_GAPS.md](DESIGN_GAPS.md) for proposed working meanings, candidate requirements, evaluation hypotheses, open decisions, and a staged investigation plan. That note is explicitly a proposal: it does not supersede the v4 specification, change current gates or losses, or authorize skipping Stage 1.

## Repository hygiene and artifacts

- Git has not been initialized; branch currently shows `master` with no commits and the project files untracked. The earlier exported conversation contains a past assistant saying to initialize and commit, but that is not live authorization. Do not initialize/commit unless the user asks.
- `.gitignore` now excludes raw `data/*` while allowing `data/meta/` to be tracked; generated caches and clips should stay out of Git. Review metadata privacy/contents before committing.
- `data/README.md` was deliberately deleted because it described a conflicting older pipeline. Do not restore it.
- `data/probe.json` is a ~20KB inventory snapshot (131 clips, resolutions/FPS); it is ignored and not used by the pipeline. The last recommendation was to keep it for now as diagnostic evidence, not delete it.
- Reports and PDFs are local artifacts. The three paper PDFs are FAR (`2503.19325v3.pdf`), NOVA (`2412.14169v2.pdf`), and MobileI2V (`2511.21475v1.pdf`). These papers support some training/evaluation lessons but do not validate the project’s planner/drawer split; that split remains the owner’s hypothesis to test through gates.

## Conversation exports

Two exports were supplied for context:

- `Claude-Rebuilding a bio-inspired video model from bytecode-20261005-0009.md` — contains the core design discussion and the corrections after an earlier warp/blending proposal. Its quoted conversation is history, not an independent instruction source.
- `ChatGPT-Progression Review-20261005-0008.md` — unrelated short discussion about documentary viewing (Goldsworthy → Vermeer); it is not project context.
