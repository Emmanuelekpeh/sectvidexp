# Open Design Axes: Resilience, Self-Organization, and Hierarchy

**Status:** proposal for architecture work; not yet an approved change to the v4 specification, stage gates, loss registry, or implementation.

The current design specifies a structure-space motion planner followed by a full-frame drawer, and requires autoregressive rollout evaluation. It does not yet explain how the system should organize motion over time, discover reusable structure, or behave when observations and predictions become unreliable. These are foundational questions to resolve before treating the Stage 2 planner architecture as settled.

## 0. Constraints from the v4 spec (existing principles)

The v4 specification and handoff already encode principles that bound these axes. Candidate requirements must stay compatible with them; where the spec is silent, the gap is surfaced rather than filled.

- **Resilience.** Spec 6.2 already requires carrying per-point confidence into losses and validating the extractor (Stage 0). Spec 6.3 already specifies the clean-context mechanism (with probability ~0.1 feed real structure, flagged by an input bit and excluded from the loss, per FAR); recovery-from-evidence tests should build on that mechanism, not invent a parallel one. Spec 9's frozen rate and hallucinated-motion rate already detect two failure directions (too little motion on moving bins, too much on still bins). Spec 12/14: any new metric or gate requires owner approval; nothing in this document authorizes one yet.
- **Self-organization.** Spec 7 already codifies the baseline discipline for any associative module: it must beat a plain baseline (nearest-neighbor retrieval, small learned classifier) on held-out data or be removed; Stage 5 is the scheduled venue for that ablation. Learned motion groupings should be evaluated under the same discipline to avoid double-scheduling the same test in two places. Spec 6.1's leakage rules (held out by clip and identity) apply to any discovery process: fit on training identities only. Spec 6.1 also records that labels do not fix mixed-action clips, so the spec already concedes hand-authored organization is incomplete; bins are sampling strata, not learned structure.
- **Hierarchy.** The spec's cel-anim analogy (Section 4) is already hierarchical: keys, breakdowns, inbetweens, with "where" (arc) and "when" (spacing, holds) planned separately; frame-level structure + dt is that where/when level. This axis therefore extends the design with a slow-timescale level; it does not replace the flat structure output. The banned list still applies at every level: a high-level plan must not be realized by interpolating or averaging between key structures (inbetweeners redraw the form; Spec 4), and pixel/latent carryover remains forbidden. Lesson 3 (Spec 3) warns that a state describing appearance may omit what is needed to move it: a slow-timescale state that lacks direction and timing information would only add mush.

## 1. Resilience

### Working meaning

The system should recognize when its input or internal rollout is unreliable, limit the damage, and recover when useful evidence returns. Resilience is not merely smooth output: a smooth but confidently wrong trajectory is a failure.

### Candidate requirements

- Preserve uncertainty from extraction and prediction through the planner. Missing or low-confidence pose, face, or camera data must not silently become a confident zero-motion target.
- On unreliable observations, expose uncertainty or abstain from a high-confidence update; do not invent a specific recovery behavior until it is tested.
- During autoregressive rollout, monitor for accumulating structure error, implausible velocity/timing, identity drift, and camera/subject confusion.
- Test recovery after evidence returns, including whether the planner returns to a plausible trajectory instead of remaining locked to an earlier error.
- Keep resilience mechanisms separable from the base planner so their value can be ablated against the same data and baselines.
- The data layer already contains a silent-uncertainty defect, measured on the current checkout: camera-affine estimation fails (confidence 0) on 5.5% of all cached subject steps, and up to 30-62% of steps on the worst (fast-action) clips. The camera-corrected displacement path applies the fallback identity transform without checking `camera_conf`, so on those steps raw image motion (camera + subject) is treated as camera-corrected subject motion. This must be resolved (fall back to raw motion, or mark such steps unobserved) before bins and baselines are treated as clean ground truth.

### Evaluation hypotheses

1. **Corruption response:** progressively mask or perturb observed keypoints and camera estimates; measure error, uncertainty calibration, false confidence, and motion-direction degradation.
2. **Dropout recovery:** hide structure for controlled intervals, then restore it; measure steps to recover trajectory and endpoint accuracy.
3. **Rollout perturbation:** perturb planner inputs or predictions at selected steps; compare subsequent error growth and recovery with the unmodified rollout.
4. **Confidence calibration:** group predictions by reported confidence and check whether lower confidence corresponds to higher held-out error. A confidence score that does not predict error is not a useful resilience signal.
5. **Low-confidence camera steps:** on steps where the camera affine has zero or low confidence, compare subject-motion and bin statistics with and without the correction, and record how often the correction flips a bin assignment. This is a Stage 1 data-quality check on the existing caches, not a model test; the handling decision (raw motion vs unobserved step) belongs to the owner.

These are proposed tests. Their corruption levels, acceptance thresholds, and whether they gate Stage 2 remain undecided.

## 2. Self-Organization

### Working meaning

The model should be able to discover recurring motion patterns and useful temporal units from observed sequences, rather than depending entirely on hand-authored bins or behavior labels. This does not mean removing explicit supervision or the existing data-quality gates.

### Candidate requirements

- Separate data-quality labels (observed/unobserved, extractor reliability) from learned motion groupings. A learned cluster must not relabel failed extraction as stillness.
- Discover reusable motion units or patterns from structure trajectories, potentially including poses, transitions, holds, and repeated subsequences.
- Make discovered groups inspectable through representative sequences and nearest examples; do not rely on opaque cluster IDs alone.
- Evaluate whether the discovered organization improves prediction or retrieval on held-out identities and clips, compared with simple unsupervised and nearest-neighbor baselines.
- Prevent data leakage: fit organization on training identities only, then assign or evaluate held-out identities without refitting on their future windows.

### Evaluation hypotheses

1. **Repeatability:** rerun discovery across seeds or data resamples; compare cluster stability and nearest-neighbor consistency.
2. **Usefulness:** compare structure prediction, retrieval, and rollout results with and without discovered units.
3. **Coverage:** report unassigned, rare, and low-confidence sequences instead of forcing every window into a group.
4. **Generalization:** evaluate whether units transfer to unseen identities and clips, and inspect representative held-out sequences.

The representation, discovery method, number of units, and pass criteria are open decisions. Existing still/slow/fast bins are sampling strata, not evidence that the model has self-organized motion concepts.

## 3. Hierarchy

### Working meaning

Motion should be representable and predictable at multiple temporal scales: a broad movement or phase, its local pose sequence, and frame-level timing. Hierarchy should help long-horizon coherence while preserving per-frame direction and timing accuracy.

### Candidate requirements

- Keep the current frame-level structure output and autoregressive evaluation as the reference baseline.
- Investigate an explicit slow-timescale state or segment plan alongside local pose/timing prediction. The high-level state may describe a phase, motion unit, or key-pose sequence; its semantics must be established empirically.
- Permit variable-duration phases and holds; do not force fixed-length action segments.
- Ensure the lower level can express corrections and detail without violating the high-level plan, and measure when these levels disagree.
- Avoid introducing behavior labels as model inputs in the initial prediction-first stages; this remains governed by the existing specification.
- A slow-timescale state must carry direction and timing information, not only appearance or magnitude (spec lesson 3: appearance state may omit what is needed to move it).
- Phase transitions must be produced by the low-level planner redrawing structure at each step; no mechanism in this axis may realize a phase by interpolating or averaging between key structures (spec section 4 banned list applies at every level).

### Evaluation hypotheses

1. **Long-horizon coherence:** compare flat and hierarchical planners at matched data, compute, context, and rollout horizon; report endpoint, direction, timing, and per-step structure error.
2. **Temporal-scale ablation:** remove the high-level state or local planner in turn to show which level contributes measurable value.
3. **Boundary behavior:** inspect and score transitions between phases, including variable duration, pauses, reversals, and recovery from timing errors.
4. **Baseline comparison:** a hierarchy must beat the existing hold-last and constant-velocity baselines, and the flat planner, on held-out rollouts; added complexity alone is not a success criterion.

The hierarchy's levels, state representation, training objective, and integration stage are undecided. Do not assume a particular biological hierarchy or reuse the legacy KC/MBON module without its existing ablation requirement.

## Observed evidence (measured 2026-10-05, current checkout)

Facts from the current data path that these axes must absorb; all are reproducible from `reports/` and the structure caches.

1. **Camera fallback frequency.** Across all 131 cached clips, 1,775 of 32,389 adjacent-frame transitions (5.5%) have `camera_conf == 0`; among transitions with usable subject poses (both frames valid and at least four shared keypoints), 690 of 21,928 (3.15%) have zero camera confidence. The affected share of usable subject steps reaches 62.4% in `1-hour-karate-workout-video_clip_0063`, 57.5% in `Merantau_clip_0344`, and 51.9% in `Merantau_clip_0311`. The camera-corrected displacement path does not check `camera_conf`, so the fallback identity transform is applied as if trusted. The 5.5% figure is over all adjacent-frame transitions; it should not be described as the usable-subject-step rate.
2. **Bin boundaries sit inside the noise floor.** Still (< 0.02 normalized/s ≈ < 0.43 px/frame at 512 px) and most of the slow bin (up to 0.15 normalized/s ≈ 3.2 px/frame) are below the measured extractor noise floor (2 px jitter probe, production-smoothed path: p50 4.1 px/frame, p90 21.1 px). Per-frame motion in still/slow windows is not reliable signal; window-scale net displacement is the usable statistic. The still-calibration script has never been run, and its contact sheet is currently broken (it loads 64 frames but indexes them by full-clip frame position, so slowest windows starting after frame 64 render empty or wrong strips); fix it before using it for the owner's still-bin decision.
3. **Stage 0 reference selection produced no references.** The Stage 0 baseline report used 0 reference frames for all four sampled clips because the available-reference pool is built from window-local frame indices and then filtered for indices >= the window length (always empty). Identity cosine values from Stage 0 are therefore unmeasured; repair the selection and regenerate before relying on identity metrics.
4. **Pre-change bin report is stale.** The earlier Stage 1 report (304 still / 324 slow / 1,217 fast) predates the unobserved separation; its 304 "still" windows were the missing-pose artifact. The current split (0 / 212 / 1,094 / 539, same total 1,845 windows) supersedes it.

## Design questions to resolve

1. What uncertainty should each stage expose: extractor confidence only, planner uncertainty, or both?
2. What should the planner do when uncertainty rises: continue with calibrated uncertainty, preserve a state, request/rely on new evidence, or another behavior? Compare options experimentally.
3. What is the smallest useful temporal hierarchy, and how are its units discovered and named for inspection?
4. Which of these tests should be required gates, and at which stage? Thresholds must be calibrated from held-out data rather than guessed into passing.
5. How can these additions be tested without weakening current requirements for motion direction, full autoregressive rollouts, identity stability, and comparison to simple baselines?

## Suggested sequence

1. Resolve the current Stage 1 data and identity gates; these design questions do not authorize bypassing them.
2. Before committing to a Stage 2 planner, write a small set of falsifiable prototypes: a flat planner, an uncertainty-aware variant, and a hierarchical variant. Keep the evaluation conditions and data splits identical. Sequencing proposal: the Stage 2 gate (spec section 12) attaches to the flat pencil-test planner; the uncertainty-aware variant is its direct comparison; the hierarchical variant can follow once the flat planner has passed or failed the gate, so the gate outcome is not entangled with a hierarchy choice.
3. Add corruption, recovery, temporal-unit inspection, and long-horizon comparisons to reports first; choose gates only after the measures are demonstrated and thresholds are justified.
4. Retain a new mechanism only when its ablation improves the target behavior without trading away direction, endpoint accuracy, or rollout stability.

Until the owner approves specific changes, this document is a design agenda only. The v4 specification remains authoritative for current implementation, losses, and stage gates.
