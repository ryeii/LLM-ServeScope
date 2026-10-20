# Production GPU timing model: method and evidence

Current production uses shape-informed functions selected from F0–F9, with a
separate function per operator/attention phase and per historical precision cohort.
Read the [full cross-validated accuracy report](accuracy-2026-10-19/README.md) for
all residual graphs, selected families, exact data coverage, and limitations.

- **599,451** imported compatible observations: KAIST BF16 (9,411), Vidur FP16
  (590,040); five GPU families. Historical profiler environments are not pooled.
- **Five grouped outer folds**, with function/transition-scale selection confined
  to **three inner training folds**. Every eligible target receives exactly one
  outer held-out prediction. Identical executed geometry stays together across
  GPU/model/TP; embedding uses token count and hidden width.
- Select the fewest parameters within one percentage point of best inner MAPE.
  Refit the final selected functions on **all data in their cohort** for deployment.
- KAIST embedding uses two independent nonnegative affine functions, switched
  at the source-defined 16-token boundary. Other selected forms include additive,
  smooth overlap, tile-wave, bounded-shape and cache-capacity terms.
- Runtime receives coefficients and coverage metadata only. No timing tables,
  exact-match anchors, device-ID corrections, NumPy or SciPy are required.
- FP16 selects the Vidur cohort; BF16 selects KAIST. Other precisions extrapolate
  KAIST. Missing FP16 operator groups explicitly extrapolate KAIST functions.
  Unknown physical L2 inputs use an exported additive fallback for F7.
- Physical L2 capacity is separate from the user-reserved KV SRAM budget.
  The runtime and offline seven-feature formulas are parity-tested.

Mean absolute operator-time error is **14.86%**, median **9.54%**, P95 **49.52%**
in grouped nested CV. These scores evaluate the selection/fitting procedure under
recorded profile conditions. Earlier data exploration informed the family library,
so they are exploratory rather than a new confirmatory test. A separate
leave-one-GPU-out audit excludes each target GPU from fitting and selection.
There is no universal new-GPU error guarantee.

SimPy serving throughput, request latency, memory traffic, power, network and energy
remain behavioral outputs with separate assumptions. Operator residuals do not
calibrate those entire systems. Specialized MoE/MLA/hybrid/MTP profiles and mixed-phase
attention remain outside the imported dense cost contract and are listed explicitly.

The former F0 model and its original validation remain archived in
[previous calibration documentation](accuracy-2026-10-19/previous-calibration.md)
and [baseline coefficients](fit-study-2026-10-19/baseline-gpu-fit.json).
Historical baseline scores should not be attributed to the new model.

Implementation: [runtime functions](../backend/gpu_functions.py),
[runtime integration](../backend/calibration.py),
[coefficient artifact](../backend/gpu-fit.json), and
[training/evaluation pipeline](../integration/production_fit).

Credit: CASYS Lab / KAIST for LLMServingSim 2.0 and its profiles; Microsoft Vidur
contributors for published FP16 profiles, under retained MIT licenses. The fitted
functions, selection/evaluation, serving UI and integration are ServeScope
extensions by [ryeii (Ryan Z An)](https://github.com/ryeii).
