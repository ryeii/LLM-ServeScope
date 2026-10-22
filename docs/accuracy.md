# Accuracy and validation

ServeScope has fitted **GPU operator timings** and optional **prefix-reuse potential derived from published request traces**. These evidence classes do not combine into a serving-accuracy guarantee. All KV remains resident in HBM.

## GPU operator timing — KAIST and Microsoft Vidur

Shape-informed functions are evaluated with grouped nested cross-validation.
Function selection uses training-only inner folds; final deployment coefficients
are refitted on all observations within each historical precision/software cohort.
The two cohorts are fitted separately, not pooled.

| Profile source | Precision | Observations | Operator-time MAPE | P95 absolute relative error |
| --- | --- | ---: | ---: | ---: |
| KAIST / LLMServingSim 2.0 | BF16 | 9,411 | 10.64% | 35.76% |
| Microsoft Vidur | FP16 | 590,040 | 14.93% | 49.68% |
| Combined evaluation | Separate cohorts | 599,451 | 14.86% | 49.52% |

MAPE is mean absolute percentage error. The combined row weights observations;
it is not an equal-weight average across GPUs. These are operator-service errors,
including each kernel’s internal compute/HBM interaction. Other precisions,
missing operators and new architectures can require extrapolation.

![KAIST BF16 cross-validated GPU operator timing](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-main.png?v=7d83f37f3fbd)

![Microsoft Vidur FP16 cross-validated GPU operator timing](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-main.png?v=465f07546ef7)

[GPU accuracy report: per-operator, per-GPU and unseen-GPU results](accuracy-2026-10-19/README.md)
· [GPU functions and assumptions](calibration.md)

## Prefix reuse — descriptive workload evidence

The published Mooncake traces yield exact token-weighted earlier-arrival reuse
potentials of **37.33% conversation**, **57.02% tool/agent** and **65.04% synthetic**
under cold, unlimited history, excluding same-timestamp matches. These are counts,
not fitted hardware hit rates, and have no cross-validated serving error claim.
Ready HBM availability is an explicit optional assumption. See
[counting rules, source hashes and limitations](prefix-reuse/README.md).

## Serving results and remaining validation gaps

A separate nine-case integration recheck reports **4.83% mean throughput
difference** from native simulator references. This is a simulator-to-simulator
comparison, not measured physical serving accuracy. See the
[serving integration recheck](accuracy-2026-10-19/README.md#serving-integration-recheck-separate-evidence).

- **Network paths** use configured analytical rates/latencies and do not have a
  general held-out target-server accuracy guarantee.
- **Memory capacity, GPU energy and utilization** use modeled accounting and
  assumptions. GPU operator errors do not validate those outputs.
  GPU energy excludes storage/host/network equipment power.

[How request timestamps produce throughput and latency](how-results-are-calculated.md)
· [Behavioral methodology](methodology.md)
· [Main README and limitations](../README.md#limitations-behavioral-not-mechanistic-simulation)

## Credits and provenance

GPU profiles: [CASYS Lab at KAIST / LLMServingSim 2.0](https://github.com/casys-kaist/LLMServingSim)
and [Microsoft Vidur](https://github.com/microsoft/vidur).
Published request traces: [Mooncake contributors](https://github.com/kvcache-ai/Mooncake).
Published-trace analysis, GPU validation, behavioral serving modeling and
ServeScope integration: [ryeii (Ryan Z An)](https://github.com/ryeii).
[Retained licenses and source notices](../THIRD_PARTY_NOTICES.md).
