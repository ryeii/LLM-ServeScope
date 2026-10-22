# Accuracy and validation

ServeScope has fitted **GPU operator timings** and a separate **prefix-reuse
policy approximation derived from published request traces**. Storage and
transfer timings use configured assumptions. The request traces contain no
measured device timings, so they provide no SSD, GDS or host–GPU timing error bound.
These evidence classes do not combine into a serving-accuracy guarantee.

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

## Storage and transfer service — configured assumptions

SSD read/write, DRAM/host–GPU, explicit HBM-copy and GDS service times use
configured hardware rates and startup latencies. **No measured timing dataset or
held-out device-service error bound is provided for these paths.** Mooncake's
published request traces are workload inputs, not storage performance profiles.

[Published-trace coverage and storage equations](storage-model.md) ·
[GDS and Dynamo-style assumptions](dynamo-alignment.md).

## Prefix-hit forecasts — trace-policy replay

These fits target **1,260 LRU trace-replay curve observations**, not measured
hardware cache hits. Nested validation withholds complete temporal regions,
including their overlapping windows and capacities.

| Mooncake trace | Mean absolute hit-rate error | P95 absolute error |
| --- | ---: | ---: |
| Conversation | 1.27 percentage points | 3.78 percentage points |
| Tool/agent | 3.79 percentage points | 13.75 percentage points |
| Synthetic | 0.53 percentage points | 3.29 percentage points |

Runtime hits are determined by the executed cache state, not these forecasts.
Single-LRU replay error does not validate every multi-tier spill policy or an
unseen workload. [Hit-model methods and evidence](trace-reuse/README.md).

## Serving results and remaining validation gaps

A separate nine-case integration recheck reports **4.83% mean throughput
difference** from native simulator references. This is a simulator-to-simulator
comparison, not measured physical serving accuracy. See the
[serving integration recheck](accuracy-2026-10-19/README.md#serving-integration-recheck-separate-evidence).

- **Active decode KV offload** adds serialized transfers and staging to HBM-resident
  GPU reference costs. The composition, streaming-kernel changes and overlap are
  not calibrated end to end. [Active-KV model and limitations](active-kv-model.md).
- **Host DRAM, explicit HBM copies and network paths** use configured effective
  rates/latencies. They have no general held-out target-server accuracy guarantee;
  kernel HBM behavior and explicit copies do not share a detailed bandwidth model.
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
Trace-policy replay, prefix-reuse fitting, GPU validation, tier/serving modeling and
ServeScope integration: [ryeii (Ryan Z An)](https://github.com/ryeii).
[Retained licenses and source notices](../THIRD_PARTY_NOTICES.md).
