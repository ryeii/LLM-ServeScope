# Accuracy and validation

ServeScope has separate evidence for **GPU operator timing**, **SSD/file-path
service** and **prefix-hit forecasts**. Their errors describe different targets;
they do not combine into a single accuracy guarantee for serving throughput,
request latency, energy or a new hardware configuration.

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

## SSD/file-path service — Mooncake benchmark integration

ServeScope collected **768 local I/O observations** using Mooncake’s archived,
unmodified `DiskHashTable` benchmark class: 384 reads and 384 writes, 16 KiB–2 MiB,
serial queue depth one, warm buffered reads and fsync writes on a development Mac.
This measures the local file path, not isolated physical SSD service. Mooncake’s
published aggregate SSD-offload serving results are separate evidence, not these
per-size training samples.

Eight outer folds withhold every measurement of one I/O size; inner folds select
the function using only the remaining sizes. The following numbers and figure
use those outer held-out predictions. Bandwidth is derived from the same latency
function as bytes/time, so its error is not independent validation.

| Operation | Selected function | Latency MAPE | Latency P95 error | Bandwidth MAPE | Bandwidth P95 error |
| --- | --- | ---: | ---: | ---: | ---: |
| Read | Continuous two-slope | 16.37% | 53.09% | 29.42% | 113.16% |
| Write | Affine | 9.97% | 27.90% | 12.04% | 38.70% |

![Local file-path latency and derived bandwidth versus nested held-out predictions](https://ryeii.github.io/LLM-ServeScope/storage-study/io-fit.png?v=17c686000e4f)

Warm-cache read bandwidth can exceed physical SSD bandwidth, and the read-error
tail is substantial. These results do not validate cold-device, higher-queue-depth,
cross-machine or end-to-end active-KV offload performance. The fitted path is used
only when selected and the scenario actually performs SSD I/O; the default
HBM-only active KV with prefix retention disabled adds no SSD service time.

[Storage evidence: coverage, fitting, validation and raw observations](storage-2026-10-20/README.md)
· [Deployed storage coefficients](../backend/storage-fit.json)

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
unseen workload. [Hit-model methods and evidence](storage-2026-10-20/README.md#fitted-hit-rate-functions-and-their-different-target).

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
  assumptions. GPU operator or local file-I/O errors do not validate those outputs.
  GPU energy excludes storage/host/network equipment power.

[How request timestamps produce throughput and latency](how-results-are-calculated.md)
· [Behavioral methodology](methodology.md)
· [Main README and limitations](../README.md#limitations-behavioral-not-mechanistic-simulation)

## Credits and provenance

GPU profiles: [CASYS Lab at KAIST / LLMServingSim 2.0](https://github.com/casys-kaist/LLMServingSim)
and [Microsoft Vidur](https://github.com/microsoft/vidur).
Storage benchmark code and request traces: [Mooncake contributors](https://github.com/kvcache-ai/Mooncake).
Local I/O measurements, fitted functions, validation, tier/serving modeling and
ServeScope integration: [ryeii (Ryan Z An)](https://github.com/ryeii).
[Retained licenses and source notices](../THIRD_PARTY_NOTICES.md).
