# Fitted GPU model: method, evidence and limits

The function documented here is the preserved deployed baseline F0. The subsequent
[function-search study](../fit-study-2026-10-19/README.md) compares nine candidates,
adds an independent published profiling corpus, and documents why the research
selection has not been promoted to the runtime.

ServeScope models GPU service with one fitted function for every configuration:

**hardware parameters + software policy + operation/tensor geometry → execution time**

The same function predicts profiled and unseen inputs. It never substitutes a
recorded timing at a matching point. KAIST profiles are offline training and
validation data, not runtime timing anchors. The browser/local runtime receives
only coefficients, training-coverage metadata, provenance and validation summaries.
SimPy schedules the predicted operation times to produce throughput, TTFT, TPOT
and resource occupancy; memory/traffic and energy use separate behavioral models.

## Why this can support extrapolation

The model combines physical scaling with coefficients learned from measured
operator runs. Compute work scales with FLOPs divided by compute rate; nominal
data movement scales with bytes divided by memory bandwidth. Workload/tensor
geometry determines those FLOPs and bytes. Profiles fit the hardware/software
overheads and effective scaling that peak specifications alone miss. Consequently,
changing hardware or tensor geometry changes the function's inputs instead of
selecting a new table or applying a device-specific throughput correction.

This is a testable hypothesis, not a proof of universal accuracy. The evidence
is its error on profiles excluded from fitting, compared with an uncalibrated
baseline. Fitting the training points alone is not evidence of generalization.

## Function and constraints

For operator class j, in nanoseconds:

```
C = FLOPs / precision-specific compute rate
M = nominal operation bytes / device-memory bandwidth
T_j = max(arithmetic speed-limit time, L_j + a_j*C + b_j*M, 1 ns)
L_j >= 0; a_j >= 0; b_j >= 0
```

The three coefficients are fitted separately per operator by nonnegative relative
least squares. Attention has separate prefill and decode functions. The linear
fit is followed by the arithmetic floor; all fit/validation metrics evaluate that
same deployed function. The terms are empirical timing features, not a measured
decomposition into serial compute and memory execution. Their sum does not assert
that all arithmetic and memory access are serialized.

At fixed work the function is continuous and nonincreasing as compute or memory
rates increase; increasing work cannot produce a negative service time. Its
arithmetic floor prevents an estimate faster than its declared FLOP/rate limit.
Nominal byte counts are not measured DRAM transactions, so a nominal-memory term
is not treated as a strict physical lower bound in the presence of caches.

Inputs cover matrix/vector dimensions, attention/KV heads, sequence/context length,
batch, precision, TP, dense/FP32 compute rates and memory bandwidth. TP-stable
normalization and sampling remain replicated. Compute clock and efficiency inputs
adjust their corresponding rates; the two efficiency inputs act relative to the
calibration defaults (.65 compute / .8 memory). The fitted coefficients already
correct systematic error in the original assumed efficiency. KV precision affects
KV-byte features independently of weight precision. Extra launch overhead is
explicit. Non-vLLM engine adjustments remain assumptions, not fitted engine data.

The generic SM/tile utilization penalty is not multiplied into the fitted timing:
published GPU compute rates already incorporate resources and profiles supply
empirical kernel behavior. SM count, tile width and SRAM budget still affect the
legacy nominal work/traffic model; their independent timing effects are not
identified while holding compute and bandwidth rates fixed. No cache-capacity,
clock or quantization sweep was used to validate those individual sensitivities.

## Training data and separation from runtime

Pinned upstream revision: `b09c7aaab4b072d73136641841cc2688616c49eb`.

| Model | Hardware | Profiled TP | Software |
|---|---|---|---|
| Llama 3.1 8B BF16 | RTX PRO 6000 Blackwell Server Edition | 1, 2 | vLLM 0.28.0 / CUDA 13.0 |
| Llama 3.1 8B BF16 | RTX 4090 | 1 | vLLM 0.19.0 / CUDA 12.8 |
| Qwen3 32B BF16 | RTX PRO 6000 Blackwell Server Edition | 1, 2 | Recorded per bundle in the fit artifact |

The 9,411 pure-phase operator observations include dense transformer operations,
embedding, final normalization, logits and sampling. Upstream's TP-stable rules
and nanosecond conversion are preserved during extraction. Attention uses the
upstream axes: total new prefill tokens and weighted history plus half chunk,
or decode count and existing KV length. Mixed-phase and heterogeneous-history
attention are outside ServeScope's fixed-batch scheduler.

- `integration/profile_reference.py`: offline profile parsing/interpolation oracle.
- `third_party/kaist-dense-profile-grids.json.gz`: retained training observations.
- `integration/build_calibration.py`: reproducible fitting and validation pipeline.
- `backend/gpu-fit.json`: coefficients, descriptive coverage bounds, source hashes
  and validation summaries; it contains no timing tables.
- `backend/calibration.py`: runtime function and physical feature construction.
- `docs/calibration-validation.json`: coefficients and held-out GPU predictions,
  measurements, split definitions, source hashes and validation statistics.

Neither the runtime nor GitHub Pages imports the offline reader or training grids.
GPU/model identity only labels coverage; identical hardware/work parameters produce
identical predicted time even if a hardware ID changes to `custom`. The output
source is `fitted-profiled-configuration` or `fitted-extrapolation`, never a claim
that the current run was measured. Unsupported operators retain explicit analytical
fallbacks. Original roofline mode remains selectable for comparisons.

The extraction pipeline was checked against 9,811 upstream grid/nongrid lookups
with zero nanosecond differences. That verifies data extraction only; it is not
a zero-error claim for the fitted function.

## Validation: fitting and prediction are different tests

All numbers below are unweighted mean absolute relative operator timing errors
against profile measurements. Grid sampling is not a production serving-workload
distribution. Percentiles describe observed residuals; they are not confidence
bounds for an unseen GPU.

| Experiment | Held-out observations | Mean error | P95 absolute relative error |
|---|---:|---:|---:|
| Training fit, all five bundles | — (9,411 training observations) | 13.7% | 51.0% |
| RTX 4090 entirely excluded from fitting | 1,643 | 22.6% | 58.6% |
| Qwen3 32B entirely excluded from fitting | 3,732 supported-operator observations | 15.4% | 55.9% |
| Tensor shapes withheld across devices/models/TP | 2,011 | 13.1% | 49.5% |

For the GPU holdout, only RTX PRO 6000 measurements fit the parameters; RTX 4090
contributes its published hardware inputs and validation targets. On the same
1,259 transformer-block observations that the original model represents, mean
error improves from **52.1% (original roofline) to 24.3% (fitted function)**.
The GPU-holdout latency-weighted absolute error is 16.1%; it weights the supplied
grid's longer operators, not a production workload.

The model holdout trains only on Llama-8B observations and predicts Qwen-32B.
QK normalization is absent from that training split and is explicitly excluded
from the score; it is not silently counted as successfully generalized. The
tensor-shape split assigns SHA256(operator, tensor coordinates) mod 5 == 0 to
validation, grouping identical coordinates across all devices, models and TP
degrees so repeated measurements of those coordinates cannot leak into training.
All splits retrain coefficients from scratch and score the function directly.

The separate [nine-scenario serving comparison](../benchmarks/fitted-vs-profiles-2026-10-19/README.md)
checks integration with queues and communications: mean throughput difference
decreases from 13.5% to 4.1%, TTFT from 30.0% to 14.8%, and TPOT from 10.8% to 4.7%.
Those scenarios use known profiled identities and are not an independent unseen-GPU
accuracy test. The integration change also adds separately fitted embedding,
final-normalization and sampling costs that the legacy model collapsed or omitted;
its end-to-end improvement must not be attributed solely to coefficient fitting. Native reference results are simulator outputs using upstream
profiles; no new end-to-end GPU measurements were collected.

## Claims the evidence does and does not support

Supported claim: a physically constrained function fitted to measured operator
profiles reduces error relative to the original roofline baseline in the tested
domain, and its hardware/model/shape transfer has been measured on explicit
withheld splits. Both local and browser editions run the same function.

The evidence does not establish a universal error bound, exact reconstruction of
every profile, or verified accuracy on H100/B200, AMD, new memory hierarchies,
quantized kernels or different serving engines. The two GPU families were profiled
with different software versions, so hardware and software effects cannot be fully
disentangled. A smooth function also cannot reproduce every kernel dispatch change
or noisy measurement. Those residuals remain visible instead of being removed by
lookup overrides or per-scenario throughput multipliers.

New hardware should be evaluated by adding operator measurements with recorded
engine/CUDA/driver/dtype/TP and tensor shapes, checking physical-feature assumptions,
refitting, then validating on withheld hardware and end-to-end serving runs.
Beyond the training domain, sensitivity results are hypotheses for strategy work.
Measured idle/active power, memory traffic, collectives and serving behavior need
their own calibration. Fused MoE, MLA and hybrid profiles cannot be assigned to
incompatible dense operators. Energy and utilization remain behavioral estimates.

## Reproduce and audit

```sh
.venv-sim/bin/python -B integration/build_calibration.py
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B integration/compare_roofline_profiles.py \
  --timing-mode calibrated --output runs/calibrated-profile-comparison-2026-10-19
```

The offline builder verifies the pinned, pristine upstream sources and records
training-grid, runtime, catalog and exporter hashes. NumPy is required only for
offline fitting; deployed simulation needs the standard library and SimPy.
Use `SERVESCOPE_BROWSER_CHANNEL=chrome` for browser tests if Playwright's bundled
Chromium is absent; installed Chrome runs in isolated test profiles.

KAIST CASYS Lab receives credit for the profiling data and extraction contracts,
under the retained [MIT license](../../licenses/LLMServingSim-MIT.txt). The fitted
function, feature/constraint choices, validation pipeline and interface are
ServeScope extensions by [ryeii (Ryan Z An)](https://github.com/ryeii).
