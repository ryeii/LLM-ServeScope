# LLM-ServeScope

<p align="center">
  <img src="https://ryeii.github.io/LLM-ServeScope/brand/servescope-serving-telescope-white.png" alt="LLM-ServeScope logo" width="96" height="96" />
</p>

**Behavioral modeling of LLM serving, backed by a SimPy discrete-event simulation.**

LLM-ServeScope combines operation-level compute, memory and network cost models
with serving-resource scheduling to estimate **throughput, latency, memory,
energy and utilization**. It reports system totals, per-GPU performance and
energy efficiency such as **output tokens per GPU-second and output tokens per
Joule**.

Use it to compare GPU specifications, model architectures, parallelism, batching,
and colocated versus separate prefill/decode servers. It is intended for
architecture exploration, capacity planning, marketing and strategy scenarios
where the critical hardware characteristics are available but a full deployment
is not. Hardware inputs and assumptions remain visible and exportable.

[Browser edition](https://ryeii.github.io/LLM-ServeScope/) ·
[Configuration guide](docs/configuration.md) ·
[Methodology](docs/methodology.md) ·
[How results are calculated](docs/how-results-are-calculated.md) ·
[Third-party credits](THIRD_PARTY_NOTICES.md)

![LLM-ServeScope overview](https://ryeii.github.io/LLM-ServeScope/docs/screenshots/github-pages.png)

## What it models

A request arrives, forms a microbatch, passes through chunked prefill, optionally
transfers its KV cache, and completes decode iterations. SimPy models queueing,
replica admission, pipeline stages, shared resources and KV-transfer contention.
Fitted GPU operator times include the kernels’ compute/HBM interaction. Separate
models cover KV transfers, cache policies, memory admission and network links;
SimPy determines when their work can run and how long it queues.

- Nine GPU presets plus custom hardware: A100, A40, H100, H200, B200, L40S,
  RTX 4090, RTX PRO 6000 and AMD MI300X.
- Thirteen model presets across Llama, CodeLlama, Qwen3 and Mixtral, plus custom
  uniform dense/MoE decoder geometry with MHA, GQA or MQA attention. Preset
  availability does not imply measured timing coverage for every model/GPU pair.
- Independent prefill/decode hardware, TP/PP/EP, replicas, engine-efficiency
  policies, batch caps, active-batch limits and memory reservations.
- Independent retained-prefix policies and active decode KV placement: HBM, DRAM,
  SSD, HBM + DRAM spill or HBM + SSD spill. Explicit staging, admission and
  shared I/O queues; [model and validation scope](docs/active-kv-model.md).
- Separate weight/KV precision, chunk sizes, hierarchical fabrics, parallel KV
  links, power assumptions and GPU-hour cost inputs.

## Outputs

| View          | Outputs                                                                                                                                            |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Overview      | Generation/prompt throughput, TTFT, TPOT, P95 latency, GPU count, resource occupancy, energy, tokens/J, tokens/GPU-second and assumed serving cost |
| Memory        | Per-GPU weight, KV and activation payload, usable capacity and headroom                                                                            |
| Roofline      | Compute/memory ceilings and per-operation arithmetic intensity and attainment                                                                      |
| Layer / chunk | Per-operation compute, memory, network and service costs; CSV export                                                                               |
| Box graph     | Simulated pipeline/resource timeline and request-latency distributions                                                                             |
| KV storage    | Independent active-KV and retained-prefix residency, backing capacity, reads/writes, cache reuse and I/O queues |
| Model graph   | Interactive representative computation DAG with KV and residual branches                                                                           |
| Configuration | Complete resolved inputs, source revisions, assumptions and SHA-256 fingerprint                                                                    |

Pin a baseline, edit a scenario and run again to compare metrics. Export/import
scenario JSON, export complete results and download SVG charts. Hover or keyboard-focus
a result number to inspect its equation, labeled values, units and fitted coefficients.
Hover or select a marked term to follow a cascade of explanations: throughput →
simulation horizon → last request completion → service and queues → GPU stage →
operator function → arithmetic/traffic, fitted coefficients and hardware inputs.
The cards use the actual executed shapes and resource reservations, and identify
scenario inputs, fitted parameters and modeling assumptions at the source.
Breadcrumbs and Back retain context; Esc steps back, and the close button or a tap
outside closes the cascade. On narrow screens, one card is shown at a time.
Explanations use the completed run, so pending edits do not change its derivation. Changed inputs
are applied only when **Run scenario** is selected.

## How a scenario becomes a result

**Inputs → service-time estimates → resource scheduling → request timestamps → metrics.**
GPU functions estimate operation service, link/storage functions estimate transfers,
and SimPy schedules those operations on shared resources. A busy GPU, link or
backing store adds waiting. Completion times emerge from that schedule.

![How model inputs, service estimates and scheduling produce generation throughput](https://ryeii.github.io/LLM-ServeScope/docs/results-explained/calculation-process.png?v=7fd8f61d14bd)

The default scenario has 16 requests, four requests per batch, 2,048 input tokens
and 64 output tokens. Request #12 follows this path:

| Event | Simulation timestamp | Elapsed since previous event |
| --- | ---: | --- |
| Request arrives | 1,500.000 ms | Arrival is a clock reading, not time spent waiting |
| Batch #12–15 is ready | 1,875.000 ms | 375.000 ms waiting for the other requests |
| Prefill finishes | 2,746.724 ms | 871.724 ms of prompt processing and queues |
| KV handoff finishes; first token is released | 2,854.109 ms | 107.384 ms of KV transfer |
| Remaining 63 tokens finish | 5,486.647 ms | 2,632.538 ms of decode service and queues |

A **timestamp** locates an event on the clock; a **span** is an elapsed duration.
For example, this request’s decode span contains **1,102.833 ms of GPU-stage
service plus 1,529.705 ms of waiting**. Its own latency is completion minus arrival:
**3,986.647 ms**. The model deliberately releases the first token after KV handoff.

Batches overlap, so the whole run ends at the **latest completion**, rather than
at the sum of completion timestamps. Requests #12–15 tie for last in this run.

```text
Simulation horizon = max(request completion timestamps) = 5.486646698 seconds
Completed output tokens = 16 × 64 = 1,024
Generation throughput = 1,024 / 5.486646698 = 186.634944 tokens/s → displayed as 187
```

The horizon includes arrivals, startup, queues and the final drain; this is a
finite-run serving estimate. Hover labels such as `finish_last` identify a
completion timestamp, while `base_result` and `display_scale` only explain unit
conversion. They are not additional processing steps.

The [illustrated walkthrough](docs/how-results-are-calculated.md) follows these
numbers down to individual GPU services, FIFO waits and the KV-transfer formula.
It includes an actual batch timeline, source-linked equations and a
[reproducible numeric breakdown](docs/results-explained/numbers.json).

## Which timing model is used where?

| Component | Data/model | When it contributes time |
| --- | --- | --- |
| GPU kernels | Functions fitted separately to KAIST BF16 and Microsoft Vidur FP16 profiles | Prefill and decode; internal HBM effects are already included |
| SSD/file-path I/O | Configured latency/bandwidth, or the selected fit to local measurements using Mooncake benchmark code | Actual SSD reads/writes for retained prefixes or active decode KV |
| Host–GPU transfers and explicit HBM copies | Configured effective bandwidth and latency | KV restore, publication, spill or active backing transfers |
| Collectives and KV handoff | Analytical network functions and configured topology | Communication inside/between GPU pools |
| Admission and resource waits | SimPy queues, dependencies and capacity limits | When work cannot yet acquire the resources it needs |

The default keeps active KV in HBM and disables prefix retention, so **no separate
DRAM/SSD I/O service time is added**. Its network costs and scheduling waits still
apply. The fitted SSD option is used only when **“Fitted local Mooncake file path”**
is selected and the scenario actually performs SSD I/O. Changing link settings
alone does not create transfers. A kernel’s HBM access time is not added again on
top of its fitted operator service.

## Fitted GPU service-time model

GPU timing uses **shape-informed functions selected with nested cross-validation**
on KAIST BF16 and Microsoft Vidur FP16 profiles. Hardware rates, operation and
tensor geometry determine execution time; no GPU/model-ID coefficients or runtime
timing-table lookups are used. SimPy schedules the predictions for serving behavior.

- Training: **599,451 compatible dense operator observations**, five GPU families,
  separate historical precision/software cohorts. Final deployment fits use all
  observations in each cohort after evaluation.
- Grouped nested validation: **14.86% mean absolute operator-time error**, **9.54%
  median**, **49.52% P95**. Five outer folds evaluate family/scale selection made
  inside three training-only inner folds.
- [Full accuracy report](docs/accuracy-2026-10-19/README.md): separate KAIST and Vidur overview figures,
  **24 corpus-specific subcomponent figures**, per-GPU breakdown, and separate unseen-GPU validation.
- GPU operator accuracy does not establish throughput, energy, memory or network
  accuracy. Those outputs require their own validation. Unsupported specialized
  profile bundles and uncalibrated extrapolations are listed explicitly.
- KAIST embedding uses **two fitted functions switched at 16 tokens**. FP16 has its
  own selected functions; missing operators or unknown cache inputs report fallbacks.
- Original roofline mode remains available for comparison.

| Profile corpus | Precision | Compatible observations | Operator-time MAPE | P95 absolute relative error |
| --- | --- | ---: | ---: | ---: |
| KAIST / LLMServingSim 2.0 | BF16 | 9,411 | 10.64% | 35.76% |
| Microsoft Vidur | FP16 | 590,040 | 14.93% | 49.68% |

These cohorts are fitted separately. The combined headline is observation-weighted,
not an equal-weight average across GPUs or workloads. Credit for the published
profiles belongs to [CASYS Lab at KAIST](https://github.com/casys-kaist/LLMServingSim)
and [Microsoft Vidur](https://github.com/microsoft/vidur); the fitting and validation
pipeline is a ServeScope contribution by [ryeii (Ryan Z An)](https://github.com/ryeii).

![KAIST cross-validated GPU operator timing accuracy](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-main.png?v=7d83f37f3fbd)

![Vidur cross-validated GPU operator timing accuracy](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-main.png?v=465f07546ef7)

See [current function and assumptions](docs/calibration.md), the earlier
[function-search experiments](docs/fit-study-2026-10-19/README.md), and the
[two-regime embedding experiment](docs/fit-study-2026-10-19/embedding-piecewise/README.md).
The [current serving integration recheck](docs/accuracy-2026-10-19/README.md#serving-integration-recheck-separate-evidence)
reports 4.83% mean throughput difference from nine native simulator references;
this is separate evidence from operator CV and from physical GPU serving accuracy.

## SSD service-time model and Mooncake evidence

ServeScope fits a size-to-latency function for the **local storage file path** and
derives bandwidth consistently as bytes divided by that time. The data comes from
**768 measurements collected by ServeScope using Mooncake’s archived, unmodified
`DiskHashTable` benchmark class**: 384 reads and 384 writes, eight I/O sizes from
16 KiB to 2 MiB, serial queue depth one, warm buffered reads and fsync writes on a
development Mac. This is not a fit to raw-device service or a customer SSD.

### Main held-out accuracy results

Each of eight outer folds withholds **all measurements of one I/O size**. Function
selection uses only the remaining training sizes in inner folds. The deployed
function is then refitted on all measurements; the figure and errors below use
outer held-out predictions, not training residuals.

| Operation | Selected function | Latency MAPE | Latency P95 error | Bandwidth MAPE | Bandwidth P95 error |
| --- | --- | ---: | ---: | ---: | ---: |
| Read | Continuous two-slope | **16.37%** | 53.09% | 29.42% | 113.16% |
| Write | Affine: fixed overhead + bytes/bandwidth | **9.97%** | 27.90% | 12.04% | 38.70% |

MAPE is mean absolute percentage error; P95 is the 95th percentile of absolute
percentage error. Bandwidth errors are a reciprocal view of the same timing
observations, not independent validation. Warm-cache reads can exceed physical
SSD bandwidth, and their read-error tail is substantial. These results do not
establish cold-SSD, higher-queue-depth, cross-machine or end-to-end serving accuracy.

![Local Mooncake file-path latency and derived bandwidth: measured values versus nested held-out predictions](https://ryeii.github.io/LLM-ServeScope/storage-study/io-fit.png?v=17c686000e4f)

[Data coverage, equations, candidate functions and validation](docs/storage-2026-10-20/README.md)
· [Raw local measurements](docs/storage-2026-10-20/io-observations.json)
· [Deployed coefficients](backend/storage-fit.json)

**Source attribution matters:** [Mooncake contributors](https://github.com/kvcache-ai/Mooncake)
supply the benchmark code and published request traces. ServeScope’s measurement
harness, local observations, fitting, validation and tier integration are by
[ryeii (Ryan Z An)](https://github.com/ryeii). Mooncake’s published aggregate
SSD-offload serving results are documented separately; they are not the per-size
training samples used for this fit.

### Independent active KV and retained prefixes

- **Active decode KV** belongs to an ongoing request. Choose HBM, host DRAM, SSD,
  HBM + DRAM spill or HBM + SSD spill. Offload adds initialization, per-step reads,
  append writes, staging capacity and backing admission. Prefill working KV remains
  in HBM. The serialized offload composition is **not end-to-end calibrated** by
  the local I/O error figures above; see [model and limitations](docs/active-kv-model.md).
- **Retained prefixes** can survive completed requests and save later prefill work.
  Choose no retention, HBM-only, DRAM-only, SSD-only, directional spill or inclusive
  copies. Ready hits reduce query work while preserving full attention context;
  publication and required spill moves block before handoff. See
  [cache policy details](docs/configuration.md#cache-layouts-and-eviction-spill).
- **KV I/O paths** configure the shared SSD and host–GPU service functions. Both
  lifetimes contend for those physical queues, with separate counters. Host DRAM
  currently uses the configured effective host–GPU path, not its own fitted
  DRAM service function. Explicit HBM copies and kernel HBM behavior do not share
  a detailed bandwidth-contention model.

Choose **Load cache demo**, then **Run scenario** for a synthetic example. Import
unchanged Mooncake JSONL or load a bundled published trace window; source identity,
relative arrivals and original token lengths remain in exports. The resulting
serving and storage intervals are simulated, not captured Mooncake hardware events.
Fixed uniform batches support one replica per pool; variable-length replay requires
batch size one. Layout changes preserve workload and batch settings.

Cache-hit forecasts have a different target: **1,260 trace-replayed LRU curve
observations**, rather than measured hardware hits. Nested held-region MAE is
1.27 percentage points for conversation, 3.79 for tool/agent and 0.53 for synthetic
traces. Runtime hits come from executed cache state, never from those fitted
forecasts. [Evidence scopes and raw artifacts](docs/storage-2026-10-20/README.md)
keep published results, local I/O measurements and policy replay distinct.

## Limitations: behavioral, not mechanistic simulation

**LLM-ServeScope is a behavioral simulation, not a mechanistic simulation.**
It abstracts hardware and serving behavior into operation costs and discrete-event
resource reservations. It does not reproduce individual GPU instructions,
cycle-level execution, page-level memory accesses, memory-controller transactions,
hardware cache replacement or physical network packets. Its box graph contains modeled
scheduling intervals, not a captured hardware execution trace.

- Peak specifications, efficiency policies, ideal KV SRAM reuse, uniform MoE
  routing, activation traffic and energy coefficients are explicit assumptions.
  Useful sensitivity comparisons do not establish absolute prediction accuracy.
- Decision-grade estimates require calibration against operation latency grids,
  memory traffic/capacity, collectives, scheduling and idle/active power data for
  the actual hardware, model and engine configuration.
- Fixed prefill microbatches may split on decode; the behavioral engine does not
  merge groups or implement native vLLM continuous batching.
- The custom architecture model covers uniform causal decoder blocks with
  RMSNorm, RoPE and SwiGLU, optional QK normalization and routed experts. MLA,
  hybrid/sliding attention, shared experts, vision encoders and SSM layers need
  additional operation models.
- Capacity checks include modeled payload/workspace and explicit reservations.
  Allocator fragmentation, unspecified runtime overhead, quantization conversion
  cost and numerical quality are outside the model.
- Energy covers GPUs, not host CPUs, network equipment or cooling. Resource
  occupancy is not tensor-core utilization. Hourly GPU costs are user assumptions.
- Simulation does not execute model kernels, generate natural-language text, or
  validate a new chip. No accuracy claim from another simulator is inherited.

The optional native reference path uses profile-driven LLMServingSim 2.0 and
ASTRA-Sim. It does not turn the behavioral model into a mechanistic GPU or memory
simulator. Use detailed traces or an appropriate mechanistic simulator when
instruction-, cycle-, page- or transaction-level behavior is the question.

## Credits

### LLM-ServeScope — ryeii (Ryan Z An)

[ryeii (Ryan Z An)](https://github.com/ryeii) created LLM-ServeScope, including
its behavioral serving model, GPU and storage fitting/validation pipelines, local
I/O measurement harness, KV-tier integration, scenario controls, visualizations,
request-timestamp explanations, user interface, browser execution layer and
native backend integration.

### Reference backend — LLMServingSim 2.0

LLMServingSim 2.0 is developed by **CASYS Lab at KAIST**. Credit goes to
**Jaehong Cho, Hyunmin Choi, Guseul Heo and Jongse Park** for
_LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM
Serving Infrastructure_ (ISPASS 2026).

- [Official repository](https://github.com/casys-kaist/LLMServingSim)
- [Paper](https://arxiv.org/abs/2602.23036)
- [Documentation](https://llmservingsim.ai/)
- [Retained MIT license](licenses/LLMServingSim-MIT.txt)

The local reference integration runs its pinned serving scheduler, measured
operation profiles, Chakra DAG generator and compiled ASTRA-Sim backend. Some
model geometries and the RTX PRO 6000 hardware reference also come from the
pinned upstream configuration. The behavioral equations, strategy controls,
visualizations and browser execution layer are independently implemented in
LLM-ServeScope. This project is not an official KAIST release or endorsement.

```bibtex
@inproceedings{cho2026llmservingsim,
  author = {Jaehong Cho and Hyunmin Choi and Guseul Heo and Jongse Park},
  title = {{LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure}},
  booktitle = {2026 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  year = {2026},
  doi = {10.1109/ISPASS69572.2026.00012}
}
```

### FP16 GPU profiles — Microsoft Vidur

[Microsoft Vidur](https://github.com/microsoft/vidur) supplies the published FP16
operator profiles for A100, A40 and H100, covering Llama 2 and CodeLlama geometries.
ServeScope imports the archived profiles with source hashes and fits an independent
FP16 cohort. The [source manifest](third_party/vidur-profiles/manifest.json) and
[retained MIT license](licenses/Vidur-MIT.txt) accompany the data. This is profile
data reuse; the default behavioral engine does not run Vidur’s simulator or inherit
its end-to-end accuracy claims.

### Storage benchmark and request traces — Mooncake contributors

[Mooncake](https://github.com/kvcache-ai/Mooncake) supplies the archived storage
benchmark code and request dumps, with Apache 2.0 licenses retained in the
[benchmark archive](third_party/mooncake-storage) and
[trace archive](third_party/mooncake-traces). Local measurement, fitted storage
functions, cache-policy replay and ServeScope integration are by
[ryeii (Ryan Z An)](https://github.com/ryeii). Published Mooncake serving results,
ServeScope’s local file-I/O measurements and trace-derived targets are credited
and evaluated separately; no Mooncake hardware-performance claim is inherited.

### Scheduling and browser runtime — SimPy and Pyodide

[SimPy](https://simpy.readthedocs.io/) supplies discrete-event scheduling;
[Pyodide](https://pyodide.org/) runs the same Python model in the browser.
Their retained licenses, pinned versions and other dependency notices are in
[third-party credits](THIRD_PARTY_NOTICES.md).

## GitHub Pages edition

The `production` branch contains prebuilt static files at its root. The same
`backend/model.py` and SimPy 4.1.1 run inside a **Pyodide Web Worker**. Scenario
inputs stay in the browser; no Python web server, API service, Hugging Face token,
model weights or GPU is required. The Python/WebAssembly runtime and SimPy are
bundled with the site and served from the same origin. The first load downloads
the runtime before the first calculation.

The browser supports the behavioral controls, result views and exports.
The native LLMServingSim/ASTRA-Sim executable is available only in the local
edition; the native tab explains how to install it locally.

To deploy this repository as requested:

1. Open **Settings → Pages**.
2. Select **Deploy from a branch**.
3. Select **`production`** and **`/(root)`**, then save.

To rebuild and update that branch after committing source changes:

```bash
npm ci
npm run build:pages
# Optional local static preview at /LLM-ServeScope/:
npx vite preview --outDir dist-pages --base /LLM-ServeScope/ --host 127.0.0.1 --port 4182
# Publish the build to production without switching the source checkout:
npm run publish:pages
```

The default base path is `/LLM-ServeScope/`. Override `SERVESCOPE_BASE_PATH` if the
repository is hosted at another path. The publish script requires a clean source
checkout and an authenticated `origin`, preserves production history, and refuses
to replace a deployment built from source that has not been pushed to `origin`.

## Local installation

Requires Node.js 22+ and Python 3.10+.

```bash
git clone https://github.com/ryeii/LLM-ServeScope.git
cd LLM-ServeScope
./scripts/start.sh
```

Open **http://127.0.0.1:8788/**. This serves the UI and Python API on loopback only.
Set `SERVESCOPE_PORT` to change the port. Behavioral simulation runs on CPU and
needs only SimPy. For frontend development, run the API and `npm run dev` in
separate terminals; Vite uses http://127.0.0.1:4180/.

For the optional native reference backend on macOS ARM64, install CMake,
Python 3.10 and a C++ compiler, then run:

```bash
./integration/setup_native.sh
```

Native profiles currently support Llama 3.1 8B BF16 on RTX PRO 6000 (TP1/TP2) and
RTX4090 (TP1). New compute hardware requires appropriate measured profiles;
custom peak TFLOPS alone do not supply them. See the
[native integration guide](integration/README.md).

## Reproducibility and validation

```bash
.venv/bin/python -m unittest discover -s tests -v
npm run build
npm run build:pages
PLAYWRIGHT_BROWSERS_PATH=.cache/playwright npx playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=.cache/playwright npm run test:ui
PLAYWRIGHT_BROWSERS_PATH=.cache/playwright npm run test:pages
```

Checks cover model geometry, source pins, token accounting, queue causality,
capacity rejection, phase policies, browser/local numeric parity, nested Pages
paths, static-only simulation, imports/exports, native graphs and responsive UI.
Native evidence checks are skipped when optional retained artifacts are absent.
The native browser test requires a built local native backend.

Examples: [heterogeneous servers](examples/heterogeneous-scenario.json),
[custom decoder](examples/custom-decoder-scenario.json). Native run receipts are
summarized in [the comparison manifest](examples/native-comparison.json); full
runtime outputs and compiled dependencies are ignored by Git. Historical receipt
hashes remain unchanged across the product rename.

LLM-ServeScope explores system-level serving behavior. The separate LLM-MemScope
project visualizes detailed recorded and modeled memory events.
