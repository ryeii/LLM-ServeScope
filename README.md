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
[Accuracy and validation](docs/accuracy.md) ·
[Configuration guide](docs/configuration.md) ·
[Methodology](docs/methodology.md) ·
[How results are calculated](docs/how-results-are-calculated.md) ·
[Third-party credits](THIRD_PARTY_NOTICES.md)

![LLM-ServeScope overview](https://ryeii.github.io/LLM-ServeScope/docs/screenshots/github-pages.png)

## What it models

A request arrives, forms a microbatch, passes through chunked prefill, optionally
transfers its KV cache, and completes decode iterations. SimPy models queueing,
replica admission, pipeline stages, shared resources and KV-transfer contention.
Fitted GPU operator times include the kernels’ compute/HBM interaction. Separate analytical models cover inter-GPU transfers and network links;
SimPy determines when their work can run and how long it queues.

- Nine GPU presets plus custom hardware: A100, A40, H100, H200, B200, L40S,
  RTX 4090, RTX PRO 6000 and AMD MI300X.
- Thirteen model presets across Llama, CodeLlama, Qwen3 and Mixtral, plus custom
  uniform dense/MoE decoder geometry with MHA, GQA or MQA attention. Preset
  availability does not imply measured timing coverage for every model/GPU pair.
- Independent prefill/decode hardware, TP/PP/EP, replicas, engine-efficiency
  policies, batch caps, active-batch limits and memory reservations.
- HBM-resident KV only. Optional Mooncake-derived prefix-reuse potential reduces
  prefill query work while preserving full attention context; no-reuse is the default.
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
| Model graph   | Interactive representative computation DAG with KV and residual branches                                                                           |
| Configuration | Complete resolved inputs, source revisions, assumptions and SHA-256 fingerprint                                                                    |

Pin a baseline, then edit a scenario: the original summary is preserved and a
second block lists changed inputs as original → new values. Run the edited setup
to calculate its results. **Hold P** to temporarily show the pinned run across all
result views; release it to return to the current run. The highlighted summary
identifies the displayed results. The shortcut is inactive while typing in inputs.
The pinned block also has a press-and-hold preview control for touch/pointer use.

Export/import
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

Hover or keyboard-focus a scenario input to see its current value, meaning,
effect on the model, and evidence or assumption. This covers numeric fields,
selectors, toggles, custom model geometry and native-reference inputs. Pool-specific
cards identify Prefill or Decode; input explanations follow live edits, while
result explanations continue to use the completed run. Press Esc to dismiss help.

## How a scenario becomes a result

**Inputs → service-time estimates → resource scheduling → request timestamps → metrics.**
GPU functions estimate operation service, analytical network functions estimate transfers,
and SimPy schedules those operations on shared resources. A busy GPU or link adds waiting. Completion times emerge from that schedule.

![How model inputs, service estimates and scheduling produce generation throughput](https://ryeii.github.io/LLM-ServeScope/docs/results-explained/calculation-process.png?v=eebe690e9974)

The default no-reuse scenario (`examples/default-scenario.json`)
has 16 requests, four requests per batch, 2,048 input tokens
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

## Evidence and scope

| Component | Evidence | Boundary |
| --- | --- | --- |
| GPU operator timing, including internal HBM behavior | Functions fitted to KAIST BF16 and Vidur FP16 profiles | Cross-validation measures kernel-time prediction error, not whole-serving accuracy |
| Prefix-reuse potential | Exact earlier-arrival prefix counts from Mooncake's published request traces | Optional workload assumption; ready HBM residency and transfer to another workload are unvalidated |
| Scheduling and network | SimPy dependencies and configured analytical network functions | Behavioral assumptions; no general measured end-to-end error guarantee |
| Memory, utilization, energy and cost | Geometry, resource accounting, power/cost inputs | Estimates; GPU timing residuals do not validate these outputs |

All active KV stays in HBM. The unsupported SSD, offload, tiering and cache-I/O
subsystem has been removed. Old exports containing its settings fail validation
rather than silently receiving different semantics. The GPU fitted payload and
its validation artifacts are unchanged.

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

## Mooncake-derived prefix reuse

Under **Model & workload → Prefix reuse assumption**, choose **No reuse** (default)
or a published Mooncake source. Its full-trace input-token-weighted reuse potential
is **37.33% conversation**, **57.02% tool/agent**, or **65.04% synthetic**.
We count only matching complete leading blocks from strictly earlier arrivals,
with cold initial history and unlimited retention. These are descriptive workload
statistics, not measured hardware hits or a validated cache policy.

ServeScope applies the selected fraction uniformly to the configured prompt length,
assumes reused KV is already ready in HBM, and evaluates the remaining prefill queries
against their full attention context. Decode context and KV capacity remain complete.
This is a warm-input scenario assumption, not replay of Mooncake's original requests.
It is not a universal reuse percentage or guaranteed speedup.

[Source counts, method, feasibility limits and reproduction](docs/prefix-reuse/README.md) ·
[Source manifest](third_party/mooncake-traces/manifest.json) ·
[Audited summary](backend/prefix-reuse.json).

![Mooncake earlier-arrival reuse potential](https://ryeii.github.io/LLM-ServeScope/prefix-reuse/reuse-potential.png)

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
its behavioral serving model, GPU fitting/validation pipeline, published-trace
prefix-reuse analysis, scenario controls, visualizations,
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

### Published request traces — Mooncake contributors

[Mooncake](https://github.com/kvcache-ai/Mooncake) supplies the unchanged published
request traces, with the Apache 2.0 license and source checksums retained in the
[trace archive](third_party/mooncake-traces). Published-trace prefix-reuse analysis and ServeScope integration are by [ryeii (Ryan Z An)](https://github.com/ryeii).
The request traces provide workload evidence, not hardware timing measurements.

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
