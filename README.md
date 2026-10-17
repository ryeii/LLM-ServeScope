# LLM MemAnalysis

A local behavioral modeling workbench for LLM serving architecture and strategy.
It is a separate application from LLM MemScope: MemAnalysis explores system-level
trade-offs; MemScope explores detailed recorded and modeled memory events.

## Start

```bash
./scripts/start.sh
```

Open **http://127.0.0.1:8788/**. This serves the built UI and the simulation API
on loopback only. Set `MEMANALYSIS_PORT` to change the port. It requires Node.js
22+ and Python 3.10+. The behavioral engine needs only SimPy and runs on CPU.
The native backend is already built on the development host; to reproduce it
on a fresh macOS ARM64 checkout, run `./integration/setup_native.sh` with
CMake, Python 3.10 and a C++ compiler installed. See
[the native integration guide](integration/README.md).

For frontend development, run `.venv/bin/python -m backend.server` and
`npm run dev` in separate terminals, then open http://127.0.0.1:4180/.

## Two execution paths

| Path | What runs | What the numbers mean |
| --- | --- | --- |
| Behavioral scenario | Our operation equations + hierarchical network model + SimPy resources | Explicit, uncalibrated strategy estimates from hardware facts and efficiency assumptions |
| Native LLMServingSim 2.0 | Pristine pinned KAIST serving scheduler + measured profile lookup + Chakra DAG + ASTRA-Sim analytical backend | Native simulated serving clocks for the selected profiled accelerator/model combination |

The native integration is real and CPU-only. It does not execute model kernels
or generate natural-language output. Request lengths and deterministic token-ID
markers are workload inputs. The behavioral path is an independently implemented
extension; its roofline, SRAM spill and four-tier strategy model must not be
reported as native KAIST results. Custom peak TFLOPS do not create a valid native
kernel profile. New hardware requires calibration.

## Interface and outputs

- **Overview:** generation/prompt throughput, TTFT, TPOT, P95 latency, per-GPU
  throughput, memory capacity/headroom, estimated GPU energy, tokens/J, GPU resource
  occupancy and scenario GPU rental cost.
- **Roofline:** per-phase tensor ceiling, HBM bandwidth slope, operation arithmetic
  intensity/attainment, vector/launch/tiling effects, and collective fabric tier.
- **Layer / chunk:** per-operation compute, memory, network and service estimates;
  per-microbatch pipeline reservations. Export a layer CSV.
- **Box graph:** stage-resource timeline plus request latency box-and-whisker plots.
- **Model graph:** interactive representative transformer DAG with KV branch and
  residual path; actual full native Chakra dependencies in the native tab.
- **Configuration:** complete resolved inputs, source pin, configuration SHA-256,
  units, provenance and limitations. Export/import scenario JSON.
- **LLMServingSim 2.0:** run the native engine from the UI, adjust memory and link
  inputs, inspect native metrics/request boxes/Chakra DAGs, download native CSV/logs
  and export the complete result and artifact hashes.

Pin a behavioral baseline, change hardware or topology, and run again to compare
metric deltas. Inputs marked as changed are not applied until **Run scenario**.
Existing result graphs retain their original resolved configuration after an
invalid run. SVG chart downloads and JSON exports contain the calculated result,
not a screenshot-derived approximation.

The behavioral architecture catalog covers Llama 3.1 8B and 70B (dense/GQA) and
Qwen3 30B-A3B (MoE). vLLM-style/TensorRT-LLM-style/eager choices are disclosed
assumed efficiency policies, not benchmarked implementations of those engines.
Native demo profiles support Llama 3.1 8B BF16 on RTX PRO 6000 (TP1/TP2) and
RTX4090 (TP1); unsupported combinations are rejected.

## Evidence

The native integration was built and executed on this Mac without a GPU.
Four requests, each with 128 input and 8 output tokens, produced:

| Simulated setup | Native total duration | Output tokens/s |
| --- | ---: | ---: |
| RTX PRO 6000, TP1 | 102.459747 ms | 312.317773 |
| RTX PRO 6000, TP2, 16 GiB/s link | 76.267560 ms | 419.575505 |
| RTX PRO 6000, TP2, 1 GiB/s link | 330.905562 ms | 96.704328 |

The last two change only link bandwidth. Raw native CSVs, commands, generated
configs, text traces, Chakra graphs, process resource observations and SHA-256
manifests are retained locally under `runs/native-demo-*`. Those runtime products
and compiled dependencies are ignored by Git; the small comparison manifest is
in [examples/native-comparison.json](examples/native-comparison.json). Native total
clocks can include a final idle/control interval beyond the last request. The
adapter preserves both native duration and request-span duration separately.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
npm run build
PLAYWRIGHT_BROWSERS_PATH=.cache/playwright npx playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=.cache/playwright npm run test:ui
```

Browser checks use full Chromium (`channel: chromium`) because the installed
headless-shell bundle lacks its ICU data. They exercise all six output families,
baseline comparison, downloads, capacity errors, a native simulator run with
actual DAG/CSV results, and responsive mobile controls. Screenshots are in
`docs/screenshots/`.

See [methodology and calibration](docs/methodology.md) and
[specification coverage](docs/specification-coverage.md).

## Sources and licensing

KAIST [LLMServingSim](https://github.com/casys-kaist/LLMServingSim),
[paper](https://arxiv.org/abs/2602.23036), and
[profiling documentation](https://llmservingsim.ai/docs/profiler).
The upstream MIT license is retained in `licenses/LLMServingSim-MIT.txt`.
All simulator checkouts are pinned and ignored under `.vendor/`; setup clones
the exact revisions. No credentials, private specification PDF, model weights,
or MemScope files are copied into this repository.
