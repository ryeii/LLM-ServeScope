# Project and third-party credits and notices

## LLM-ServeScope — project design and implementation

[ryeii (Ryan Z An)](https://github.com/ryeii) created LLM-ServeScope's behavioral
modeling extension, serving scenario controls, visualizations, user interface,
browser execution layer and native backend integration. The upstream reference
backend and third-party runtime components are credited separately below.

## LLMServingSim 2.0 — CASYS Lab at KAIST

Authors: Jaehong Cho, Hyunmin Choi, Guseul Heo and Jongse Park.

- Repository: https://github.com/casys-kaist/LLMServingSim
- Paper: https://arxiv.org/abs/2602.23036
- DOI: https://doi.org/10.1109/ISPASS69572.2026.00012
- Pinned integration revision: `b09c7aaab4b072d73136641841cc2688616c49eb`
- Copyright (c) 2024, CASYS Lab at KAIST; MIT license retained in
  `licenses/LLMServingSim-MIT.txt`.

The optional local reference engine uses LLMServingSim's serving scheduler,
measured profiles, Chakra DAG generation and ASTRA-Sim integration. Selected
model/hardware configuration references are derived from its published configs.
Selected dense operator timings are retained as offline training/validation
data in `third_party/kaist-dense-profile-grids.json.gz`, with source hashes.
`integration/profile_reference.py` adapts the pure-phase extraction/interpolation
contracts and retains KAIST's MIT copyright and license. The local and browser
runtime ships fitted coefficients/metadata in `backend/gpu-fit.json`; it does not
read the timing tables. The fitted function, physical features/constraints and
validation pipeline are ServeScope extensions by ryeii (Ryan Z An).
See `docs/calibration.md` for training coverage and validation.
LLM-ServeScope's behavioral model and interface are independently implemented;
this is not an official KAIST product and inherits no upstream validation claim.
The upstream source checkouts and native dependencies are fetched separately,
with their own retained licenses; they are not shipped as native binaries in the
GitHub Pages branch.

## Microsoft Vidur — published profiling corpus

- Repository: https://github.com/microsoft/vidur
- Pinned revision: `abae7f63aa857300f5cdc6f5e0d27860cd24721b`.
- MIT license: `licenses/Vidur-MIT.txt` (also included in GitHub Pages).
- Published FP16 operator profiles are retained with exact paths and hashes in
  `third_party/vidur-profiles/manifest.json`. They train a separate FP16 cohort;
  KAIST BF16 and Vidur FP16 measurements are not pooled. Runtime receives only
  fitted coefficients and descriptive coverage metadata, not timing tables.
- Historical CUDA/driver/clock details are incomplete. Profile-reference model
  presets preserve the profiler's padded vocabulary and documented geometry.

## NVIDIA Dynamo and GPUDirect Storage — architectural references

The Dynamo-style cache preset and GDS behavioral path are independently
implemented by ryeii (Ryan Z An), informed by NVIDIA's
[KVBM design](https://docs.nvidia.com/dynamo/v1.2.0/design-docs/component-design/kvbm-design),
[configuration](https://docs.dynamo.nvidia.com/dynamo/v1.5.1/reference/components/kvbm-configuration)
and [GDS design guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html).
This addition distributes no Dynamo/NIXL source code or measured Dynamo timing
dataset. It is not an NVIDIA product, exact runtime reproduction or inherited
performance validation. GDS timing is configured and uncalibrated.

## SimPy 4.1.1

SimPy provides the discrete-event scheduling engine in local and browser editions.

- Project: https://simpy.readthedocs.io/
- Package: https://pypi.org/project/simpy/4.1.1/
- Original copyright and MIT license: `licenses/SimPy-MIT.txt`.
- The unmodified wheel is retained under `third_party/` and checked against
  SHA-256 `7c5ae380240fd2238671160e4830956f8055830a8317edf5c05e495b3823cd88`.

## Pyodide 314.0.7

Pyodide supplies Python/WebAssembly execution for the static browser edition.
Runtime assets are copied unmodified from the exact npm version locked in
`package-lock.json`. License: MPL-2.0, retained in
`licenses/Pyodide-MPL-2.0.txt`. Source: https://github.com/pyodide/pyodide/tree/314.0.7.
The source/model `.py` files are also shipped beside the browser runtime.

Pyodide includes CPython and compiled runtime components with their original
notices. The build preserves the distributed runtime assets and the Python
standard-library archive. The CPython 3.14.2 license is retained separately as
`licenses/CPython-3.14-PSF.txt`.

## Frontend and preset references

React (MIT), Vite (MIT), Lucide (ISC) and other npm dependencies retain their
licenses in their installed packages. React, ReactDOM and Lucide license texts
are also retained under `licenses/` and copied into the Pages deployment. Their exact dependencies are locked in
`package-lock.json`. GPU/model reference links and frozen publisher configuration
revisions are documented in `docs/configuration.md` and
`docs/model-sources.json`. No model weights are distributed.

## Mooncake storage benchmark and workload traces

The pinned benchmark sources and request dumps are by the Mooncake contributors,
under Apache License 2.0. See `third_party/mooncake-storage/manifest.json` and
`third_party/mooncake-traces/manifest.json` for source revisions and SHA-256 hashes.
The licenses and upstream documentation are retained with each archive.
ServeScope’s local measurement harness, fitted functions, policy-replay study,
KV-tier behavioral model and serving/UI integration are by ryeii (Ryan Z An).
No Mooncake end-to-end validation or device-performance claim is inherited.
