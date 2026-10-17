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
LLM-ServeScope's behavioral model and interface are independently implemented;
this is not an official KAIST product and inherits no upstream validation claim.
The upstream source checkouts and native dependencies are fetched separately,
with their own retained licenses; they are not shipped as native binaries in the
GitHub Pages branch.

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
