# Browser simulation dependency

`simpy-4.1.1-py3-none-any.whl` is the unmodified SimPy 4.1.1 wheel from PyPI.
It is bundled so the static deployment does not need a package-index request.
The build verifies SHA-256:

`7c5ae380240fd2238671160e4830956f8055830a8317edf5c05e495b3823cd88`

Original MIT license: `../licenses/SimPy-MIT.txt`. Pyodide is installed through
the pinned npm lockfile, then copied into the Pages build without modification.

## Offline KAIST profiling data

`kaist-dense-profile-grids.json.gz` retains selected pure-phase dense operator
observations from pinned KAIST LLMServingSim profiles for reproducible fitting
and validation. It is not shipped in the Pages runtime. Source digests, coverage
and retained MIT credit are documented in `../docs/calibration.md` and
`../licenses/LLMServingSim-MIT.txt`. Runtime coefficients are in `backend/gpu-fit.json`.
