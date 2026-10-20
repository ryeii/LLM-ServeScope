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

## Offline Vidur profiling data

`vidur-profiles/` archives exact published Microsoft Vidur CSVs and profiler
contract source files at revision `abae7f63aa857300f5cdc6f5e0d27860cd24721b`.
Every decompressed file has a source URL and SHA-256 in `manifest.json`.
The original MIT license is retained in `../licenses/Vidur-MIT.txt`.

The function-search study uses FP16 profiles on A100, A40 and H100 separately
from KAIST's BF16 corpus. These are offline research inputs; they are not shipped
with GitHub Pages or imported by application code. See
`../docs/fit-study-2026-10-19/README.md` for timing units, duplicate handling,
hardware assumptions, software metadata gaps, holdout policy and results.
