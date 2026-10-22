# Prefix-reuse functions from published Mooncake request traces

This study uses **only the unchanged published request traces**, preserved with
SHA-256 checksums in [the manifest](../../third_party/mooncake-traces/manifest.json).
It contains no measured SSD, DRAM, GDS or host–GPU timings.

The reference policy is cold-start, fully associative LRU with instantaneous
publication. A hit requires a contiguous prefix. Chained prefix identities match
the runtime's key construction and use the source's 512-token block size.
Conversation and tool/agent are production-sampled request workloads; upstream
synthetic requests retain their synthetic classification.

The three traces are each divided into six non-overlapping temporal regions.
Within a region, nested windows of 8, 32, 128, 512 and full-region request counts
are evaluated at 14 capacities. This produces **1,260 replay observations**.
All overlapping windows and capacity points share their region holdout.

Candidate functions are capped-linear (two parameters), saturating exponential
(two), and Weibull (three), with bounded nonnegative coefficients. Selection
prefers the simpler family within 0.5 percentage points of the best inner-fold
MAE. Six outer folds exclude one complete temporal region; function selection
and coefficient fitting use only the remaining regions. Final runtime fits use
all replay observations after evaluation.

| Published trace | Nested held-out MAE | P95 absolute error |
| --- | ---: | ---: |
| Conversation | 1.27 percentage points | 3.78 percentage points |
| Tool/agent | 3.79 percentage points | 13.75 percentage points |
| Synthetic | 0.53 percentage points | 3.29 percentage points |

![Published-trace LRU replay and held-out prefix-reuse predictions](https://ryeii.github.io/LLM-ServeScope/trace-reuse/hit-fit.png)

The fitted response is `H_max × a × min(1,x/s)` or
`H_max × a × (1−exp(−(x/s)^p))`, where `x` is capacity/unique prefix blocks;
`p=1` gives the exponential. `H_max` excludes compulsory first references.

These are **errors against a specified simulated policy**, not errors against
observed production cache telemetry. They do not establish device timing or
end-to-end accuracy. Finite publication times, tiering, admission, fixed batches
and other policies can yield different runtime hits. The runtime executes its
cache state; it does not sample hits from this forecast.

Artifacts: [replay observations](hit-observations.json),
[held-out predictions](held-out-predictions.json), [selection and held-out models](validation.json),
[runtime functions](../../backend/prefix-hit-fit.json), and
[reproducible trace-only harness](../../integration/storage/trace_reuse.py).

```bash
.venv-sim/bin/python -B -m integration.storage.trace_reuse all
```

Source request traces: [Mooncake contributors](https://github.com/kvcache-ai/Mooncake).
Replay, fitting, validation and integration: [ryeii (Ryan Z An)](https://github.com/ryeii).
