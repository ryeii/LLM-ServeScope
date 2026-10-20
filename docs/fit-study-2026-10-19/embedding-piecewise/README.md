# F9: two embedding functions with a fixed dispatch switch

> Historical experiment. The subsequent [production promotion and all-data nested validation](../../accuracy-2026-10-19/README.md) supersede statements below about the deployed baseline. Original experiment scores are preserved.

Follow-up to the [original nine-function study](../README.md) and
[embedding-dispatch diagnosis](../embedding-dispatch.md).

**The two regimes do fit substantially better separately.** For the originally
plotted configuration, small-index and gather training errors are
**0.89%** and
**4.51%** respectively.
Five-fold shape-validation errors are **0.97%**
and **4.57%**.

![Two independently fitted embedding functions](two-function-fit.png)

[Full-resolution SVG](two-function-fit.svg).

## Functions and the switch

```
M_ns = n * (2 * dtype_bytes * hidden + 8) / (DRAM_bandwidth_TBps * 1000)
T(n, hardware) = max(1 ns, L_small + b_small * M_ns)  if n ≤ 16
                max(1 ns, L_gather + b_gather * M_ns) if n > 16
```

Both intercepts and both slopes are nonnegative. Four coefficients are fitted
by nonnegative relative least squares. There is no compute slope for embedding,
which has zero FLOPs in this abstraction. The boundary is fixed at 16 from the
observed discontinuity and published software dispatch, not optimized against
validation targets. The functions have independent intercepts: enforcing continuity
would erase the observed latency drop. Integer counts 17–19 are unmeasured in the
KAIST grid and follow the source-defined gather branch, rather than invented data.

For **the plotted configuration only**, the fitted equations reduce to:

```
n ≤ 16: T_us = 2.786829 + 0.260255110 * n
n > 16: T_us = 3.020871 + 0.002988497 * n
```

These local coefficients illustrate how well each line can be fitted with fixed
hardware/tensor width. They are diagnostic fits, not per-GPU runtime lookup entries.
The separately exported portable fit uses one set of four coefficients shared
across every KAIST GPU/model/TP bundle and operates on nominal memory time `M`.
No device or model ID selects numerical coefficients. Increasing bandwidth lowers
the predicted cost within each branch. Latency is allowed to jump at dispatch.

## Fit and validation

| Corpus | Regime | Points | Training MAPE | Shape-CV MAPE | Shape-CV P95 |
| --- | --- | --- | --- | --- | --- |
| Plotted configuration only | small_index | 16 | 0.89% | 0.97% | 2.56% |
| Plotted configuration only | gather | 136 | 4.51% | 4.57% | 10.01% |
| KAIST, all five bundles | small_index | 80 | 8.97% | 9.50% | 20.70% |
| KAIST, all five bundles | gather | 680 | 8.24% | 8.29% | 18.20% |
| Vidur, A100/A40 development | small_index | 80 | 31.44% | 32.88% | 65.12% |
| Vidur, A100/A40 development | gather | 4064 | 37.78% | 37.84% | 65.90% |

Overall KAIST shape-CV error: **F8 10.49% →
F9 8.42%**. These are embedding-only scores.
Both are retrained under the same stricter split: equal `(token count, hidden width)`
queries stay in one fold across GPU, model, TP and vocabulary shard. Thus repeated
work cannot appear in both training and validation. F8 scores may differ slightly
from the original study, whose folds also included vocabulary-shard geometry.

F9 was proposed after examining these curves, so this is exploratory follow-up
validation. It is not a pristine confirmatory test of an untouched hypothesis.

| Transfer audit | Test points | F8 mean → F9 mean | F8 weighted → F9 weighted |
| --- | --- | --- | --- |
| KAIST: GPU_rtx4090 | 152 | 18.75% → 18.61% | 17.83% → 18.88% |
| KAIST: GPU_rtx6000 | 608 | 16.19% → 15.85% | 16.33% → 15.84% |
| KAIST: model_qwen32 | 304 | 11.83% → 9.23% | 10.79% → 8.65% |
| Vidur: H100 | 2072 | 47.48% → 46.55% | 54.53% → 54.32% |
| Vidur: CodeLlama_seen_GPUs | 2072 | 38.35% → 37.84% | 39.89% → 39.47% |
| Vidur: H100_CodeLlama | 1036 | 49.02% → 48.29% | 55.03% → 54.97% |

Vidur FP16 coefficients are fitted separately from KAIST BF16. Its H100 and
CodeLlama profiles are excluded from development fitting. These tests assess the
function form within that separate corpus; they do not validate KAIST coefficients
on H100. Full per-regime results, coefficients and input/source hashes are in
[results.json](results.json).

The local curves are cleaner than the portable fit. The switch addresses a real
missing software regime; it does not resolve cross-GPU intercept/rate differences,
unknown cache behavior, engine versions or timing methodology. It is a simple,
source-informed improvement rather than a universal GPU accuracy claim.

## Reproduce and use

```sh
.venv-sim/bin/python -B -m integration.fit_study.embedding_piecewise
.venv-sim/bin/python -B -m unittest integration.fit_study.test_embedding_piecewise -v
```

The working predictor is `predict_piecewise(model, token_count, memory_ns)` in
[embedding_piecewise.py](../../../integration/fit_study/embedding_piecewise.py).
It evaluates exactly one of the two fitted functions using the fixed switch.
The exported models include branch coefficients and five-fold models. The original
F0–F8 results and application baseline are preserved; this follow-up uses F9 for
its reported predictions and figures, without claiming a production deployment.

The switch is specific to the documented eager PyTorch embedding/index-select
path and its indexing conditions. Other software versions, fused implementations,
padding/CUDA-graph policies and quantization may dispatch differently. Source
references and the distinction between source evidence and actual kernel traces
are retained in the [dispatch diagnosis](../embedding-dispatch.md).

Credit: KAIST CASYS Lab / LLMServingSim 2.0 and Microsoft Vidur contributors for
published profiles; [ryeii (Ryan Z An)](https://github.com/ryeii) for ServeScope
and this experimental function implementation.
