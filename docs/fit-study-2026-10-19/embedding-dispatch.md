# Embedding: why the profile has two regimes

The two branches in the selected-functions figure are from **one GPU, one model,
and TP1**. They are not two hardware configurations or different vocabulary shards.
The raw embedding measurements have a sharp drop between the last sampled point
at 16 tokens and the next point at 20 tokens:

| Tokens | RTX PRO 6000, Llama 8B, TP1 (µs) |
| --- | --- |
| 1 | 3.07167 |
| 8 | 4.81067 |
| 16 | 6.89067 |
| 20 | 2.90133 |
| 24 | 2.89033 |
| 32 | 2.88033 |

[Exact published CSV](https://github.com/casys-kaist/LLMServingSim/blob/b09c7aaab4b072d73136641841cc2688616c49eb/profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/tp1/dense.csv).
The same 16-to-20-token drop is present in all five retained KAIST bundles,
including RTX 4090 and both Qwen32 TP settings.

## Source-supported explanation

The vLLM 0.28.0 unquantized embedding method calls `F.embedding`. PyTorch's embedding
implementation selects rows with `index_select`. In PyTorch 2.13.0, CUDA index
selection takes the small-index path when there are at most 16 indices and the
32-bit indexing conditions hold. Above that threshold, it takes a gather path.
The small-index kernel loops over selected indices while parallelizing the row;
the other path exposes more parallel work across selected rows. The algorithm
change can lower elapsed time even when the token count increases.

Sources:

- [vLLM 0.28.0 embedding method, lines 77–78](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/vocab_parallel_embedding.py#L77-L78).
- [PyTorch 2.13.0 embedding → index_select, lines 35–52](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/Embedding.cpp#L35-L52).
- [Small-index kernel loop](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/Indexing.cu#L1502-L1535).
- [Dispatch predicate and gather fallback](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/Indexing.cu#L1621-L1658).

Inspected Git blob identities: PyTorch `Indexing.cu`
`01342250e97354e82d0f52fd4972549ad6e13477`; vLLM embedding source
`79695084e4f23b6878dbeb79b16dd488b6bd5023`.

This is a **strongly supported explanation, not direct kernel-trace confirmation**.
The retained hardware YAML lists PyTorch 2.13.0, but its collection date differs
from the CSV profiling dates; it is not a per-run build manifest. The CSVs contain
aggregate operator times, not kernel names. The matching threshold and call path
support the diagnosis; a kernel trace at 16 and 17 tokens would confirm the exact
execution path. The current measurement grid has no 17-, 18-, or 19-token samples.

## Implication for fitting

F8 fixes an embedding traffic/TP representation error. It does not capture this
kernel dispatch. A single smooth increasing line necessarily compromises between
the two regimes. The previous panel title, “vocabulary sharding,” failed to
identify the visible discontinuity and was misleading for this TP1 slice.

A defensible next candidate is a piecewise function selected by the documented
software dispatch predicate, with separate small-index and gather cost functions.
The threshold should come from the software implementation/version, not be chosen
to match these timing points. Its coefficients still need grouped validation and
hardware-transfer tests. No new fit has been claimed or deployed in this follow-up.

The plot is now titled “Embedding: kernel-dispatch regimes,” with the two groups
marked separately and the source-defined boundary annotated. The earlier study's
numerical results and baseline remain unchanged.
