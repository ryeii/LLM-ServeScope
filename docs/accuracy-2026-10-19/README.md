# How accurate is LLM-ServeScope?

GPU operator timing evaluation · 19 October 2026 · [ryeii (Ryan Z An)](https://github.com/ryeii)

The production fitting procedure has **14.86% mean absolute relative
operator-time error**, **9.54% median error**, and
**49.52% P95 error** across **599,451 imported compatible
observations**. **74.8%** of predictions fall within ±20%.
These are grouped nested cross-validation results, not training residuals.
They describe GPU operator service time under recorded profile conditions.
They are **not an error guarantee for serving throughput, request latency, energy,
new architectures, or a customer workload distribution**.

![Main cross-validated residual graph](main-residuals.png)

[Vector figure](main-residuals.svg). Left: prediction versus profile; center: signed
relative residual; right: cumulative absolute error. Every eligible observation
enters the plots and metrics; no tail clipping, random point subsampling or outlier
removal is used. Positive signed residual means overprediction. Rasterization of
scatter marks reduces vector-file size without discarding observations.

## What data and model are being tested?

- **KAIST / BF16:** all 9,411 imported pure-phase operator observations from
  RTX PRO 6000 and RTX 4090; Llama 3.1 8B and Qwen3 32B; TP1/2 where published.
- **Vidur / FP16:** all 590,040 distinct imported observations from A100, A40 and
  H100; Llama 2 7B/70B and CodeLlama 34B; TP1/2/4/8. This expands the previous
  development-only Vidur study to include its previously reserved GPU/model groups.
- Original Vidur observations: 602,856 before aggregation of repeated exact
  coordinates within the same model/GPU/TP. Published timing means are converted
  from ms to ns. KAIST pure-phase extraction preserves upstream ns and TP-stable rules.
- Historical precision/software cohorts are **fitted separately**, never pooled.
  Production selects Vidur coefficients for FP16, KAIST coefficients for BF16.
  Other precisions extrapolate KAIST functions. Operators missing from Vidur use
  an explicitly labeled KAIST extrapolation. No GPU or model ID selects coefficients.
- Existing F0–F8 families are evaluated; embedding also considers F9's two affine
  functions with a fixed ≤16/>16 token switch. The fixed switch is source-informed,
  not an optimized validation threshold. Runtime shapes, rates and physical L2
  capacity evaluate the selected formula. If an F7 input lacks physical L2 capacity,
  runtime uses the exported additive fit and records the fallback reason.

### Cross-validation and final training

1. Five **outer folds** group canonical executed operator geometry across GPU,
   model and TP. Embedding uses `(token count, hidden width)` to prevent vocabulary
   shards or TP copies of equal work crossing folds.
2. For each outer fold, **three inner folds use only its training observations**
   to choose the family and any transition scale. We choose the fewest parameters
   within one percentage point of minimum inner-CV MAPE, then lower MAPE for ties.
3. The chosen function is refitted on that outer training set and predicts the
   untouched outer fold. Every eligible observation receives one out-of-fold prediction.
4. After evaluation, family/scale selection runs on the full cohort's inner folds,
   and the selected functions are refitted on **all observations in their cohort**
   for deployment. Those full-data coefficients do not generate the CV graph.

This evaluates the **selection and fitting procedure**, since selected families
can differ among folds. It does not imply that the exact final full-data coefficient
vector was tested on data excluded from that final fit. The function space was
informed by earlier exploration of these profiles: this is exploratory nested CV,
not a preregistered or newly collected confirmatory dataset.

The re-fitted additive F0 comparator scores **15.21%**
mean error on the same folds, versus **14.86%** for selection.
That comparator also gets a separate FP16 fit; it is not the former KAIST-only
production model evaluated indiscriminately on another profiler.

## Accuracy by corpus and GPU

| Corpus | Observations | MAPE | P95 absolute relative error |
| --- | --- | --- | --- |
| KAIST | 9,411 | 10.64% | 35.76% |
| VIDUR | 590,040 | 14.93% | 49.68% |

| GPU family | Observations | MAPE | P95 | Latency-weighted error |
| --- | --- | --- | --- | --- |
| a100 | 196,680 | 14.42% | 40.52% | 12.25% |
| a40 | 196,680 | 11.94% | 40.46% | 10.25% |
| h100 | 196,680 | 18.41% | 58.28% | 26.23% |
| rtx4090 | 1,643 | 13.31% | 37.42% | 11.13% |
| rtx6000 | 7,768 | 10.07% | 35.62% | 9.19% |

![Cross-validated predictions by GPU](by-gpu.png)

The pooled score is dominated by Vidur's denser attention grids. Equal-operator
macro MAPE is **12.89%**. Per-GPU and
per-operator tables must accompany the pooled number. Latency-weighted error is
sum of absolute ns errors divided by sum of profiled ns, not serving-workload weighting.

## Unseen-GPU accuracy is a different test

![Leave-one-GPU-out residual graph](gpu-transfer-residuals.png)

| Excluded GPU/cohort | Scored observations | MAPE | P95 | Unsupported groups |
| --- | --- | --- | --- | --- |
| kaist:rtx4090 | 1,643 | 21.54% | 45.95% | None |
| kaist:rtx6000 | 7,464 | 19.29% | 51.05% | qk_norm |
| vidur:a100 | 196,680 | 16.77% | 44.72% | None |
| vidur:a40 | 196,680 | 23.83% | 102.98% | None |
| vidur:h100 | 196,680 | 20.03% | 61.20% | None |

The target GPU is absent from coefficient fitting **and** function/scale selection.
QK normalization cannot be learned when RTX PRO 6000 is withheld: RTX 4090 only
contains Llama, which lacks this operation. Those 304 targets are explicitly
unsupported rather than scored as a successful prediction. Thus this transfer
figure has **599,147** points, versus 599,451 in shape CV.
KAIST also confounds hardware with software version. This audit is not evidence
that peak specifications alone predict new GPU architectures within a fixed error.

## Main residual graphs for every subcomponent

| Subcomponent | Observations | MAPE | P95 | Deployed KAIST family | Deployed Vidur family |
| --- | --- | --- | --- | --- | --- |
| Activation / SwiGLU | 10,084 | 9.38% | 32.45% | F7 | F0 |
| Decode attention | 364,815 | 9.64% | 26.02% | F4 | F0 |
| Prefill attention | 152,500 | 28.27% | 62.05% | F4 | F0 |
| MLP down projection | 10,084 | 8.42% | 23.66% | F5 | F3 |
| Embedding | 10,084 | 40.09% | 60.99% | F9 | F8 |
| Final normalization | 760 | 4.90% | 12.35% | F0 | — |
| MLP gate + up projection | 10,084 | 7.38% | 20.68% | F3 | F3 |
| Layer normalization | 10,084 | 11.55% | 26.96% | F4 | F5 |
| Language-model output head | 200 | 2.58% | 10.07% | F0 | — |
| Attention output projection | 10,084 | 7.95% | 21.55% | F5 | F3 |
| QK normalization | 304 | 0.90% | 2.61% | F4 | — |
| QKV projection | 10,084 | 8.58% | 21.93% | F1 | F3 |
| Rotary embedding | 10,084 | 12.49% | 35.35% | F0 | F5 |
| Sampling | 200 | 28.29% | 66.64% | F0 | — |

A dash means no measured operator group in that corpus. The deployed runtime
retains a documented extrapolation/fallback where a scenario requests such a group.
Families: F0/F1 additive; F2 max roofline; F3 smooth overlap; F4 tile-wave;
F5/F6 bounded shape transition; F7 cache-capacity transition; F8 corrected embedding
traffic; F9 independently fitted embedding dispatch branches.

### Activation / SwiGLU

![Activation / SwiGLU residuals](operator-act_fn.png)

[Vector figure](operator-act_fn.svg).
### Decode attention

![Decode attention residuals](operator-attention-decode.png)

[Vector figure](operator-attention-decode.svg).
### Prefill attention

![Prefill attention residuals](operator-attention-prefill.png)

[Vector figure](operator-attention-prefill.svg).
### MLP down projection

![MLP down projection residuals](operator-down_proj.png)

[Vector figure](operator-down_proj.svg).
### Embedding

![Embedding residuals](operator-embedding.png)

[Vector figure](operator-embedding.svg).
### Final normalization

![Final normalization residuals](operator-final_layernorm.png)

[Vector figure](operator-final_layernorm.svg).
### MLP gate + up projection

![MLP gate + up projection residuals](operator-gate_up_proj.png)

[Vector figure](operator-gate_up_proj.svg).
### Layer normalization

![Layer normalization residuals](operator-layernorm.png)

[Vector figure](operator-layernorm.svg).
### Language-model output head

![Language-model output head residuals](operator-lm_head.png)

[Vector figure](operator-lm_head.svg).
### Attention output projection

![Attention output projection residuals](operator-o_proj.png)

[Vector figure](operator-o_proj.svg).
### QK normalization

![QK normalization residuals](operator-qk_norm.png)

[Vector figure](operator-qk_norm.svg).
### QKV projection

![QKV projection residuals](operator-qkv_proj.png)

[Vector figure](operator-qkv_proj.svg).
### Rotary embedding

![Rotary embedding residuals](operator-rotary_emb.png)

[Vector figure](operator-rotary_emb.svg).
### Sampling

![Sampling residuals](operator-sampler.png)

[Vector figure](operator-sampler.svg).

## Serving integration recheck (separate evidence)

The final selected KAIST functions were also compared with nine existing native
KAIST/ASTRA profile-backed serving scenarios: Llama 3.1 8B BF16, TP1/2 on RTX PRO
6000 and TP1 on RTX 4090; batches 1/4 and inputs 128/512. Mean differences are:

| Metric | Previous deployed F0 | Selected production model |
| --- | --- | --- |
| Output throughput | 4.10% | 4.83% |
| TTFT | 14.83% | 11.60% |
| TPOT | 4.65% | 3.87% |

Throughput agreement worsens slightly while TTFT/TPOT improve. No coefficients
were adjusted to improve these nine serving targets. This is a simulator-to-simulator
integration check on known profiled identities, not new end-to-end GPU measurements
or an unseen-GPU test. The native artifact hashes were reverified. Read the
[recheck metrics and source hashes](serving-recheck-comparison.json),
[scenario CSV](serving-recheck-comparison.csv), and
[operator CSV](serving-recheck-operators.csv). Differences also include scheduling,
communications and endpoint semantics; these scores cannot be inferred from the
main operator residual plot.

## Coverage exclusions and limits

“All data” in the main graph means **all imported compatible dense-decoder
observations**, not every CSV row present in an upstream repository. The following
locally available specialized bundles remain outside the runtime cost contract:

| Locally available excluded bundle | CSV rows available; not comparable operator count |
| --- | --- |
| RTXPRO6000/MiniMaxAI/MiniMax-M3/bf16/tp1 | 28289 |
| RTXPRO6000/Qwen/Qwen3-30B-A3B-Instruct-2507/bf16/tp1 | 52877 |
| RTXPRO6000/Qwen/Qwen3-30B-A3B-Instruct-2507/bf16/tp2 | 52678 |
| RTXPRO6000/Qwen/Qwen3.8-27B/bf16/tp1 | 11444 |
| RTXPRO6000/Qwen/Qwen3.8-27B/bf16/tp2 | 11444 |
| RTXPRO6000/deepseek-ai/DeepSeek-V3.2-Exp/fp8/tp1 | 202466 |
| RTXPRO6000/deepseek-ai/DeepSeek-V3.2-Exp-16L64E/fp8/tp1 | 145796 |
| RTXPRO6000/zai-org/GLM-5/bf16/tp1 | 20470 |

These MoE/MLA/hybrid/MTP tables have incompatible operator structures and were
neither silently mapped onto dense kernels nor counted as accurate. Exact CSV
paths, row counts and hashes are in [coverage-exclusions.json](coverage-exclusions.json).
Mixed-phase and heterogeneous-history KAIST attention also remain outside the
pure-phase imported corpus. Published hardware specifications and model configs
are inputs, not additional timing measurements.

No new GPU inference was run. No profiled energy, power-counter, network-collective,
cache-miss or memory-controller measurements were used to calibrate those outputs.
SimPy serving queues and analytic energy/network estimates require separate checks.
Historical environment metadata is incomplete, especially for Vidur. The FP16
profile-reference model presets preserve Vidur's padded 32768 vocabulary; these
are labeled as profile geometry, not a claim of a new checkpoint measurement.

## Reproduce and inspect

```sh
# Python 3.10 and the pinned study dependencies; CPU only.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv-sim/bin/python -u -B -m integration.production_fit.train
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv-sim/bin/python -u -B -m integration.production_fit.transfer
.venv-sim/bin/python -B -m integration.production_fit.export
.venv-sim/bin/python -B -m integration.production_fit.report
.venv-sim/bin/python -B -m unittest integration.production_fit.test_production_fit -v
```

[All nested validation models and selections](validation.json),
[every out-of-fold prediction](cross-validated-predictions.jsonl.gz),
[GPU-transfer models and selections](gpu-transfer-validation.json),
[GPU-transfer predictions](gpu-transfer-predictions.jsonl.gz),
[production coefficient artifact](../../backend/gpu-fit.json), and
[implementation](../../integration/production_fit). Caches are ignored and checked
against input-array and source hashes. Graphs can be recreated from archived
predictions without refitting. Runtime uses standard-library Python only.

Sources: [KAIST pinned profiles](https://github.com/casys-kaist/LLMServingSim/tree/b09c7aaab4b072d73136641841cc2688616c49eb/profiler/perf),
[Vidur pinned profiles](https://github.com/microsoft/vidur/tree/abae7f63aa857300f5cdc6f5e0d27860cd24721b/data/profiling/compute),
[Vidur archive manifest](../../third_party/vidur-profiles/manifest.json).
Profile authors receive credit under retained MIT licenses. ServeScope's function
selection, implementation, UI and evaluation: [ryeii (Ryan Z An)](https://github.com/ryeii).
