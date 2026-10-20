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

## Separate corpus overviews

### KAIST · BF16

![KAIST · BF16 cross-validated accuracy](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-main.png?v=7d83f37f3fbd)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-main.svg?v=b27aa6cb5bd7).
### Vidur · FP16

![Vidur · FP16 cross-validated accuracy](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-main.png?v=465f07546ef7)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-main.svg?v=ed542daa7779).

Each figure has two panels: **predicted versus profiled time** and the
**cumulative absolute-error distribution**. KAIST and Vidur are never overlaid.
Dense prediction clouds use hexagonal bins colored by observation count; smaller
plots show individual points. Every eligible observation contributes to the bins,
error distributions and metrics; there is no point subsampling, outlier removal,
or tail clipping. The signed-residual middle panel has been removed.
The numerical validation results and production coefficients are unchanged.

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

![KAIST · BF16 by GPU](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-by-gpu.png?v=d68899dc6fc8)
![Vidur · FP16 by GPU](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-by-gpu.png?v=fca04b9d9d41)

The pooled score is dominated by Vidur's denser attention grids. Equal-operator
macro MAPE is **12.89%**. Per-GPU and
per-operator tables must accompany the pooled number. Latency-weighted error is
sum of absolute ns errors divided by sum of profiled ns, not serving-workload weighting.

## Unseen-GPU accuracy is a different test

![KAIST · BF16 leave-one-GPU-out accuracy](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-gpu-transfer.png?v=e764a5da17c0)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-gpu-transfer.svg?v=419e4466b20a).
![Vidur · FP16 leave-one-GPU-out accuracy](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-gpu-transfer.png?v=5d7943cd7af0)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-gpu-transfer.svg?v=81bdb76a56c0).

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

## Subcomponent figures, separated by profiling corpus

KAIST has 14 measured operator groups; Vidur has 10. A missing group has no figure
in that corpus. The deployed runtime retains a documented extrapolation/fallback
where a scenario requests an operator missing from its precision-specific corpus.
Families: F0/F1 additive; F2 max roofline; F3 smooth overlap; F4 tile-wave;
F5/F6 bounded shape transition; F7 cache-capacity transition; F8 corrected embedding
traffic; F9 independently fitted embedding dispatch branches.

### KAIST · BF16

| Subcomponent | Observations | MAPE | P95 | Deployed family |
| --- | --- | --- | --- | --- |
| Activation / SwiGLU | 760 | 17.95% | 59.44% | F7 |
| Decode attention | 783 | 8.22% | 24.08% | F4 |
| Prefill attention | 1,084 | 20.85% | 44.35% | F4 |
| MLP down projection | 760 | 6.20% | 15.80% | F5 |
| Embedding | 760 | 8.36% | 18.35% | F9 |
| Final normalization | 760 | 4.90% | 12.35% | F0 |
| MLP gate + up projection | 760 | 7.96% | 21.01% | F3 |
| Layer normalization | 760 | 11.19% | 36.36% | F4 |
| Language-model output head | 200 | 2.58% | 10.07% | F0 |
| Attention output projection | 760 | 7.59% | 19.54% | F5 |
| QK normalization | 304 | 0.90% | 2.61% | F4 |
| QKV projection | 760 | 9.92% | 23.38% | F1 |
| Rotary embedding | 760 | 10.94% | 34.74% | F0 |
| Sampling | 200 | 28.29% | 66.64% | F0 |

#### Activation / SwiGLU

![KAIST · BF16 Activation / SwiGLU](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-act_fn.png?v=91c91482f0b8)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-act_fn.svg?v=2ca191b8c1a5).

#### Decode attention

![KAIST · BF16 Decode attention](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-attention-decode.png?v=20e0baf88f17)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-attention-decode.svg?v=db1c475d309c).

#### Prefill attention

![KAIST · BF16 Prefill attention](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-attention-prefill.png?v=1a8d97e42f22)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-attention-prefill.svg?v=b5c4d51439d9).

#### MLP down projection

![KAIST · BF16 MLP down projection](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-down_proj.png?v=fef119d46559)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-down_proj.svg?v=bfe59264f135).

#### Embedding

![KAIST · BF16 Embedding](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-embedding.png?v=5355e108bda9)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-embedding.svg?v=a67b16c32b37).

#### Final normalization

![KAIST · BF16 Final normalization](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-final_layernorm.png?v=20d18331282a)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-final_layernorm.svg?v=2bf7b20d5167).

#### MLP gate + up projection

![KAIST · BF16 MLP gate + up projection](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-gate_up_proj.png?v=d1bbc5247c38)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-gate_up_proj.svg?v=b3386b649c1d).

#### Layer normalization

![KAIST · BF16 Layer normalization](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-layernorm.png?v=2919e1c84268)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-layernorm.svg?v=96756f9b8206).

#### Language-model output head

![KAIST · BF16 Language-model output head](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-lm_head.png?v=96fdf5cbcd23)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-lm_head.svg?v=96229e264180).

#### Attention output projection

![KAIST · BF16 Attention output projection](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-o_proj.png?v=575a3c5765ac)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-o_proj.svg?v=591a37b5fed4).

#### QK normalization

![KAIST · BF16 QK normalization](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-qk_norm.png?v=999490ca119a)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-qk_norm.svg?v=918c5e7ff0a4).

#### QKV projection

![KAIST · BF16 QKV projection](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-qkv_proj.png?v=d2a11c77265a)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-qkv_proj.svg?v=906fe985e7a3).

#### Rotary embedding

![KAIST · BF16 Rotary embedding](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-rotary_emb.png?v=b730de184ff2)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-rotary_emb.svg?v=1fd605cfe1c1).

#### Sampling

![KAIST · BF16 Sampling](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-sampler.png?v=7b8db557d52b)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/kaist-operator-sampler.svg?v=3e5532b61b9c).
### Vidur · FP16

| Subcomponent | Observations | MAPE | P95 | Deployed family |
| --- | --- | --- | --- | --- |
| Activation / SwiGLU | 9,324 | 8.68% | 28.62% | F0 |
| Decode attention | 364,032 | 9.65% | 26.02% | F0 |
| Prefill attention | 151,416 | 28.32% | 62.08% | F0 |
| MLP down projection | 9,324 | 8.60% | 24.04% | F3 |
| Embedding | 9,324 | 42.67% | 61.08% | F8 |
| MLP gate + up projection | 9,324 | 7.33% | 20.61% | F3 |
| Layer normalization | 9,324 | 11.58% | 26.71% | F5 |
| Attention output projection | 9,324 | 7.98% | 21.77% | F3 |
| QKV projection | 9,324 | 8.47% | 21.76% | F3 |
| Rotary embedding | 9,324 | 12.61% | 35.35% | F5 |

#### Activation / SwiGLU

![Vidur · FP16 Activation / SwiGLU](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-act_fn.png?v=48d939e241e5)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-act_fn.svg?v=2f1766d10988).

#### Decode attention

![Vidur · FP16 Decode attention](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-attention-decode.png?v=1ae567c088cf)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-attention-decode.svg?v=2aa527002a13).

#### Prefill attention

![Vidur · FP16 Prefill attention](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-attention-prefill.png?v=d44a37434f6f)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-attention-prefill.svg?v=aad3b9880dae).

#### MLP down projection

![Vidur · FP16 MLP down projection](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-down_proj.png?v=402bdc0adc4f)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-down_proj.svg?v=78c6aa1954ee).

#### Embedding

![Vidur · FP16 Embedding](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-embedding.png?v=b85ff139685f)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-embedding.svg?v=5232203775b8).

#### MLP gate + up projection

![Vidur · FP16 MLP gate + up projection](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-gate_up_proj.png?v=8c734d685ec8)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-gate_up_proj.svg?v=02e36a3e1a91).

#### Layer normalization

![Vidur · FP16 Layer normalization](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-layernorm.png?v=607c5248dcfa)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-layernorm.svg?v=09b0f0b04c78).

#### Attention output projection

![Vidur · FP16 Attention output projection](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-o_proj.png?v=927e6bd0625c)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-o_proj.svg?v=ce1f1966ecf6).

#### QKV projection

![Vidur · FP16 QKV projection](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-qkv_proj.png?v=d3aaca1400d1)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-qkv_proj.svg?v=0708545b3c1d).

#### Rotary embedding

![Vidur · FP16 Rotary embedding](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-rotary_emb.png?v=5f77305ed038)

[Vector figure](https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/vidur-operator-rotary_emb.svg?v=293dcbe6bc84).

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

“All data” across the two overview figures means **all imported compatible dense-decoder
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
