# Mooncake-derived prefix-reuse assumption

These are exact descriptive counts from the pinned published request traces,
not measured hardware cache hits or a fitted service-time model.

| Workload | Requests | Input tokens | Potentially reusable tokens | Token-weighted potential |
| --- | ---: | ---: | ---: | ---: |
| Conversation | 12,031 | 144,793,823 | 54,058,496 | 37.33% |
| Tool & agent | 23,608 | 202,940,084 | 115,716,608 | 57.02% |
| Synthetic workload | 3,993 | 61,194,628 | 39,802,880 | 65.04% |

![Earlier-arrival prefix reuse potential](https://ryeii.github.io/LLM-ServeScope/prefix-reuse/reuse-potential.png)

## Counting rule and evidence

Start with empty history and inspect requests in timestamp order. Only complete
512-token leading prefix blocks present at an earlier timestamp can be reused.
All tied arrivals are checked before adding that timestamp's blocks. History has
no eviction and at least one prompt token is recomputed for logits. The denominator
includes every input token, including incomplete blocks. Prefix identities include
their ancestors, so matching a suffix alone does not count as reusable context.

`fraction = sum(potentially reusable input tokens) / sum(all input tokens)`.

These percentages are ideal earlier-arrival reuse potential. Previous requests
may not have completed prefill, finite HBM may not hold the history, and new
workloads can differ. Six equal-duration descriptive windows in the
[audited summary](../../backend/prefix-reuse.json) carry earlier history forward;
their variation is not a held-out error or confidence interval. No regression
or accuracy percentage is appropriate for exact counts of this fixed dataset.

The conversation and tool/agent traces are production samples; the synthetic
trace is upstream-generated. Raw text and token IDs are absent. Source arrivals,
lengths and identities remain unchanged in the [trace archive](../../third_party/mooncake-traces).

## Use in ServeScope

The default is **No reuse**. Selecting a Mooncake source applies its full-trace
fraction to the representative uniform workload:

`reused = min(input_length - 1, floor(input_length * fraction))`.

Cached KV is assumed already ready in each request's HBM allocation, including at
the start of the scenario. This is an explicitly warm-input assumption, not a
simulation of a persistent cache, eviction or cache readiness. The scenario does
not replay the trace's original arrivals, lengths or per-request matches.

Prefill executes only the remaining query tokens. Attention still reads the
complete preceding context; decode length, complete per-request KV capacity and
full-context handoff payload do not shrink. We do not multiply throughput by a
cache-hit percentage. The fitted GPU function evaluates the actual shortened
query/full-context shapes and SimPy schedules their service.

Nonlinear timing means a scalar cannot reproduce a heterogeneous workload's mean.
Resulting shapes may be extrapolations. GPU operator validation does not establish
the accuracy of the reuse assumption or the resulting whole-serving speedup.
Capacity accounts conservatively for complete per-request KV, including reused
prefixes; there is no separate unbounded physical cache allocation in the model.

## Reproduction and credits

Run `python -B -m integration.analyze_prefix_reuse`. The standard-library analysis
checks all source-file SHA-256 values and records its source hash and upstream
revision in the summary. Run `python -B -m integration.plot_prefix_reuse` to refresh
the figure using matplotlib. Counting edge cases and deployed Python/browser
parity are covered by tests.

Source: [Mooncake FAST'25 request traces](https://github.com/kvcache-ai/Mooncake/tree/e4229791f39d1f4cdb00df1b28366130f742deb0/FAST25-release).
Original trace data: [Mooncake contributors](https://github.com/kvcache-ai/Mooncake).
Analysis and ServeScope: [ryeii (Ryan Z An)](https://github.com/ryeii).
GPU timing evidence remains separately credited to
[KAIST LLMServingSim 2.0](https://github.com/casys-kaist/LLMServingSim) and
[Microsoft Vidur](https://github.com/microsoft/vidur).
