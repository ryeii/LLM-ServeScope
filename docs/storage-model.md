# Request-trace evidence and storage timing assumptions

Mooncake means its **published request traces** in ServeScope. The repository
contains the unchanged FAST'25 request dumps and their source manifest; no
storage benchmark is used to calibrate device timing.

## What the published traces supply

| Trace | Requests | Origin |
| --- | ---: | --- |
| Conversation | 12,031 | Sampled production workload |
| Tool/agent | 23,608 | Sampled production workload |
| Synthetic | 3,993 | Upstream generated workload |

Each record supplies `timestamp` (arrival milliseconds), `input_length`,
`output_length`, and remapped `hash_ids` for 512-token prefix blocks. It does
not contain raw text/token IDs, completion timestamps, observed cache hits,
SSD read/write events, transfer durations, queue depths, device addresses or
power measurements. The data cannot identify GDS versus host staging.

Sources: [published trace release](https://github.com/kvcache-ai/Mooncake/tree/e4229791f39d1f4cdb00df1b28366130f742deb0/FAST25-release),
[retained source manifest](../third_party/mooncake-traces/manifest.json), and
[unchanged trace archive](../third_party/mooncake-traces).

## How ServeScope uses them

Load a contiguous window under **Retained prefixes → Published Mooncake workload**.
Arrival differences, input/output lengths and ordered prefix hashes are
preserved. The first selected arrival defines simulation time zero. Source file
checksum, revision and original line numbers accompany exports. No text or token
IDs are invented. Variable-length windows require explicit batch size one.

Import the [published conversation example](../examples/mooncake-conversation-scenario.json)
for the first four unchanged requests. Its [result summary](../examples/mooncake-conversation-result-summary.json)
records original arrivals/lengths and simulated outcomes separately. Reproduce it
with `python -B -m integration.storage.published_trace_example`; the device timing
inputs remain explicitly configured assumptions.

Runtime cache hits depend on ready contiguous prefixes and the chosen capacity,
replacement policy and publication schedule. Restored KV changes the actual
prefill query shape; storage queues and admission affect request timestamps.
Those are **simulated consequences**, not events contained in the source dump.
The source keeps its `dumped-as-is-trace` classification; the serving result is
clearly a simulated replay. Generated demonstration workloads are labeled
separately and are not published Mooncake records.

The independent [prefix-reuse study](trace-reuse/README.md) fits functions to
explicit LRU replay of the published traces. Its held-out errors describe only
that policy approximation, not physical storage, transfer or serving accuracy.
Runtime hits never come from the fitted forecast.

## Device timing has no calibration from these traces

| Component | Current timing input | Measured device error bound |
| --- | --- | --- |
| SSD reads/writes | Configured read/write latency and effective bandwidth | None |
| Host–GPU copies | Configured startup and effective bandwidth | None |
| Direct SSD–GPU GDS | Configured startup + max(SSD service, direct-link service) | None |
| Explicit HBM copies | Configured HBM bandwidth | None |
| Network/handoff | Configured fabric rates, latency and topology | None from request traces |
| GPU operators | Separate KAIST/Vidur profile fits | See [GPU validation](accuracy.md) |

With milliseconds and decimal GB/s, affine service is
`latency_us/1000 + bytes/(bandwidth_GBps × 10^6)`. SSD service is charged per
configured I/O chunk; queue wait is separate. GDS combines the storage and
direct-link streaming bottleneck once, without a host bounce copy. These equations
are **scenario assumptions**, not functions fitted to request-arrival times.

GDS remains the default GPU–SSD transport; active decode KV remains in HBM by
default. [Dynamo-style cache lifecycle and remaining differences](dynamo-alignment.md).
Experimental active offload has no end-to-end validation.

To calibrate device timing, a separate dataset must specify target hardware,
software, transfer size, direction, concurrency and measured duration. The
published request traces cannot substitute for it. Historical exports requesting
a removed device-fit profile fail validation rather than silently using new
coefficients.

Published traces: [Mooncake contributors](https://github.com/kvcache-ai/Mooncake).
Trace analysis and behavioral model: [ryeii (Ryan Z An)](https://github.com/ryeii).
