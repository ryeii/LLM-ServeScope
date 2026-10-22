# Dynamo-style cache behavior and direct SSD–GPU transfers

21 October 2026 · ryeii (Ryan Z An)

ServeScope defaults to **GPU-resident active decode KV, tiered retained prefixes,
asynchronous publication and a configured GDS path**. This independently
implemented behavioral approximation is inspired by NVIDIA Dynamo KVBM. It does
not execute Dynamo or inherit its performance validation.

## Lifetimes, policy and readiness

Active attention reads KV from HBM. Lower tiers retain reusable prompt prefixes
between requests. A ready prefix is restored into prefill HBM before suffix
computation. The default does not read active attention KV from SSD every token.

The **Dynamo-style tiered cache** keeps independent LRU copies in HBM and host
DRAM, and admits a block to SSD after a configurable number of distinct request
references (default: two). This simple filter does not reproduce Dynamo's LFU
aging or priorities. Only completed copies are visible to a lookup; hits are
determined by actual cache state, not sampled from a fitted hit curve.

Up to four publication jobs can be in flight by default. Placement commits are
serialized to protect capacity/readiness, while background copies can overlap
GPU computation, handoff and independent restores. Each physical I/O path still
has one shared FIFO bandwidth budget: four jobs do not imply four times the
bandwidth or a calibrated NVMe queue depth of four.

Source prefill KV stays allocated until **both handoff and publication finish**.
This can cause capacity waits. The first response need not wait for background
publication. Exports distinguish its request-path barrier, background finish and
actual source release. Throughput uses the last response completion, not the
background drain time. Execution charts also include background I/O draining
after that response.

Old single-tier/spill layouts and blocking publication remain available.
Historical exports without a transfer-path field retain their original
host-staged, blocking timing. They are not silently converted to the new preset.
New GDS/asynchronous runs use export schema `servescope.behavioral.v6` and
include transfer/storage source hashes. Legacy staged/blocking runs retain
their previous schema version.

## Transfer mechanisms

| Mechanism | GPU read path | Service model |
| --- | --- | --- |
| Host-staged | SSD → host DRAM → GPU | SSD service + host–GPU service |
| GDS, default | SSD → GPU HBM | Startup + max(SSD service, direct-link service) |

Writes reverse direction. SSD–DRAM copies still use the storage path; DRAM–GPU
copies still use the host–GPU path. GDS removes the host relay only when SSD and
GPU memory are the endpoints. Queue waits are charged separately.

Time is in milliseconds; bandwidth uses decimal GB/s:

```text
T_storage(Q) = floor(Q/C) × f(C) + f(Q mod C)   [omit zero remainder]
f_manual(Q)  = SSD_latency_us/1000 + Q/(SSD_GBps × 10^6)
T_link(Q)    = Q/(GDS_GBps × 10^6)
T_GDS(Q)     = GDS_startup_us/1000 + max(T_storage(Q), T_link(Q))
T_hostGPU(Q) = hostGPU_latency_us/1000 + Q/(hostGPU_GBps × 10^6)
```

The max is an ideal streaming bottleneck approximation. The whole direct
operation reserves the shared SSD path, preventing direct and host-targeted
operations from each spending the full SSD bandwidth concurrently. It does not
model detailed PCIe/NUMA contention, GDS tile scheduling or CPU control work.
The direct link defaults to **24 GB/s and 10 μs startup**, editable in **KV I/O
paths**. GDS requires compatible hardware, filesystem and drivers; ServeScope
does not probe or enable them. Direct transfer is not guaranteed faster for
every transfer size or deployment.

The measured Mooncake fit covers warm buffered file I/O on a development Mac.
It cannot calibrate GDS, so it is available only with the host-staged path.
Selecting GDS switches to configured SSD coefficients. Existing file-path error
figures do not validate direct transfers or end-to-end serving.

## Alignment and remaining differences

| Behavior | ServeScope | Dynamo reference |
| --- | --- | --- |
| Working attention KV | HBM by default | Cached blocks onboarded into GPU memory |
| Prefix tiers | HBM, host abstraction, SSD | GPU, pinned host, disk, object storage |
| Disk transport | Host-staged or GDS assumption | NIXL backends, optional GDS |
| Background work | Async publication; restore before use | Async transfer manager with connector scheduling |
| Policy | Independent LRU copies; reference-count filter | Presence/LFU/priority policies |
| I/O concurrency | Bounded jobs; one physical-path reservation | Actual batched/concurrent NIXL transfers |
| Calibration | GPU fits; optional local staged file-I/O fit | Actual deployment hardware/software |

Experimental **Active decode KV → SSD backing** remains a separate scenario:
it streams remote KV every decode token. GDS changes that transport, without
making the policy equivalent to prefix offloading. Its kernel overhead and
transfer–compute overlap have no measured validation.

The **Executed resource service** chart separates GPU service, GDS, SSD–host,
host–GPU, HBM copies and pipeline/handoff service. Active parent I/O stalls are
not charged again as GPU service. Aggregate resource time is not request latency;
resources overlap. The old **Nominal GPU resource work** diagnostic remains
expandable and uses an analytical HBM-reference workload.

## Credits and reproduction

Architectural references: [NVIDIA KVBM design (v1.2.0)](https://docs.nvidia.com/dynamo/v1.2.0/design-docs/component-design/kvbm-design),
[configuration (v1.5.1)](https://docs.dynamo.nvidia.com/dynamo/v1.5.1/reference/components/kvbm-configuration)
and [GDS design guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html).
GDS is optional in Dynamo; **ServeScope chooses it as its own default**.
No Dynamo/NIXL source or measured Dynamo timing data was imported. Potential
future calibration: [NIXL KVBench GDS profiling](https://github.com/ai-dynamo/nixl/blob/main/benchmark/kvbench/docs/tutorial-gds.md).
Implementation and experiments: [ryeii (Ryan Z An)](https://github.com/ryeii).

```bash
.venv/bin/python -B -m integration.storage.dynamo_comparison
.venv/bin/python -B -m unittest tests.test_gds -v
SERVESCOPE_BROWSER_CHANNEL=chrome npx playwright test tests/browser/gds.spec.ts
npm run build:pages
SERVESCOPE_BROWSER_CHANNEL=chrome npm run test:pages -- gds.spec.ts
```

Comparison outputs are **model predictions, not measurements**. Tests establish
routing, accounting, capacity lifetimes and Python/browser parity, not physical
GDS or Dynamo accuracy. KAIST, Vidur, Mooncake and SimPy credits remain separate.

Verification on 21 October 2026: **81 Python tests, 15 local browser tests and
12 static Pages browser tests passed**. The GDS workflow was rechecked in both
editions after adding v6 source-hash metadata. The local file-I/O coefficients
and held-out errors are unchanged; only their runtime provenance was refreshed.
