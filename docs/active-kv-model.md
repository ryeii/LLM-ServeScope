# Independent active decode KV and retained prefixes

ServeScope models two independent KV lifetimes. **Retained prefixes** survive completed requests and can save future prefill query work. **Active decode KV** belongs to an admitted request and must be read during every subsequent attention step. Disabling prefix retention does not remove active KV. These are separate physical allocations; no page aliasing or zero-copy sharing is assumed.

## Interface and policies

The current default is **HBM-resident active KV with Dynamo-style retained-prefix
caching and GDS**, not active streaming from SSD. See [default cache lifecycle,
transfer mechanisms and validation limits](dynamo-alignment.md). SSD transport
in the equations below has two selectable mechanisms: the historical serialized
host-staged path, or direct GDS service computed as startup + max(storage service,
direct-link service). GDS reserves no host relay buffer. Prefix publication is
asynchronous by default; only blocking mode delays handoff. In asynchronous mode,
source prefill KV stays allocated until publication and handoff both finish.

- **Active decode KV**: HBM only (default), host DRAM backing, SSD backing, HBM + DRAM spill, or HBM + SSD spill.
- **Retained prefixes**: independently select no retention, single-tier, directional spill or inclusive copies. Existing prefix hit and eviction policies are unchanged.
- **KV I/O paths**: shared effective host–GPU bandwidth/latency, I/O chunk size and configured SSD service. Both lifetimes use the same FIFO SSD and host–GPU resources. Their byte/service/queue counters are reported separately.

Selecting either placement preserves the other placement, workload and serving batch caps. The result has separate active/prefix sections, per-request residency and nested formula explanations; the box graph includes backing initialization, per-step reads, append writes and their queue dependencies.

**Prefill working KV remains in HBM**, including restored prefixes. The active placement control applies to decode; “SSD backing” is not an entirely HBM-free inference engine. An oversized prefill still fails its HBM capacity check.

## Geometry and placement equations

For a GQA/MHA/MQA model, let `d` be KV bytes/element. Per-token payload on one decode GPU is:

```text
b_gpu = 2 × max(1, KV_heads / TP) × head_dim × d × layers / PP
shards = TP × PP × EP
Q_max_i = b_gpu × (input_i + output_i − 1)
```

The factor two represents K and V. `max(1, KV_heads/TP)` includes KV-head replication when TP exceeds the number of KV heads. PP shards layers; EP replicates KV. A request generating only one output token requires no decode KV allocation or backing transfer.

An entire prefill microbatch reserves maximum decode capacity atomically. Hybrid placement allocates `min(group Q_max, free HBM active budget)` per GPU. The same resident fraction is assigned to every member, uniformly across tokens and layers. The remainder is reserved in the backing store across all shards. Pure DRAM/SSD modes have zero resident fraction. Fractions stay pinned until completion: this is static partitioning, not dynamic page replacement or promotion.

```text
resident_fraction = allocated_HBM_per_GPU / group_Q_max_per_GPU
remote_fraction = 1 − resident_fraction
Q_remote_reserved = Σ Q_max_i × remote_fraction × shards
```

Backing admission waits for completed requests to release capacity; a microbatch that can never fit is rejected before scheduling. Reserving the maximum context is conservative and avoids an expanding decode allocation deadlock. A hybrid group can use more backing when earlier groups hold HBM; it waits rather than revising an admitted request’s residency.

## Initialization and every-token I/O

Prefill produces prompt KV in HBM. After any blocking prefix publication, backing admission reserves decode capacity. Disaggregated serving transfers the prompt through its configured handoff channel; initialization writes the remote prompt fraction using GDS or the selected staged path. Colocated serving also charges HBM migration for the resident fraction into its separate decode arena. Source prefill KV is released after initialization and any background prefix publication both finish; asynchronous prefix publication need not delay the first token.

For each decode pipeline stage and token:

```text
read_bytes = Σ b_gpu × (context − 1) × remote_fraction × TP × EP
append_bytes = Σ b_gpu × query_tokens × remote_fraction × TP × EP
stage_elapsed = read_queue_and_service + HBM_reference_GPU_service
                + append_queue_and_service
```

Existing KV is read before the current query appends its new KV. Host-staged SSD reads traverse SSD then host–GPU; writes reverse those phases. GDS uses a single direct reservation with startup + max(storage service, direct-link service). DRAM backing uses the configured effective host–GPU path, assumed to include host memory limits; there is no independent DRAM bandwidth model. Initialization spans all PP shards; later stages service their own layers. GPU kernel reference service retains its actual batch/query/context shape and fitted HBM stalls. External stalls are **added once**, without another synthetic HBM kernel-memory cost.

The SSD function is applied per configured I/O chunk:

```text
T_SSD(Q) = floor(Q/C) × f(C) + f(Q mod C)  [omit zero remainder]
f_manual(Q) = latency_us / 1000 + Q / (bandwidth_GBps × 10^6)  [ms]
T_host_GPU(Q) = latency_us / 1000 + Q / (effective_GBps × 10^6)
```

All SSD coefficients are configured assumptions. Queue wait is added separately, once. Reads and writes share a single QD1 SSD FIFO; prefix operations and active operations contend for that same resource. Aggregate host–GPU transfers reserve one shared FIFO. No transfer/compute overlap or parallel outstanding I/O is modeled.

## Bounded staging and event interpretation

Offload reserves the configured GPU staging buffer **per decode GPU**, separately from the hybrid resident budget. It must fit at least one I/O chunk. Large payloads represent serial tile cycles through this bounded buffer, not full-context materialization in HBM. The aggregate SSD-then-PCIe schedule models a serialized service sum; it does not expose individual tile interleaving or infer overlap. A larger buffer above one chunk currently changes memory capacity, not throughput.

During initialization, all decode-stage staging buffers are acquired before streaming; an occupied stage can delay the handoff. When separated, the handoff channel stays reserved until initialization finishes. The wire and device durations summarize serial tile cycles. The intermediate `wire_end_ms` is an accounting boundary, not an assertion that the whole prompt fits in GPU staging. The source prefill KV remains available until the entire initialization completes.

GPU-stage reservations include the I/O stalls while that stage is occupied. Nested storage events explain those same durations. Likewise staging-buffer reservations and handoff describe overlapping reservations. **Do not sum parent and child durations as disjoint wall time.** Formula cascades follow the executed request path and FIFO predecessors.

## Capacity and energy

HBM active and staging budgets are per GPU. Retained HBM capacity is per prefill GPU. Physical DRAM/SSD device totals and backing allocations are aggregate across decode shards:

```text
available_backing = total_device_bytes − enabled_retained_prefix_reservation
                    − relay_reservation_if_applicable
```

Host-staged SSD transfers reserve one I/O chunk of host DRAM relay space; direct GDS transfers reserve none. Inactive prefix tiers reserve zero. Admission never spends the prefix partition on active allocations. Memory charts conservatively combine peak components rather than claiming their peaks were simultaneous. Colocated prefill and decode have distinct arenas within one GPU capacity.

GPU resource occupancy includes stalled stages and reserved staging buffers. GPU activity energy charges the reference GPU service; I/O stall time uses GPU idle power. CPU, SSD, NIC and DMA energy are excluded, so tokens/J is **GPU-energy efficiency**, not full-system efficiency.

## Evidence and limitations

This active-offload composition is **unvalidated**. KAIST/Vidur fits calibrate
HBM-resident operator service, not streamed-attention kernels. Published Mooncake
request traces contain no device-service timings. SSD, DRAM, host–GPU and GDS
service use configured assumptions, with no measured error bound. Prefix-hit
errors describe trace-policy replay only. See [evidence coverage](storage-model.md).

The model provides an auditable serialized estimate. It excludes measured streaming-kernel overhead, tiling-dependent changes to kernel efficiency, transfer/compute overlap, fine-grained layer prefetch, dynamic active-KV eviction, remote object stores, cold-device service, NVMe/NAND/FTL mechanics, MLA and page-level execution. It can therefore overestimate time by serializing real overlap, and underestimate it by omitting real offload-kernel overhead. It is not a proven upper or lower bound.

One replica per pool is supported. Fixed uniform-length microbatches use the common restored-prefix floor; variable-length imported workloads require batch size one. GPU and prefix fitting validation remains unchanged. To calibrate this composition, obtain end-to-end HBM/DRAM/SSD runs with the same model, batch/context lengths, topology, offload kernel and scheduler; measure per-step reads, append writes, overlap, queueing and GPU/system power, then hold out complete configurations.

## Reproduction

[Controlled default matrix](active-kv-comparison.md) · [Importable example](../examples/active-kv-scenario.json) · [Native Python result](../examples/active-kv-result.json) · [Implementation](../backend/active_kv.py)

```bash
.venv/bin/python -B -m integration.storage.active_comparison
.venv/bin/python -B -m unittest tests.test_active_kv -v
SERVESCOPE_BROWSER_CHANNEL=chrome npx playwright test tests/browser/active-kv.spec.ts
SERVESCOPE_BROWSER_CHANNEL=chrome npm run test:pages -- active-kv.spec.ts
```

Regression coverage includes exact read/init/append bytes, backing capacity waits/releases, staging limits, PP-stage timing reconstruction, GPU-energy accounting, shared FIFO ordering, independent policies and native/browser parity. Tests establish implementation consistency, not physical accuracy. Exports include active-model source SHA-256 and storage-function provenance.

### Verification receipt · 2026-10-21

- Python: **74 tests passed**, including 12 active-KV cases.
- Local browser: **13 tests passed**, including optional native backend integration.
- Static Pages browser: **10 tests passed**, including Python/Pyodide numeric parity and mobile layouts.
- Prior default, retained-prefix and eviction-spill example performance metrics are unchanged. I/O and hit function coefficients, observations and validation errors are unchanged; the storage artifact runtime fingerprint was refreshed.

These are software verification results. They do not provide empirical active-offload accuracy. [Desktop interface](screenshots/pages-active-kv.png) · [Mobile interface](screenshots/pages-active-kv-mobile.png).
