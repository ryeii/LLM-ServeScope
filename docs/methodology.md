# Behavioral methodology and calibration

The aim is strategic sensitivity analysis, analogous to a semiconductor behavioral
model. This is not a page-level memory simulator, a cycle-accurate GPU emulator,
or a validated predictor of a new chip. Every run separates architecture/workload
facts from equations, scenario assumptions and native profile-derived results.

## Credits

LLM-ServeScope's behavioral modeling extension, scenario controls, visualizations,
interface, browser execution layer and native backend integration are by
[ryeii (Ryan Z An)](https://github.com/ryeii). The optional LLMServingSim 2.0
reference backend is by CASYS Lab at KAIST (Jaehong Cho, Hyunmin Choi, Guseul Heo
and Jongse Park), using their scheduler, measured profiles, Chakra DAG generation
and ASTRA-Sim integration. SimPy supplies discrete-event scheduling for the
behavioral engine. See [full credits and notices](../THIRD_PARTY_NOTICES.md).

## Fitted GPU service-time function

The default mode evaluates the same nonnegative compute/memory function for every
GPU and input. Coefficients are learned offline from KAIST profiles; no timing table
or residual anchor is consulted during simulation. Profiled configurations also
have fitted prediction error. See [equations, physical constraints, assumptions
and withheld-data validation](calibration.md). SimPy serving, network, capacity/
traffic and energy remain separate behavioral models. The equations below describe
the retained original roofline mode and nominal resource-work estimates.

## Original roofline operation model

Model geometry is taken from pinned KAIST or publisher configuration files (or
explicitly user-defined decoder inputs): hidden and
intermediate widths, layers, attention/KV heads, head dimension, vocabulary, and
expert count/top-k. Dense and MoE linear work uses matrix multiplication FLOP
counts; prefill attention includes the causal triangular sequence length.
KV precision is independent of weights: BF16/FP16 occupies two bytes and FP8 one byte per element. Conversion overhead and quality changes are excluded.

For each operation and GPU:

- Compute time = FLOPs / (selected dense peak × compute clock scale × assumed efficiency × tile utilization × engine factor).
- Per-phase engine overrides select the same disclosed efficiency/launch policy as the workload default. Extra hardware launch overhead is added per operation. Clock scaling does not change TDP automatically.
- Tied embedding/head storage is counted once. Qwen3 QK normalization and routed-MoE router projections are included in the layer cost and DAG; router selection/softmax overhead remains abstracted.
- Tile utilization estimates available 64-token-by-tensor-width tiles relative
  to two resident waves across the configured SM count, with an explicit 8% floor.
- Vector operations use FP32 throughput. Norm/rotary/activation traffic is
  approximate. Quantization dequantization and numerical quality are not modeled.
- Memory time = modeled HBM bytes / (peak HBM bandwidth × assumed efficiency).
- Service time = max(compute time, memory time) + launch overhead.
- Collectives add separate ring all-reduce/all-to-all costs with efficiency,
  fixed latency and dispatch overhead. Both attention/FFN reductions and MoE
  dispatch/combine are charged separately.
- Dense parameter payload is consumed per module/batch. MoE uses uniform routing
  and an expected unique-expert count; no captured expert routing is claimed.
- Logit projection is charged on the last PP stage at the prefill boundary and
  every subsequent token. Embedding lookup and residual/sampling overhead remain
  approximations; the representative layer view excludes the separate head cost.

## Network and memory

The end-to-end group bandwidth is bounded by the fabric tier it crosses:
intra-node → same rail/TOR → scale-out NIC → DC spine/WAN. Peak links are explicit
inputs, not assertions about a vendor product. Ring phases, collective efficiency
and latency are added. The method does not simulate switch ports or packets.
Pipeline activation transfers use the fabric tier of the entire pool as a
conservative placement approximation. Prefill/decode KV handoff uses a serialized
spine links, their payload bytes, per-link bandwidth and fixed latency. Parallel
handoff channels are independent links assigned round-robin to prefill groups;
the configured spine bandwidth applies to each link. This is an aggregate link
model, not a physical switch placement model.

The SRAM budget is reserved exclusively for ideal reusable KV read data. KV reads
beyond the budget hit HBM. KV append writes still hit HBM; the full KV payload
remains counted in HBM capacity. This ideal-cache assumption does not model cache
line replacement, bandwidth, fill latency or competition with weights.

Per-GPU capacity includes sharded weights, conservative max-context KV payload for every request assigned to a replica, including queued batches
and approximate activation workspace. EP shards experts and replicates shared
parameters/KV across EP ranks. KV heads replicate when TP exceeds KV-head count.
Usable capacity is physical capacity × memory allocation fraction − runtime reservation
in GB. These server controls affect capacity rejection and headroom. The activation
workspace above is still included. Extra allocator fragmentation and unreserved
framework overhead are excluded. A capacity violation is rejected instead of producing optimistic
throughput from an impossible fit.

## Serving and metrics

SimPy models FIFO resource contention for fixed microbatches, per-PP-stage
resources, replicas, pipeline messages and KV-handoff links. Each pool has an
independent batch cap and active-batch admission limit per replica. Admission
slots are held across all prefill chunks or decode iterations, then released.
Prefill groups split into smaller decode groups when the decode cap is lower;
groups from different prefill batches are never merged. Larger decode caps do not
combine requests across prefill groups. Both phases route groups round-robin to
replicas. Every interval retains its request IDs, and request rows retain both
phase batch IDs. Batches form
when their last request arrives at the configured deterministic arrival rate.
Separate prefill chunks traverse the prefill stages; each decode iteration
traverses the decode stages. PP stages can overlap across batches. Colocation
shares prefill/decode resources and removes KV handoff. This is not the native
vLLM continuous-batching scheduler. The native tab executes that scheduler.

Prefill emits the first output token; remaining output tokens cause decode work.
TTFT includes batch formation, queueing, prefill and KV handoff. This demo releases
the first token after handoff, deliberately making transfer a TTFT floor. TPOT is
the mean interval after that first token; a one-token output has no interval and
reports zero. Throughput is completed output tokens divided by the full simulated
horizon. Resource occupancy includes compute/memory/network stalls inside each
reserved stage and is not tensor-core utilization. Resource-work bars overlap
and must not be summed to obtain end-to-end latency.

Energy defaults to idle power over the horizon plus 85% of the TDP-minus-idle
increment during modeled stage occupancy. Optional nonzero pJ coefficients switch
to component-work dynamic energy, capped at the same device TDP budget. In that
mode a zero coefficient excludes that term; supply all relevant coefficients.
Network equipment, host CPUs and cooling are excluded. Cost is supplied hourly
GPU rental rate times allocated GPUs times horizon; it is a scenario assumption,
not a live cloud price. Tokens/J and cost/million tokens count output tokens.

## Critical hardware data

Peak TFLOPS, memory capacity/bandwidth and link bandwidth make a useful exploratory
model, but do not establish accuracy. For decision-grade calibration, collect:

1. Operation latency grids by batch, token count/context, precision, TP and engine.
2. Memory capacity and actual working-set/traffic measurements.
3. Collectives by participant count, message size and topology, including latency.
4. KV/cache behavior and scheduling/engine configuration.
5. Idle and active GPU power; component energy when available.

The native LLMServingSim path consumes its measured operation grids and exact
configuration. Editing memory/link inputs preserves those measured kernel costs.
New compute hardware needs new profiles; the UI never presents a peak-spec
extrapolation as a measured KAIST result. No accuracy percentage is claimed here.

## Reproducibility

All results include resolved inputs, configuration SHA-256 and source revisions.
Behavioral results now use `servescope.behavioral.v3`, including timing provenance and calibration fingerprints. Legacy scenarios with a
single top-level `batch_size` and missing server controls are still accepted;
that batch size initializes both pool caps. Resolved per-pool caps take precedence,
and the top-level field remains an alias for the prefill cap.
The native adapter additionally records three source pins, clean tracked-source
checks, binary/profile/runner hashes, exact command/config/workload, raw CSV/logs,
actual Chakra DAGs, and wall/RSS watchdog measurements. Its build adaptation is
isolated and documented in `integration/README.md`.

## Optional cache-aware storage mode

The retained original capacity/fixed-workload descriptions above apply when KV
tiers are disabled. Tier mode reserves active KV only when memory admission
grants it, uses actual per-request lengths, restores ready contiguous prefixes,
and executes remaining query tokens with full attention context. Inclusive
retained copies use explicit per-tier capacities. Publication can be asynchronous
or blocking before KV handoff. Active decode KV defaults to HBM and has a separate
placement control.
Published Mooncake request traces provide arrivals, token lengths and prefix
identities. They contain no device-service timing measurements. SSD, host–GPU
and GDS timing are configured assumptions. Trace-policy prefix-reuse fits are
separate from device timing and never assign runtime cache hits. Read the
[trace coverage and storage assumptions](storage-model.md).

Cache placement now follows the selected [single-tier, exclusive-spill or inclusive-copy policy](configuration.md#cache-layouts-and-eviction-spill). Spill transfer costs and readiness are scheduled explicitly. Active decode placement is independent; hit-fit residuals remain single-LRU replay validation, with no new multi-tier or streamed-attention calibration claim.

Fixed uniform-length prefix-cache microbatches preserve the configured prefill/decode caps and use a common restored-prefix floor. Group HBM admission is atomic. [Controlled baseline comparisons](prefix-layout-comparison.md) hold active decode KV in HBM. The [active-KV model](active-kv-model.md) separately documents serialized DRAM/SSD backing, staging, admission and queue assumptions. Its end-to-end accuracy is unvalidated.
