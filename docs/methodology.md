# Behavioral methodology and calibration

The aim is strategic sensitivity analysis, analogous to a semiconductor behavioral
model. This is not a page-level memory simulator, a cycle-accurate GPU emulator,
or a validated predictor of a new chip. Every run separates architecture/workload
facts from equations, scenario assumptions and native profile-derived results.

## Operation model

Model geometry is taken from the pinned KAIST model configuration: hidden and
intermediate widths, layers, attention/KV heads, head dimension, vocabulary, and
expert count/top-k. Dense and MoE linear work uses matrix multiplication FLOP
counts; prefill attention includes the causal triangular sequence length.
BF16 KV occupies two bytes per element independently of weight precision.

For each operation and GPU:

- Compute time = FLOPs / (selected dense peak × assumed efficiency × tile utilization).
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
spine resource, its payload bytes, bandwidth and fixed latency.

The SRAM budget is reserved exclusively for ideal reusable KV read data. KV reads
beyond the budget hit HBM. KV append writes still hit HBM; the full KV payload
remains counted in HBM capacity. This ideal-cache assumption does not model cache
line replacement, bandwidth, fill latency or competition with weights.

Per-GPU capacity includes sharded weights, conservative max-context KV payload for every request assigned to a replica, including queued batches
and approximate activation workspace. EP shards experts and replicates shared
parameters/KV across EP ranks. KV heads replicate when TP exceeds KV-head count.
Temporary allocator fragmentation, runtime reservations and framework workspaces
are excluded. A capacity violation is rejected instead of producing optimistic
throughput from an impossible fit.

## Serving and metrics

SimPy models FIFO resource contention for fixed microbatches, per-PP-stage
resources, replicas, pipeline messages and the KV-handoff resource. Batches form
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

All results include resolved inputs, configuration SHA-256 and source revision.
The native adapter additionally records three source pins, clean tracked-source
checks, binary/profile/runner hashes, exact command/config/workload, raw CSV/logs,
actual Chakra DAGs, and wall/RSS watchdog measurements. Its build adaptation is
isolated and documented in `integration/README.md`.
