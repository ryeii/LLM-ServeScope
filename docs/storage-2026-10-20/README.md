# KV storage tiers: Mooncake evidence, fitted functions and validation

20 October 2026 · ryeii (Ryan Z An)

ServeScope now connects prefix cache decisions to real executed operator shapes,
active-HBM admission and explicit I/O/link queues. This is a behavioral simulation.
**Published Mooncake results, local measured I/O and trace-replayed cache outcomes
are three different evidence classes. They must not share an accuracy claim.**

## Published coverage: what is available and what it establishes

The source audit is pinned to `7e4fc11ee0fe9553a82a408009ea2b499c76842e`. The unchanged
FAST25 request dumps use the earlier pinned release `e4229791f39d1f4cdb00df1b28366130f742deb0`.
Source paths and SHA-256 hashes are in [benchmark manifest](../../third_party/mooncake-storage/manifest.json)
and [request-dump manifest](../../third_party/mooncake-traces/manifest.json).

| Evidence | Detailed coverage | Missing information / permitted use |
| --- | --- | --- |
| FAST25 request dumps | Conversation: 12,031 requests; tool/agent: 23,608; upstream synthetic: 3,993. Arrival milliseconds, input/output lengths, ordered 512-token prefix hashes. Conversation and tool/agent are sampled production workloads. | No text/tokens, completion times, measured cache hits, SSD I/O, device addresses, power or tool-dependency graph. Suitable workload inputs, not measured serving/storage targets. |
| Older storage-benchmark documentation | Trace-replayed file I/O; reported per-block/read/write/request metrics and an illustrative output; older command and model-size table. | No machine/durability manifest or raw latency sample grid in that example. Its 405B bytes/token label and displayed 1 MB block size disagree. Do not use it as a fit dataset. |
| Current Python `storage_benchmark_v1` | GLM-5 and Kimi-K2.6 MLA layouts; configurable token-page size; one preallocated file or per-file storage; buffered `pread`/`pwrite`; fsync none/batch/always/end; replay speed and client threads; request wall and summed I/O timing separated. | A benchmark generator, not a published hardware service curve. Threads have independent storage namespaces, each replaying the entire trace. Default writes are not durable. Modulo slots and the logical written-set do not implement finite-capacity LRU. |
| Published SSD-offload serving experiment | Four cache configurations, Qwen3-8B, repeated-turn prefill workload with output length 1. TTFT, input throughput and per-round cache behavior. | Aggregate serving evidence on one setup; no per-size/QD service samples, physical-power data or held-out devices. Cannot isolate an SSD service function or general decode performance. |
| Rust Store-RS benchmark | Put/get/mixed, batched and single-key APIs, concurrency, size, warmup, duration, verification and fault injection; JSON/CSV/text reporting. | Tool capability and illustrative outputs are not a measured SSD latency grid. API calls and objects have different counter units. |
| Transfer Engine `tebench` | A separate transfer benchmark for transport paths. | Transfer figures are not SSD service measurements and cannot be substituted for filesystem I/O calibration. |

### The published SSD result is useful, but narrow

Mooncake reports a DGX host with 8 A100-SXM4-40GB GPUs, dual HDR200 RDMA NICs and
five mixed-generation Samsung NVMe drives in RAID0. The documented serving command
uses TP1; the node inventory alone does not establish eight-GPU serving execution.
The comparison adds host caching, an 80 GB Mooncake memory pool, then SSD offload
with a 20 GiB buffer. Its multi-turn workload uses 20 clients, 10 rounds,
4096-token request increments and one output token, with a round barrier.
Reported SSD gains include 57% lower mean TTFT versus GPU-only, 34% versus Mooncake
without SSD, and 2.4x input throughput versus GPU-only. These are reported system
results, not newly reproduced ServeScope measurements. The roughly 27 GB/s RAID
figure is theoretical aggregate sequential-read capability, not a measured service curve.

[Published experiment](https://github.com/kvcache-ai/Mooncake/blob/7e4fc11ee0fe9553a82a408009ea2b499c76842e/docs/source/performance/mooncake/ssd-offload-benchmark-results.md)
· [Current benchmark source](https://github.com/kvcache-ai/Mooncake/blob/7e4fc11ee0fe9553a82a408009ea2b499c76842e/benchmarks/storage_benchmark_v1/benchmark.py)
· [Current layout equations](https://github.com/kvcache-ai/Mooncake/blob/7e4fc11ee0fe9553a82a408009ea2b499c76842e/benchmarks/storage_benchmark_v1/layout/mla.py)
· [Disk-path implementation](https://github.com/kvcache-ai/Mooncake/blob/7e4fc11ee0fe9553a82a408009ea2b499c76842e/benchmarks/storage_benchmark_v1/storage/disk.py).

### Source issues relevant to calibration

- The older documentation refers to `storage_benchmark.py` and a broad preset table;
  the audited implementation is `benchmarks/storage_benchmark_v1/benchmark.py`,
  whose shipped layouts cover GLM-5 and Kimi-K2.6. They are not interchangeable versions.
- GQA KV bytes are derived by ServeScope from **2 × layers × KV heads × head dimension × dtype bytes**,
  then sharding/replication. We do not import the old documentation's ambiguous bytes/token labels.
- Current MLA layout includes layer count. GLM-5 at 512 tokens is 56,229,888 bytes;
  Kimi-K2.6 is 35,979,264 bytes. These are storage layouts, not evidence that ServeScope now models MLA compute.
- Buffered I/O includes filesystem/page-cache/client effects. `fsync` is the OS call,
  not verification of power-loss persistence. Per-file `batch/end` durability differs
  from persistent-single-file mode as documented upstream.
- `_written_pages` remembers logical IDs even after physical modulo slots collide;
  it is not a valid finite-capacity hit/eviction ground truth. No customer SSD curve
  or production hit telemetry was found in the reviewed benchmark source/docs artifacts.

## New local I/O measurements

We exercised the **archived, unmodified Mooncake DiskHashTable class** using a
ServeScope harness. This does not claim to run the complete Mooncake serving stack
or reproduce its published SSD experiment.

- 768 measured operations: **384 reads and 384 writes**.
- Sizes: 16, 32, 64, 128, 256, 512, 1024 and 2048 KiB.
- Four shuffled trials per size; 12 measured read/write pairs per trial;
  two warmup pairs excluded. One client, serial QD1, two file slots.
- Single-file buffered access; writes use fsync=always; reads immediately follow
  writes and are warm. No O_DIRECT or cold-cache control.
- Host OS/runtime metadata, exact per-operation timing and source hash are in
  [raw I/O observations](io-observations.json). No hostname or private device path is collected.
- The harness uses bounded temporary files on this Mac and removes them afterwards.
  It does not identify/calibrate a customer SSD, RAID, GPU DMA path or remote store.

Warm buffered read bandwidth can exceed physical SSD bandwidth. This is a measured
**local file path** profile, and the UI names it that way. The default configurable
SSD profile remains an explicit uncalibrated scenario assumption.

## Fitted latency and bandwidth functions

Let `x = I/O bytes / 10^6`, and `t` be milliseconds. All coefficients are nonnegative.

| Candidate | Equation | Parameters | Shape rationale |
| --- | --- | ---: | --- |
| Affine | `t = L + b x` | 2 | Fixed path overhead plus byte-proportional service |
| Sqrt-affine | `t = L + b x + a sqrt(x)` | 3 | A small monotone curvature candidate |
| Two-slope | `t = L + b x + a max(0,x−0.25)` | 3 | A continuous size-regime transition; 0.25 MB is fixed, not fitted to held-out targets |

**Bandwidth is constrained to `B_MBps = bytes / (1000 × t_ms)`.** We fit one
consistent timing function, rather than independent latency and bandwidth functions
that could contradict each other. Both errors are evaluated; they are two views of
the same observations, not independent evidence. NNLS uses relative-latency weights.

Each of eight outer folds withholds **every trial/repetition of one I/O size**.
Family selection inside that training set uses inner leave-size-out evaluation.
Choose the fewest parameters within one percentage point of best joint mean
latency/bandwidth MAPE. The two endpoint-size folds are extrapolation tests.
The final selected family is refitted on all data for runtime. No held-out graph
uses the final full-data coefficients. Every outer fold archives its fitted coefficients
and training/excluded groups in `validation.json`; regression tests reconstruct
every held-out prediction from those archived fold models.

### Selected-function nested held-out error

| Path | Final family | Latency MAPE | Latency P95 error | Bandwidth MAPE | Bandwidth P95 error |
| --- | --- | ---: | ---: | ---: | ---: |
| read | two-slope | 16.37% | 53.09% | 29.42% | 113.16% |
| write | affine | 9.97% | 27.90% | 12.04% | 38.70% |

![Local file-path latency and consistent bandwidth: measurements versus nested held-out predictions](https://ryeii.github.io/LLM-ServeScope/storage-study/io-fit.png?v=17c686000e4f)

![Every candidate I/O function and fixed-family fit](https://ryeii.github.io/LLM-ServeScope/storage-study/io-candidates.png?v=999682c7e5a0)

### Candidate iteration and full-data coefficients

**Read candidates**, fixed-family leave-size-out scores (distinct from nested selection scores above):

| Family | Latency MAPE | Bandwidth MAPE | Joint MAPE |
| --- | ---: | ---: | ---: |
| affine | 16.47% | 29.25% | 22.86% |
| sqrt-affine | 16.47% | 29.25% | 22.86% |
| two-slope | 15.57% | 27.45% | 21.51% |

Final full-data coefficients `[L,b,a]`: `0.001475888815, 0.0300104461, 0.007012321548`.

**Write candidates**, fixed-family leave-size-out scores (distinct from nested selection scores above):

| Family | Latency MAPE | Bandwidth MAPE | Joint MAPE |
| --- | ---: | ---: | ---: |
| affine | 9.66% | 11.39% | 10.53% |
| sqrt-affine | 9.66% | 11.39% | 10.53% |
| two-slope | 9.64% | 11.41% | 10.53% |

Final full-data coefficients `[L,b]`: `0.02081865826, 0.3288717914`.

The read tail is noisy: reciprocal bandwidth error is particularly large for very
fast warm-cache operations. A deterministic size function cannot explain every
microsecond-scale scheduling/cache fluctuation. We retain and report the tail;
no outliers are removed. These errors do not establish cold-SSD, mixed-QD,
cross-machine or production serving accuracy.

## Fitted hit-rate functions and their different target

The hit target is an independently generated **policy-replay reference**, not
measured production telemetry. All three unchanged traces are divided into six non-overlapping temporal regions. Each region contributes cold-start windows
of 8, 32, 128, 512 and full-region requests; all requests count. We score
longest-contiguous complete 512-token prefix hits under fully associative LRU and
instantaneous publication. Fourteen capacities (0 through 65,536 blocks) give
**1,260 curve observations across 90 windows in 18 temporal regions**. Nested
windows overlap, so they always share their parent held-out region. These are
correlated observations, not 1,260 independent workloads.

Normalize `x = cache_capacity_blocks / unique_blocks_in_window`. From the known
trace, calculate the compulsory-miss upper bound `H_max = 1 − unique_blocks / total_block_references`.
This uses workload features, not measured capacity-specific hits. Candidates are:

- Capped linear: `H = H_max × a × min(1,x/s)` (2 parameters).
- Exponential: `H = H_max × a × [1−exp(−x/s)]` (2 parameters).
- Weibull: `H = H_max × a × [1−exp(−(x/s)^p)]` (3 parameters).

`a` is bounded to [0,1], `s>0`, `p>0`: curves are bounded and monotone.
Fit each workload separately. Six outer folds withhold complete temporal regions, including all nested window
lengths/capacities; inner leave-region-out selection uses only outer training regions. Choose the simplest
family within 0.5 percentage points of best MAE. Report percentage-point errors,
not unstable percentage errors at zero hit rate. Deployment refits all six regions. The trace-derived H_max is available for offline
what-if replay, not a promise of foreknowledge for unknown online requests.
This is not unseen-workload validation or a causal model of tool calls.

| Trace | Selected family | Nested MAE | RMSE | P95 absolute error |
| --- | --- | ---: | ---: | ---: |
| conversation | exponential | 1.27 pp | 1.87 pp | 3.78 pp |
| toolagent | weibull | 3.79 pp | 6.97 pp | 13.75 pp |
| synthetic | exponential | 0.53 pp | 1.50 pp | 3.29 pp |

![LRU trace-replay hit references versus nested held-out fitted hit curves](https://ryeii.github.io/LLM-ServeScope/storage-study/hit-fit.png?v=3cc0438638a4)

![Every candidate hit-rate function and resulting fit](https://ryeii.github.io/LLM-ServeScope/storage-study/hit-candidates.png?v=6b4f3f541d13)

**conversation candidate search**:

| Family | Fixed-family held-region MAE | Parameters |
| --- | ---: | ---: |
| capped-linear | 1.45 pp | 2 |
| exponential | 1.27 pp | 2 |
| weibull | 1.16 pp | 3 |

Final `[a,s]`: `0.986960634, 0.2204393151`.

**toolagent candidate search**:

| Family | Fixed-family held-region MAE | Parameters |
| --- | ---: | ---: |
| capped-linear | 6.24 pp | 2 |
| exponential | 5.75 pp | 2 |
| weibull | 3.79 pp | 3 |

Final `[a,s,p]`: `1, 0.02517243608, 0.2890166428`.

**synthetic candidate search**:

| Family | Fixed-family held-region MAE | Parameters |
| --- | ---: | ---: |
| capped-linear | 0.55 pp | 2 |
| exponential | 0.53 pp | 2 |
| weibull | 0.49 pp | 3 |

Final `[a,s]`: `1, 0.3328386115`.

The compulsory-miss bound captures much of the variation in short windows. The
tool-agent tail remains weak, and candidate tables retain that error. Windows of
8–128 requests overlap the fitted size range; shorter slices are flagged. These
curves are conditional on known workload reuse and their policy assumptions, not
unseen-workload, unknown-future-request or hardware-hit predictions. They do not
assign runtime hits: actual I/O queues and pending fills can make observed
simulated reuse differ from the instantaneous-publication reference.

## Runtime tier model and connection to computation

1. Each request retains its actual arrival, input/output lengths and known prefix
   hashes. Timestamps are shifted by the first selected arrival; lengths are never
   clipped. Source hashes and selected window metadata are exported.
2. Reserve active per-GPU KV bytes before restore/prefill. If HBM is occupied,
   the request waits in the SimPy memory container; queued requests do not
   preallocate all their KV. A request too large for its budget is rejected.
3. Look up the longest contiguous ready prefix, HBM → DRAM → SSD. Pending writes
   do not create hits. Pin selected sources so queued restores cannot read evicted data.
4. Restore lower-tier blocks through explicit shared SSD and host/GPU resources.
   Reads/writes share one serial SSD path. The local QD1 fit covers service after
   resource grant; outer queue delay is charged once.
5. Compute only uncached query tokens, using **full context length** in attention.
   Always recompute at least the final prompt token for logits. Do not multiply
   full-prefill time by a cache-hit fraction. Actual shapes and coefficients are exported.
6. Publish complete known prompt blocks as retained copies. The initial policy is
   **blocking write-through before KV handoff**, rather than an unmodeled promise
   of free asynchronous writes. Concurrent duplicate publication waits on one fill.
7. Transfer the complete prompt KV to decode when disaggregated. Release prefill
   active KV, reserve decode active KV, then release the first token. Decode keeps
   active KV in HBM through its remaining token iterations.

HBM retained-cache capacity is per prefill GPU; aggregate DRAM/SSD payload includes
prefill TP/PP/EP shard replication. Inclusive tiers can hold the same block, but
one block consumes space only once within each retained tier. Active-request KV
and retained cache are deliberately separate copies; we do not claim cross-request
physical page aliasing. Cache reservation reduces available active HBM.
The memory chart conservatively combines peak active and peak retained copies.

SSD objects are serviced in configurable 16–2048 KiB chunks, using the measured
function only in its documented size domain; remainder extrapolations are flagged.
This chunking is a behavioral path assumption, not reproduction of Mooncake’s
56 MB MLA storage operations. Host/GPU and HBM copy costs remain configured bandwidth
assumptions. Cache hit curves do not decide runtime cache hits.

### Scope and exclusions

- One replica and batch size one per pool; concurrent requests and TP/PP/EP remain
  supported. No continuous batching, cache-aware routing or multi-replica cache placement.
- GQA/MHA/MQA KV geometry only. MLA layout source is audited, not imported as compute support.
- No active attention-KV streaming from SSD, weight offload, remote object-store tier,
  NAND/FTL simulation, filesystem simulator, cold-device service or automatic durability inference.
- GPU energy includes the changed timeline but excludes SSD/CPU/NIC power.
- The tier model does not reproduce the published four-tier RDMA Mooncake experiment.
  No end-to-end customer accuracy is claimed without separate serving measurements.

## UI, artifacts and reproduction

Open **KV storage**, choose **Load cache demo**, then **Run scenario**. The demo is
explicitly synthetic and deliberately sparse so fills can finish before reuse.
Its finite-run throughput may be arrival-limited; compare per-request TTFT as well. The sidebar’s **KV storage tiers** controls can load an
unchanged Mooncake trace and select a contiguous request window. Loading a trace
sets batch/replica counts to one. Invalid contexts or excessive work are rejected,
not silently truncated. Storage rows, timeline I/O and throughput have cascading
formula explanations. The browser uses the same Python model through Pyodide.

Artifacts: [I/O measurements](io-observations.json), [policy-replay observations](hit-observations.json),
[held-out predictions](held-out-predictions.json), [candidate scores and outer selections](validation.json),
[runtime coefficients](../../backend/storage-fit.json), [measurement/fitting harness](../../integration/storage/study.py),
[KV tier model](../../backend/storage.py), [serving integration](../../backend/model.py).

```bash
# Offline study dependencies are in the existing .venv-sim environment.
# Measurement performs real bounded temporary file I/O on the current host.
.venv-sim/bin/python -B -m integration.storage.study measure
# Optional: measure a chosen filesystem with measure --storage-dir /path/to/mount
.venv-sim/bin/python -B -m integration.storage.study replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-sim/bin/python -B -m integration.storage.study train
.venv-sim/bin/python -B -m integration.storage.study figures
.venv/bin/python -B -m unittest tests.test_storage -v
```

Re-measuring changes the host-specific data and must be followed by refitting and
updating this report with `python -m integration.storage.report`. Archived sources are by Mooncake contributors under Apache
2.0; retained licenses and manifests accompany them. Local measurements, fitting,
validation, tier modeling and ServeScope integration: **[ryeii (Ryan Z An)](https://github.com/ryeii)**.
