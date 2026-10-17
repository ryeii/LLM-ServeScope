# Model and server configuration

These options belong to the behavioral engine. Peak specifications are inputs to
our equations; they are not measured LLMServingSim operation profiles. Native
simulation retains its own supported model/hardware selectors.

## Accelerator presets

| Preset       | Memory GB | Memory TB/s | Dense BF16 TFLOPS | Primary reference                                                                                                                                                                                                                         |
| ------------ | --------: | ----------: | ----------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A100 SXM     |        80 |       2.039 |               312 | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/)                                                                                                                                                                             |
| H100 SXM     |        80 |        3.35 |               989 | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)                                                                                                                                                                             |
| H200 SXM     |       141 |         4.8 |             989.5 | [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/)                                                                                                                                                                             |
| B200 SXM     |       180 |         7.7 |              2250 | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/), [Lenovo B200 product guide](https://lenovopress.lenovo.com/lp2226.pdf)                                                                                             |
| L40S         |        48 |       0.864 |            362.05 | [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/)                                                                                                                                                                             |
| RTX 4090     |        24 |       1.008 |             165.2 | [NVIDIA Ada architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)                                                                                                             |
| RTX PRO 6000 |       102 |      1.5976 |  503 (assumption) | [Pinned upstream hardware config](https://github.com/casys-kaist/LLMServingSim/blob/b09c7aaab4b072d73136641841cc2688616c49eb/profiler/perf/RTXPRO6000/hardware.yaml)                                                                      |
| AMD MI300X   |       192 |         5.3 |            1307.4 | [AMD MI300 architecture](https://rocmdocs.amd.com/en/docs-7.2.4/conceptual/gpu-arch/mi300.html), [AMD specifications](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) |

All memory and bandwidth values are decimal. Tensor peaks use dense values;
when a vendor lists sparse throughput, it is halved. The B200 preset follows the
180 GB/1000 W Lenovo HGX specification (7.7 TB/s, FP32 75 TFLOPS), rather than
mixing it with the rounded 8 TB/s DGX figure. RTX 4090 floating-point tensor
peaks use FP32 accumulation; INT8 uses its separate integer rate. INT8 uses the equivalent
operation-rate convention. The RTX PRO 6000 preset retains the original demo's
rounded upstream capacity/BW and explicitly illustrative compute peaks. It is
not the native measured profile.

Fabric links are per-device, one-direction bandwidth inputs. Bidirectional NVLink
marketing bandwidths are divided by two. The PCIe presets use a conservative
31.5 GB/s intra-node link, a one-device fast domain, and the editable hierarchical
fabric abstraction for larger groups. This is not a PCIe switch or actual topology
benchmark. New presets default the reserved KV SRAM budget to zero; physical
cache is not automatically assumed to be exclusively available for KV. SM/CU
counts feed one generic tiling abstraction, including AMD CUs; this is not a
cross-architecture occupancy model. B200's 148-unit tiling input is an explicit
scenario assumption. Cache allocation, tile width, topology, efficiencies, idle
power and hourly rental rates all require calibration for a real deployment.

Changing a hardware field marks the pool as edited while retaining its source
preset identity. Clock scale multiplies tensor/vector peaks (0.1–2); it does not
change memory bandwidth or power automatically. Extra launch overhead adds to
the selected engine policy's launch cost per operation. Unsupported device
precision peaks are zero and cause a validation error when selected.

## Model presets and custom geometry

Ten presets cover Llama 3.1 8B/70B; Qwen3 0.6B/4B/8B/14B/32B;
Qwen3 30B-A3B/235B-A22B; and Mixtral 8x7B. Qwen3 publisher snapshots and immutable
Hub revisions are retained in `model-sources.json`. The original three models
and Mixtral use the pinned KAIST configuration. Each resolved model includes its
own source URL/revision, separate from the simulator source revision. Qwen3 QK
normalization follows the [Transformers Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen3/modeling_qwen3.py).

Choose **Customize architecture** to start from the current preset's geometry,
or choose **Custom decoder architecture**. Edit the name, layer count, hidden and
FFN/expert width, attention heads, KV heads, head dimension, vocabulary, total
experts, active experts, maximum context, tied embeddings and QK normalization.
Attention width need not equal hidden width, as with some Qwen3 variants.
KV heads must divide attention heads; MHA uses equal counts, MQA uses one KV head,
and GQA uses an intermediate divisor. One expert/top-1 represents a dense FFN.
TP must divide attention heads, PP must divide layers, and EP must divide experts.
Selecting a different preset reduces incompatible parallelism to a valid divisor;
custom edits are validated when run. Combined input/output context must fit the
model's context limit and the interactive workload bounds.

The operation model covers uniform causal decoder layers with RMSNorm, RoPE,
SwiGLU, optional QK normalization, and uniformly routed experts. It counts tied
embedding/head storage once and adds router projections for MoE. It does not
represent MLA, hybrid/sliding attention, shared experts, encoder-decoder stacks,
vision encoders or SSM layers. An arbitrary Hugging Face model name alone is not
sufficient input for a supported architecture.

## Prefill and decode servers

Use the **Prefill pool / Decode pool** selector to edit each independently:

| Control                           | Effect                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Hardware preset and edited fields | Phase-specific compute, memory, communication, energy and cost                                                |
| TP, PP, EP, replicas              | Sharding, pipeline stages, collectives, parallel serving resources and GPU count; bounded at 64 per dimension |
| Server engine policy              | Override the workload policy's assumed efficiency/launch costs for this phase                                 |
| Prefill microbatch                | Fixed request grouping before prefill; bounded at 32                                                          |
| Decode batch cap                  | Split each completed prefill group into smaller decode groups; no merging across groups                       |
| Prefill chunk                     | Number of prompt tokens per prefill work chunk; bounded at 8192                                               |
| Active batches / replica          | FIFO admission limit, held across all chunks/tokens of the phase; 1–128                                       |
| Memory allocation limit           | Usable fraction of physical device memory; 0.1–1                                                              |
| Runtime reservation               | Additional GB subtracted after the allocation fraction                                                        |

Usable memory = physical GB × allocation fraction − runtime reservation GB.
Weights, resident KV and the modeled activation workspace must fit this amount.
The memory chart reports this usable capacity, and exports also retain physical
capacity and the two reservation controls. KV capacity is conservatively reserved
for assigned requests, including queued groups; admission limits do not magically
remove their KV storage.

Colocated mode resolves decode to the prefill pool and shares stage/admission
resources. The decode editor is disabled. Disaggregated mode preserves different
hardware, parallelism and policies. Replica assignment is round-robin. Decode
batch caps larger than prefill groups do not merge groups. This is fixed batching,
not a reproduction of native vLLM continuous batching.

KV cache precision (BF16, FP16, FP8) changes KV capacity, attention traffic and
handoff payload. Weight precision stays separate. Conversion/scale-storage costs
and numerical quality are not modeled. KV transfer channels are independent
round-robin links, each with the configured spine bandwidth/latency. Their
intervals occupy distinct lanes in the box graph; the graph/export never presents
concurrent transfers as one exclusively reserved resource.

## Replay and compatibility

Scenario/result JSON includes the resolved model geometry or preset ID, pool
policies, all hardware inputs and a SHA-256 fingerprint. Behavioral output uses
`servescope.behavioral.v2`; legacy scenarios without new controls remain valid
and receive documented defaults. A legacy top-level `batch_size` initializes
both pools; explicit per-pool values take precedence. On resolution, that legacy
field becomes an alias for the prefill batch cap.

`examples/heterogeneous-scenario.json` is a runnable Qwen3 MoE scenario with an
H200 prefill pool and A100 decode pool, independent batch caps and engine policies.
`examples/custom-decoder-scenario.json` demonstrates custom MHA geometry. These
are strategy examples, not benchmark claims.
