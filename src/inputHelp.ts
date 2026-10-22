type Obj = Record<string, any>;
export type InputHelpContext = { config: Obj; catalog: Obj; phase: string };
export type InputExplanation = { title: string; meaning: string };

const entries: Record<string, string> = {
  "Model architecture":
    "The decoder architecture whose layers, attention heads, vocabulary and expert dimensions define the work.",
  "Weight precision":
    "The numeric format used for weights and modeled operators.",
  "Engine efficiency policy":
    "The workload-wide engine policy, inherited by pools unless they override it.",
  "Operator timing model":
    "How GPU operation service times are estimated before SimPy schedules them.",
  "Model name": "A descriptive label for the custom decoder.",
  "Transformer layers": "The number of repeated decoder blocks.",
  "Hidden width": "The width of each token's hidden-state vector.",
  "FFN / expert width":
    "The intermediate width of the feed-forward network or each routed expert.",
  "Attention heads": "The number of query attention heads.",
  "KV heads":
    "The number of distinct key/value heads, which can be fewer than query heads.",
  "Head dimension": "The vector width of one attention head.",
  "Vocabulary size":
    "The number of token entries in the embedding/output vocabulary.",
  "Total experts":
    "The number of feed-forward experts in each modeled decoder block; one represents a dense FFN.",
  "Active experts / token": "How many experts each token is routed to.",
  "Model context limit":
    "The maximum combined input and generated context allowed for this architecture.",
  "Tied embeddings":
    "Whether input embeddings and the output head share one weight matrix.",
  "QK normalization":
    "Whether queries and keys have an additional normalization operation.",
  "KV cache precision":
    "The numeric format of resident attention keys and values, configured separately from weights.",
  "Input length":
    "The full prompt length of every request in this uniform workload.",
  "Output length":
    "The fixed total output-token count for each request, including the first token produced during prefill.",
  Requests: "The number of requests that must complete in this finite run.",
  "Arrival rate":
    "A deterministic request arrival rate in requests per second.",
  "Prefix reuse assumption":
    "An optional fraction of prompt tokens whose KV is assumed already ready in HBM for every scenario request.",
  "Separate prefill and decode":
    "Whether prefill and decode use separate GPU pools.",
  "Pool editor":
    "Chooses which pool's hardware and server-policy values you are editing.",
  "Hardware preset":
    "A starting set of per-device compute, memory, topology, power and cost inputs for this pool.",
  TP: "Tensor parallelism: the number of GPUs sharing a layer's tensors within this pool.",
  PP: "Pipeline parallelism: the number of sequential GPU stages sharing the model's layers.",
  EP: "Expert parallelism: the number of GPU groups sharing routed experts.",
  Replicas: "The number of independently scheduled copies of this pool.",
  "Server engine policy": "An engine-efficiency override for this pool.",
  "Prefill microbatch":
    "The maximum number of consecutive requests grouped for prefill.",
  "Decode batch cap":
    "The largest decode subgroup formed from one completed prefill group.",
  "Prefill chunk":
    "The maximum number of new prompt tokens processed per sequence in one prefill work item.",
  "Active batches / replica":
    "The number of batches allowed into a phase's admission pool on one replica.",
  "Memory allocation limit":
    "The usable fraction of each GPU's physical memory before the runtime reservation.",
  "Runtime reservation":
    "Additional per-GPU memory withheld for runtime overhead, in decimal GB.",
  "Spine bandwidth":
    "The configured bandwidth of each prefill-to-decode KV handoff channel and the outermost fabric bottleneck.",
  "Spine latency":
    "The configured startup delay of the outermost fabric/KV handoff, in microseconds.",
  "KV transfer channels":
    "The number of independent parallel handoff links between separate prefill and decode pools.",
  "Main memory":
    "Physical memory capacity per GPU, in decimal GB; this is GPU memory, not host RAM.",
  "Memory bandwidth":
    "The per-GPU memory bandwidth used by the selected pool, in decimal TB/s.",
  "Physical L2 cache": "The physical GPU L2 capacity, in binary MiB.",
  "KV SRAM budget":
    "An idealized per-GPU SRAM allocation exclusively available for KV reads in the analytical diagnostic.",
  "Compute efficiency":
    "A dimensionless adjustment to effective compute rate relative to the nominal hardware peak.",
  "Memory efficiency":
    "A dimensionless adjustment to effective GPU memory bandwidth.",
  "Compute clock scale":
    "A multiplier on tensor/vector compute peaks for the selected pool; 2× represents twice the modeled compute rate.",
  "Extra launch overhead":
    "Additional startup time per modeled GPU operation, in microseconds.",
  "SM / CU count":
    "The number of GPU multiprocessor or compute units used by the shape-informed utilization features.",
  "Tensor tile width":
    "The assumed output tile width used by the analytical tensor-utilization heuristic.",
  "Intra-node bandwidth":
    "The per-device one-direction bandwidth for the smallest GPU fabric domain.",
  "GPUs / chip domain":
    "The maximum group size served by the smallest fast fabric domain.",
  "Same-rail bandwidth":
    "The per-device one-direction bandwidth of the same-rail/TOR tier.",
  "GPUs / same rail": "The configured number of GPUs in a same-rail domain.",
  "Scale-out NIC":
    "The per-device one-direction bandwidth of the scale-out fabric tier.",
  "GPUs / rack":
    "The configured GPU count per rack in the simplified fabric hierarchy.",
  "Cluster racks": "The number of racks available in the configured cluster.",
  "Base fabric latency":
    "The base communication startup latency in microseconds.",
  "All-reduce efficiency":
    "The usable fraction of modeled link bandwidth for tensor-parallel all-reduce.",
  "All-to-all efficiency":
    "The usable fraction of modeled link bandwidth for expert dispatch/combine.",
  "All-to-all offset":
    "An extra microsecond startup offset on modeled all-to-all communication.",
  "Device TDP": "The configured GPU power cap in watts.",
  "Idle power":
    "The assumed watts consumed by an allocated GPU when it is not doing modeled active work.",
  "GPU hourly cost":
    "The assumed rental cost of one GPU per hour, in US dollars.",
  "Tensor energy":
    "The assumed energy per tensor floating-point operation, in picojoules.",
  "Vector energy": "The assumed energy per vector operation, in picojoules.",
  "HBM energy":
    "The assumed energy per byte of modeled GPU memory traffic, in picojoules.",
  "Network energy":
    "The assumed GPU-side energy per modeled communication byte, in picojoules.",
  "Selected model geometry":
    "A read-only summary of the currently selected decoder's layer and attention geometry.",
  "Selected hardware specifications":
    "A read-only summary of this pool's current per-GPU memory capacity and bandwidth.",
  "Usable GPU memory":
    "The memory available per GPU after allocation and runtime reservations.",
};

const native: Record<string, string> = {
  "Native hardware":
    "The published hardware/profile bundle used by the native LLMServingSim reference.",
  "Native requests":
    "The number of requests supplied to the native reference experiment.",
  "Native input tokens":
    "The uniform prompt token count in the native workload.",
  "Native output tokens":
    "The requested fixed output length in the native workload.",
  "Native tensor parallel":
    "The tensor-parallel degree supported by the selected native profile bundle.",
  "Native memory capacity":
    "The device-memory capacity supplied to the native reference, in binary GiB.",
  "Native memory bandwidth":
    "The memory-bandwidth configuration supplied to the native reference.",
  "Native link bandwidth": "The native interconnect bandwidth in binary GiB/s.",
  "Native link latency":
    "The native communication startup delay in nanoseconds.",
};

const globalLabels = new Set([
  "Model architecture",
  "Weight precision",
  "Engine efficiency policy",
  "Operator timing model",
  "Model name",
  "Transformer layers",
  "Hidden width",
  "FFN / expert width",
  "Attention heads",
  "KV heads",
  "Head dimension",
  "Vocabulary size",
  "Total experts",
  "Active experts / token",
  "Model context limit",
  "Tied embeddings",
  "QK normalization",
  "KV cache precision",
  "Input length",
  "Output length",
  "Requests",
  "Arrival rate",
  "Prefix reuse assumption",
  "Separate prefill and decode",
  "Pool editor",
  "Spine bandwidth",
  "Spine latency",
  "KV transfer channels",
  "Selected model geometry",
]);

export function inputExplanation(
  label: string,
  context: InputHelpContext,
): InputExplanation {
  let meaning = entries[label];
  if (
    /^(BF16|FP16|FP8|FP4|TF32|INT8) tensor$/.test(label) ||
    label === "FP32 vector"
  )
    meaning = `Per-GPU dense ${label} peak rate, in trillions of operations per second.`;
  if (label.startsWith("Native "))
    return {
      title: label,
      meaning: native[label] || "A configurable native-reference input.",
    };
  return {
    title: globalLabels.has(label)
      ? label
      : `${context.phase === "prefill" ? "Prefill" : "Decode"} · ${label}`,
    meaning: meaning || "A configurable scenario input.",
  };
}
