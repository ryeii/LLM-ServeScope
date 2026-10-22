type Obj = Record<string, any>;
type Entry = [meaning: string, effect: string, evidence?: string];
export type InputHelpContext = { config: Obj; catalog: Obj; phase: string };
export type InputExplanation = {
  title: string;
  meaning: string;
  effect: string;
  evidence: string;
};

const assumption =
  "Configured behavioral assumption; no separate measured accuracy guarantee.";
const geometry =
  "Model geometry comes from the selected preset or custom inputs. Measured GPU timing coverage is narrower than the model catalog.";
const network =
  "Configured analytical network model; GPU operator validation does not validate link timings or serving queues.";
const power =
  "Configured GPU power/cost accounting; KAIST/Vidur operator profiles do not calibrate measured energy or rental cost.";
const entries: Record<string, Entry> = {
  "Model architecture": [
    "The decoder architecture whose layers, attention heads, vocabulary and expert dimensions define the work.",
    "Changing models changes operation shapes, parameter memory and KV size. Incompatible TP/PP/EP values are reduced to valid divisors. A preset is not proof of measured timing coverage.",
    geometry,
  ],
  "Weight precision": [
    "The numeric format used for weights and modeled operators.",
    "Changes weight bytes and the matching compute peak. FP16 selects the historical Vidur fit cohort; BF16 selects KAIST. Other precisions extrapolate; numerical accuracy and conversion overhead are excluded.",
    "KAIST BF16 and Microsoft Vidur FP16 profiles, fitted separately. Other formats are unvalidated extrapolations.",
  ],
  "Engine efficiency policy": [
    "The workload-wide engine policy, inherited by pools unless they override it.",
    "Applies assumed efficiency and launch-cost adjustments. This does not execute vLLM or TensorRT-LLM, or reproduce their schedulers.",
    assumption,
  ],
  "Operator timing model": [
    "How GPU operation service times are estimated before SimPy schedules them.",
    "Fitted mode evaluates shape-informed functions trained on KAIST/Vidur profiles. Roofline mode uses analytical compute/memory ceilings. Operator errors are not end-to-end serving error bounds.",
    "Grouped nested cross-validation on KAIST BF16 and Vidur FP16 profiles; final functions refit on all compatible observations.",
  ],
  "Model name": [
    "A descriptive label for the custom decoder.",
    "Changes names in the interface and exports. The name alone does not change architecture or fetch a model.",
    "User-supplied label; no model weights are loaded.",
  ],
  "Transformer layers": [
    "The number of repeated decoder blocks.",
    "Scales repeated operator work, parameter memory and KV capacity. Pipeline parallelism must divide this count.",
    geometry,
  ],
  "Hidden width": [
    "The width of each token's hidden-state vector.",
    "Changes projection shapes, normalization work, activations and parameter memory. Attention width is separately determined by heads × head dimension.",
    geometry,
  ],
  "FFN / expert width": [
    "The intermediate width of the feed-forward network or each routed expert.",
    "Sets gate/up and down projection shapes, activation work and expert parameter memory.",
    geometry,
  ],
  "Attention heads": [
    "The number of query attention heads.",
    "Together with head dimension, sets query width and attention arithmetic. TP must divide the head count; KV heads must divide query heads.",
    geometry,
  ],
  "KV heads": [
    "The number of distinct key/value heads, which can be fewer than query heads.",
    "Sets KV capacity, attention traffic and handoff payload. Equal query/KV counts mean MHA; one KV head means MQA; intermediate divisors mean GQA.",
    geometry,
  ],
  "Head dimension": [
    "The vector width of one attention head.",
    "Changes attention arithmetic, Q/K/V projection shapes and KV bytes per token.",
    geometry,
  ],
  "Vocabulary size": [
    "The number of token entries in the embedding/output vocabulary.",
    "Changes embedding and output-head geometry, parameter memory and sampling work. Padding in profiling-reference presets is significant.",
    geometry,
  ],
  "Total experts": [
    "The number of feed-forward experts in each modeled decoder block; one represents a dense FFN.",
    "Changes expert parameter memory, routing and communication. EP must divide the expert count. Routed-expert behavior is an analytical extrapolation from dense operator fits.",
    geometry,
  ],
  "Active experts / token": [
    "How many experts each token is routed to.",
    "Changes per-token FFN work and dispatch/combination traffic. Routing is assumed uniform; fused MoE kernels and routing imbalance are not validated.",
    assumption,
  ],
  "Model context limit": [
    "The maximum combined input and generated context allowed for this architecture.",
    "Validation requires input tokens + output tokens − 1 to fit. Raising the limit permits a scenario; it does not validate long-context timing or model quality.",
    geometry,
  ],
  "Tied embeddings": [
    "Whether input embeddings and the output head share one weight matrix.",
    "Counts shared parameter storage once. It does not remove the output-head calculation.",
    geometry,
  ],
  "QK normalization": [
    "Whether queries and keys have an additional normalization operation.",
    "Adds the modeled QK-normalization work to each decoder block. GPU timing can fall back or extrapolate if that operator is absent from a profile cohort.",
    geometry,
  ],
  "KV cache precision": [
    "The numeric format of resident attention keys and values, configured separately from weights.",
    "Changes KV capacity, attention traffic and prefill-to-decode handoff bytes. Conversion costs and numerical quality are not modeled. All active KV remains in GPU memory.",
    "Model geometry plus bytes per value. Precision changes beyond the original profile cohort are extrapolations.",
  ],
  "Input length": [
    "The full prompt length of every request in this uniform workload.",
    "Increases prefill work, attention context, KV capacity and handoff bytes. Optional prefix reuse reduces new prefill queries while preserving full context and KV size.",
    "Configured workload assumption. Selecting a Mooncake reuse fraction does not replay its original request lengths.",
  ],
  "Output length": [
    "The fixed total output-token count for each request, including the first token produced during prefill.",
    "The scheduler performs output length − 1 decode iterations. Longer outputs increase GPU occupancy and competition. TPOT is zero for a one-token output.",
    "Configured workload assumption; no sampled text, early stopping or model inference is performed.",
  ],
  Requests: [
    "The number of requests that must complete in this finite run.",
    "Changes batch count, queues, resident KV accounting and the startup/drain contribution to throughput. It is not a steady-state request rate.",
    assumption,
  ],
  "Arrival rate": [
    "A deterministic request arrival rate in requests per second.",
    "Request i arrives at i/rate seconds. Higher rates can increase competition and queues; lower rates can make throughput arrival-limited. Arrivals are not Poisson-sampled.",
    assumption,
  ],
  "Prefix reuse assumption": [
    "An optional fraction of prompt tokens whose KV is assumed already ready in HBM for every scenario request.",
    "No reuse computes complete prompts. Mooncake options use input-token-weighted earlier-arrival prefix potential with unlimited history; this is not a measured finite-cache hit rate. Remaining queries still attend to full context.",
    "Published Mooncake request traces; cold history, complete 512-token leading blocks and no reuse among tied arrivals. Ready HBM availability and transfer to this uniform workload remain assumptions.",
  ],
  "Separate prefill and decode": [
    "Whether prefill and decode use separate GPU pools.",
    "Enabled: each pool has independent hardware and policies, and full-context KV handoff adds link service. Disabled: both phases share the prefill pool and no inter-pool KV transfer is charged.",
    assumption,
  ],
  "Pool editor": [
    "Chooses which pool's hardware and server-policy values you are editing.",
    "Switching the editor does not change the scenario. Prefill hardware controls prompt processing; decode hardware controls subsequent token iterations. Colocated mode uses the prefill pool for both phases.",
    "Interface selection; the active pool name is shown above each pool-specific explanation.",
  ],
  "Hardware preset": [
    "A starting set of per-device compute, memory, topology, power and cost inputs for this pool.",
    "Selecting a preset replaces this pool's hardware fields. GPU identity does not select a correction table; fitted timings use the numeric hardware/shape features. Some presets are extrapolations.",
    "Preset specifications/references plus disclosed assumptions. A catalog entry does not establish measured kernel or end-to-end coverage.",
  ],
  TP: [
    "Tensor parallelism: the number of GPUs sharing a layer's tensors within this pool.",
    "Shards supported matrices/heads, changes per-GPU memory and adds tensor collectives. Must divide attention heads. Pool GPUs = TP × PP × EP × replicas.",
    "GPU fits include limited profiled TP configurations; new combinations and communication remain modeled.",
  ],
  PP: [
    "Pipeline parallelism: the number of sequential GPU stages sharing the model's layers.",
    "Splits layers and weights across stages and adds activation transfers and pipeline scheduling. Must divide transformer layers. More stages do not guarantee a lower latency.",
    assumption,
  ],
  EP: [
    "Expert parallelism: the number of GPU groups sharing routed experts.",
    "Shards expert parameters and adds dispatch/combine traffic. Must divide total experts; dense models require EP = 1. Uniform routing is assumed.",
    "Analytical MoE/communication extrapolation from dense operator timing evidence.",
  ],
  Replicas: [
    "The number of independently scheduled copies of this pool.",
    "Fixed groups are assigned round-robin across replicas. Adds GPUs, duplicated model weights and parallel capacity. Pool GPUs = TP × PP × EP × replicas.",
    assumption,
  ],
  "Server engine policy": [
    "An engine-efficiency override for this pool.",
    "Inherit uses the workload-wide policy. Other options change assumed efficiency and launch adjustments for this phase, without running that serving engine.",
    assumption,
  ],
  "Prefill microbatch": [
    "The maximum number of consecutive requests grouped for prefill.",
    "A fixed group waits for its last request to arrive. Sets GPU operation shapes and memory accounting. Decode may split the group; there is no batching timeout or cross-group merge.",
    assumption,
  ],
  "Decode batch cap": [
    "The largest decode subgroup formed from one completed prefill group.",
    "Smaller values split a group and change per-step work and contention. Larger values cannot merge separate prefill groups. This is not continuous batching.",
    assumption,
  ],
  "Prefill chunk": [
    "The maximum number of new prompt tokens processed per sequence in one prefill work item.",
    "Smaller chunks create more scheduled work items; each later chunk attends to the preceding full context. With prefix reuse, chunking starts after the assumed cached prefix.",
    "Configured chunking policy; GPU timing is evaluated at the resulting query/context shapes.",
  ],
  "Active batches / replica": [
    "The number of batches allowed into a phase's admission pool on one replica.",
    "An admission slot is held across all prefill chunks or decode iterations. Admitted batches still share the same stage resources; more slots do not create GPUs and can increase per-request waiting.",
    assumption,
  ],
  "Memory allocation limit": [
    "The usable fraction of each GPU's physical memory before the runtime reservation.",
    "Usable GB = physical GB × allocation limit − runtime reservation. Controls capacity validation, not HBM bandwidth or execution speed directly.",
    "Configured capacity allowance; allocation/fragmentation behavior is not measured.",
  ],
  "Runtime reservation": [
    "Additional per-GPU memory withheld for runtime overhead, in decimal GB.",
    "Subtracted after the allocation fraction. Weights, full per-request KV and modeled activations must fit the remainder; insufficient memory rejects the scenario.",
    assumption,
  ],
  "Spine bandwidth": [
    "The configured bandwidth of each prefill-to-decode KV handoff channel and the outermost fabric bottleneck.",
    "Handoff service uses payload/bandwidth plus startup. More bandwidth may reduce TTFT; it does not change decode kernel memory bandwidth. This is shared scenario topology, not a pool-specific HBM value.",
    network,
  ],
  "Spine latency": [
    "The configured startup delay of the outermost fabric/KV handoff, in microseconds.",
    "Adds to each modeled handoff and the selected outer-fabric service. GPU kernel timing is unchanged.",
    network,
  ],
  "KV transfer channels": [
    "The number of independent parallel handoff links between separate prefill and decode pools.",
    "Each channel has the full configured spine bandwidth. Batches are assigned round-robin; contention waits are tracked per channel. Increasing channels assumes more physical link capacity.",
    network,
  ],
  "Main memory": [
    "Physical memory capacity per GPU, in decimal GB; this is GPU memory, not host RAM.",
    "Sets capacity/headroom checks. Changing capacity alone does not increase HBM bandwidth or shorten kernels. GB uses 10⁹ bytes; native GiB uses 2³⁰ bytes.",
    "Hardware preset capacity/reference or user override; modeled payload accounting is separate from measured timing.",
  ],
  "Memory bandwidth": [
    "The per-GPU memory bandwidth used by the selected pool, in decimal TB/s.",
    "Changes memory-service features in the fitted GPU function and analytical roofline. It does not change capacity or the other pool. Slower prefill can lower observed TPOT by spacing decode arrivals farther apart while worsening TTFT and throughput.",
    "Hardware-rate feature in KAIST/Vidur-fitted functions. Large bandwidth edits are unvalidated hardware extrapolations; internal HBM effects are already included in kernel timing.",
  ],
  "Physical L2 cache": [
    "The physical GPU L2 capacity, in binary MiB.",
    "Used as a working-set/cache-pressure feature in applicable fitted functions. It is not a reserved KV pool or an explicit cache simulator. Zero selects an additive fitted fallback for cache-sensitive functions.",
    "Cache-pressure function fitted to operator profiles; no per-request cache-hit prediction.",
  ],
  "KV SRAM budget": [
    "An idealized per-GPU SRAM allocation exclusively available for KV reads in the analytical diagnostic.",
    "Divided among layers on a pipeline stage. Reduces analytical KV read traffic, but writes remain in GPU memory. It does not mechanically speed up the calibrated GPU functions or replace physical L2.",
    "Unvalidated analytical sensitivity assumption; not a measured SRAM allocation or cache policy.",
  ],
  "Compute efficiency": [
    "A dimensionless adjustment to effective compute rate relative to the nominal hardware peak.",
    "Higher values lower the compute feature; effective analytical rate includes peak × clock scale × efficiency. Fitted mode preserves the calibration reference of 0.65. Memory-bound kernels may barely change.",
    "Configured adjustment around the fitted reference; not a measured SM utilization percentage.",
  ],
  "Memory efficiency": [
    "A dimensionless adjustment to effective GPU memory bandwidth.",
    "Analytical effective bandwidth = bandwidth × memory efficiency. Fitted mode normalizes its memory feature around 0.8. Increasing it affects memory cost, not compute rate or capacity.",
    "Configured adjustment around the fitted reference; not a measured cache-hit rate.",
  ],
  "Compute clock scale": [
    "A multiplier on tensor/vector compute peaks for the selected pool; 2× represents twice the modeled compute rate.",
    "Does not automatically change memory bandwidth, launch cost or power. Faster prefill can feed a saturated decoder sooner and increase waiting between tokens even when total throughput improves. Compute-bound and memory-bound kernels respond differently.",
    "Analytical clock sensitivity applied to fitted GPU features; the exact clock change has not been validated by measured clock sweeps.",
  ],
  "Extra launch overhead": [
    "Additional startup time per modeled GPU operation, in microseconds.",
    "Added after the engine policy's launch adjustment. Accumulates across repeated operators/layers; it does not scale with token arithmetic.",
    assumption,
  ],
  "SM / CU count": [
    "The number of GPU multiprocessor or compute units used by the shape-informed utilization features.",
    "Changes roofline tile utilization and wave features in applicable fitted families. Its effect depends on the selected operator function; it is not a direct multiplier on peak TFLOPS or an instruction/occupancy simulation.",
    "Shape-informed feature fitted to operator profiles; the generic SM/CU abstraction is not cross-vendor occupancy validation.",
  ],
  "Tensor tile width": [
    "The assumed output tile width used by the analytical tensor-utilization heuristic.",
    "Changes the roofline tiling estimate, not the learned calibrated service function directly. It does not configure real GPU kernels.",
    assumption,
  ],
  "Intra-node bandwidth": [
    "The per-device one-direction bandwidth for the smallest GPU fabric domain.",
    "Sets the analytical communication bottleneck for groups within that domain. Used for collectives/pipeline traffic; distinct from HBM bandwidth.",
    network,
  ],
  "GPUs / chip domain": [
    "The maximum group size served by the smallest fast fabric domain.",
    "Larger groups select a more distant configured fabric tier. These thresholds describe a simplified hierarchy rather than physical link placement.",
    network,
  ],
  "Same-rail bandwidth": [
    "The per-device one-direction bandwidth of the same-rail/TOR tier.",
    "Bounds communication for groups exceeding the chip domain but within the configured same-rail capacity.",
    network,
  ],
  "GPUs / same rail": [
    "The configured number of GPUs in a same-rail domain.",
    "Determines when communication moves from the same-rail tier to the scale-out NIC tier, subject to the smaller domain threshold.",
    network,
  ],
  "Scale-out NIC": [
    "The per-device one-direction bandwidth of the scale-out fabric tier.",
    "Limits modeled collectives and pipeline communication at larger group sizes; the outer spine can impose a further bottleneck.",
    network,
  ],
  "GPUs / rack": [
    "The configured GPU count per rack in the simplified fabric hierarchy.",
    "Sets the group-size boundary between scale-out NIC and outer spine service. Together with cluster racks, also limits the allowed GPU count.",
    network,
  ],
  "Cluster racks": [
    "The number of racks available in the configured cluster.",
    "Validation requires pool GPUs ≤ GPUs per rack × cluster racks. This input changes the allowed GPU-count limit, not link bandwidth or service time directly.",
    network,
  ],
  "Base fabric latency": [
    "The base communication startup latency in microseconds.",
    "The hierarchy uses this delay for intra-node service and multiples of it for intermediate tiers. The outer tier uses the separate spine latency.",
    network,
  ],
  "All-reduce efficiency": [
    "The usable fraction of modeled link bandwidth for tensor-parallel all-reduce.",
    "Lower fractions lengthen all-reduce service. These costs are added separately from fitted GPU kernels; the value is not a measured utilization result.",
    network,
  ],
  "All-to-all efficiency": [
    "The usable fraction of modeled link bandwidth for expert dispatch/combine.",
    "Lower fractions increase the analytical MoE communication cost when EP is enabled.",
    network,
  ],
  "All-to-all offset": [
    "An extra microsecond startup offset on modeled all-to-all communication.",
    "Adds to expert dispatch/combine service independently of message bytes; relevant to expert-parallel scenarios.",
    network,
  ],
  "Device TDP": [
    "The configured GPU power cap in watts.",
    "Bounds estimated active power and energy. It does not throttle compute automatically. Default activity power uses idle + 0.85 × (TDP − idle), capped at TDP.",
    power,
  ],
  "Idle power": [
    "The assumed watts consumed by an allocated GPU when it is not doing modeled active work.",
    "Charged across the entire simulation horizon for every GPU. Also anchors the active-power estimate, so longer waits still consume estimated energy.",
    power,
  ],
  "GPU hourly cost": [
    "The assumed rental cost of one GPU per hour, in US dollars.",
    "Serving cost = GPU count × elapsed hours × price, summed across pools. It does not affect scheduling, timing or energy.",
    power,
  ],
  "Tensor energy": [
    "The assumed energy per tensor floating-point operation, in picojoules.",
    "If any component-energy coefficient is positive, component accounting replaces the default activity increment. Dynamic energy is capped by the idle-to-TDP power budget.",
    power,
  ],
  "Vector energy": [
    "The assumed energy per vector operation, in picojoules.",
    "Weights modeled normalization, activation and other vector work in component-energy mode. Zero contributes no vector increment in that mode.",
    power,
  ],
  "HBM energy": [
    "The assumed energy per byte of modeled GPU memory traffic, in picojoules.",
    "Weights analytical byte counters in component-energy mode. These counters are not measured DRAM transactions; zero contributes no memory increment in that mode.",
    power,
  ],
  "Network energy": [
    "The assumed GPU-side energy per modeled communication byte, in picojoules.",
    "Contributes to component-energy mode. It does not estimate the separate power of network switches, host CPUs or cooling.",
    power,
  ],
  "Selected model geometry": [
    "A read-only summary of the currently selected decoder's layer and attention geometry.",
    "These dimensions set parameter, activation and KV shapes. Use Customize architecture to edit them; changing a model can change timing coverage.",
    geometry,
  ],
  "Selected hardware specifications": [
    "A read-only summary of this pool's current per-GPU memory capacity and bandwidth.",
    "Capacity is decimal GB and bandwidth is decimal TB/s. Editing hardware fields changes this summary; a larger memory capacity alone does not provide a higher memory bandwidth.",
    "Selected hardware preset/reference or edited scenario values.",
  ],
  "Usable GPU memory": [
    "The memory available per GPU after allocation and runtime reservations.",
    "Usable GB = physical GB × allocation fraction − runtime reservation GB. Complete weights, resident KV and workspace must fit; there is no offload path.",
    "Conservative geometry-based capacity accounting; not a measured allocator peak.",
  ],
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
  const { config, catalog, phase } = context;
  let entry = entries[label];
  if (
    /^(BF16|FP16|FP8|FP4|TF32|INT8) tensor$/.test(label) ||
    label === "FP32 vector"
  ) {
    entry = [
      `Per-GPU dense ${label} peak rate, in trillions of operations per second. INT8 uses an equivalent integer operation-rate convention.`,
      "Sets the corresponding compute feature and analytical ceiling when that precision/operator uses it. Clock scale and compute efficiency apply separately. A zero peak marks that precision unsupported.",
      "Hardware preset/specification or user override; peak rate is not measured operator throughput. Quantized precision extrapolations are not validated by BF16/FP16 profiles.",
    ];
  }
  if (label.startsWith("Native ")) {
    const native: Record<string, Entry> = {
      "Native hardware": [
        "The published hardware/profile bundle used by the native LLMServingSim reference.",
        "Only the installed profile-supported hardware and TP choices are available. This path runs the native scheduler and ASTRA-Sim, without GPU model inference.",
      ],
      "Native requests": [
        "The number of requests supplied to the native reference experiment.",
        "Changes the workload and measured-simulator request CSV; it does not sample or generate natural-language output.",
      ],
      "Native input tokens": [
        "The uniform prompt token count in the native workload.",
        "Changes profile-derived prefill work and KV payload. Original native limits apply.",
      ],
      "Native output tokens": [
        "The requested fixed output length in the native workload.",
        "Controls the number of native decode iterations; these are token counts, not generated text.",
      ],
      "Native tensor parallel": [
        "The tensor-parallel degree supported by the selected native profile bundle.",
        "Changes GPU sharding and native collectives; arbitrary new TP values require matching profiles.",
      ],
      "Native memory capacity": [
        "The device-memory capacity supplied to the native reference, in binary GiB.",
        "Changes the native memory configuration while preserving recorded kernel-latency profiles. GiB uses 2³⁰ bytes.",
      ],
      "Native memory bandwidth": [
        "The memory-bandwidth configuration supplied to the native reference.",
        "Changes the native memory-system setting; measured operator profile costs remain fixed. This does not reprofile GPU kernels.",
      ],
      "Native link bandwidth": [
        "The native interconnect bandwidth in binary GiB/s.",
        "Changes ASTRA-Sim communication service while preserving GPU kernel profiles. GiB/s differs from the behavioral model's decimal GB/s.",
      ],
      "Native link latency": [
        "The native communication startup delay in nanoseconds.",
        "Changes the native network configuration and communication timing; kernel profile values stay fixed.",
      ],
    };
    entry = native[label];
    return {
      title: label,
      meaning: entry?.[0] || label,
      effect: entry?.[1] || "Configured native-reference input.",
      evidence:
        "Local profile-driven LLMServingSim 2.0 / ASTRA-Sim reference. Native inputs are independent of the behavioral scenario.",
    };
  }
  if (!entry)
    return {
      title: label,
      meaning: "A configurable scenario input.",
      effect: "Changes are applied when Run scenario is selected.",
      evidence: assumption,
    };
  let evidence = entry[2] || assumption;
  if (
    label === "Prefix reuse assumption" &&
    config.prefix_reuse_source !== "none"
  ) {
    const record =
      catalog.prefix_reuse_evidence.workloads[config.prefix_reuse_source];
    evidence += ` Selected source: ${record.label}; ${(record.fraction * 100).toFixed(2)}% token-weighted potential across ${record.requests.toLocaleString()} requests. Applied uniformly to the configured prompt length, with at least one recomputed token.`;
  }
  const scoped = !globalLabels.has(label);
  return {
    title: scoped
      ? `${phase === "prefill" ? "Prefill" : "Decode"} · ${label}`
      : label,
    meaning: entry[0],
    effect: entry[1],
    evidence,
  };
}
