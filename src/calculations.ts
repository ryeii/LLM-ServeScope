import { explainCalculation, horizonCalculation } from "./derivations";
/** Explanations use the completed run snapshot, never unsubmitted form inputs. */
export type Calculation = {
  title: string;
  formula: string;
  source?: string;
  summary?: string;
  links?: { label: string; url: string }[];
  variables: {
    symbol: string;
    label: string;
    value: number | string;
    unit?: string;
    explain?: () => Calculation;
  }[];
  result: number | string;
  unit: string;
  note?: string;
};
type Obj = Record<string, any>;
export const v = (
  symbol: string,
  label: string,
  value: number | string,
  unit = "",
  explain?: () => Calculation,
) => ({ symbol, label, value, unit, explain });
export const calc = (
  title: string,
  formula: string,
  variables: Calculation["variables"],
  result: number | string,
  unit: string,
  note?: string,
): Calculation => ({ title, formula, variables, result, unit, note });
const bytes: Obj = {
  fp4: 0.5,
  fp8: 1,
  int8: 1,
  bf16: 2,
  fp16: 2,
  tf32: 4,
  fp32: 4,
};
const metricNames: Obj = {
  throughput_tps: "Generation throughput",
  request_throughput: "Request throughput",
  prompt_tps: "Prompt throughput",
  ttft_mean_ms: "Time to first token",
  ttft_p95_ms: "P95 time to first token",
  tpot_mean_ms: "Time per output token",
  latency_p95_ms: "P95 request latency",
  elapsed_ms: "Simulated elapsed time",
  gpu_count: "GPU count",
  energy_j: "GPU energy",
  tokens_per_joule: "Energy efficiency",
  tokens_per_gpu_s: "Generation throughput per GPU",
  cost_usd: "GPU rental cost",
  cost_per_million_tokens: "Serving cost",
  gpu_utilization: "GPU resource occupancy",
};
const simulationNote =
  "From this completed SimPy run. Operator costs, FIFO resource reservations, arrivals and transfers determine the timeline; this is a behavioral estimate.";
export const numberText = (x: number | string) =>
  typeof x === "number"
    ? Number.isFinite(x)
      ? x !== 0 && (Math.abs(x) < 0.00001 || Math.abs(x) >= 1e10)
        ? x.toExponential(6)
        : x.toLocaleString("en-US", { maximumSignificantDigits: 9 })
      : "—"
    : x;

function metricBase(r: Obj, key: string): Calculation {
  const m = r.metrics,
    c = r.config,
    requests = r.requests,
    terms = r.accounting?.energy_terms || [];
  const n = c.requests * c.output_tokens,
    t = m.elapsed_ms / 1000,
    g = m.gpu_count;
  const throughput = [
    v("N_req", "Completed requests", c.requests),
    v("L_out", "Output tokens per request", c.output_tokens, "tokens"),
    v("T_s", "Simulation horizon", t, "s"),
  ];
  const title = metricNames[key] || key;
  if (key === "throughput_tps")
    return calc(
      title,
      "Throughput = N_req × L_out / T_s",
      throughput,
      m[key],
      "tokens/s",
      simulationNote,
    );
  if (key === "request_throughput")
    return calc(
      title,
      "Request rate = N_req / T_s",
      throughput,
      m[key],
      "requests/s",
      simulationNote,
    );
  if (key === "prompt_tps")
    return calc(
      title,
      "Prompt throughput = N_req × L_in / T_s",
      [...throughput, v("L_in", "Input tokens per request", c.input_tokens)],
      m[key],
      "tokens/s",
    );
  if (key === "tokens_per_gpu_s")
    return calc(
      title,
      "Tokens/GPU-second = N_req × L_out / (T_s × G)",
      [...throughput, v("G", "Allocated GPUs", g)],
      m[key],
      "tokens/GPU-second",
    );
  if (key === "tokens_per_joule")
    return calc(
      title,
      "Tokens/J = N_req × L_out / max(E, 10⁻¹²)",
      [
        ...throughput.slice(0, 2),
        v("E", "Estimated GPU energy", m.energy_j, "J"),
      ],
      m[key],
      "tokens/J",
      "Energy uses the pool power coefficients shown in the GPU energy calculation. Host CPUs, network equipment and cooling are excluded.",
    );
  if (key === "elapsed_ms")
    return calc(
      title,
      "T_ms = maxᵢ(finishᵢ)",
      [
        v(
          "finishᵢ",
          "Request completion timestamps",
          requests.map((q: Obj) => numberText(q.finish_ms)).join(", "),
          "ms",
        ),
      ],
      m[key],
      "ms",
      simulationNote,
    );
  if (key === "gpu_count")
    return calc(
      title,
      "G = Σ_unique pools TP × PP × EP × replicas",
      terms.flatMap((q: Obj) => {
        const p = c[q.phase];
        return [
          v(q.phase + " TP", "Tensor parallelism", p.tp),
          v(q.phase + " PP", "Pipeline parallelism", p.pp),
          v(q.phase + " EP", "Expert parallelism", p.ep),
          v(q.phase + " replicas", "Pool replicas", p.replicas),
        ];
      }),
      m[key],
      "GPUs",
      c.separated
        ? "Prefill and decode are separate pools."
        : "Colocated prefill/decode share one pool; GPUs are counted once.",
    );
  if (key === "gpu_utilization") {
    const busy = Object.entries(r.accounting?.busy_ms || {}).reduce(
      (sum, [resource, ms]) => {
        const p = c[resource.startsWith("decode:") ? "decode" : "prefill"];
        return sum + Number(ms) * p.tp * p.ep;
      },
      0,
    );
    return calc(
      title,
      "Occupancy = Σ_resource(busy_ms × TP × EP) / (T_ms × G)",
      [
        v("B", "Weighted GPU busy time", busy, "GPU·ms"),
        v("T_ms", "Simulation horizon", m.elapsed_ms, "ms"),
        v("G", "Allocated GPUs", g),
      ],
      m[key],
      "fraction",
      "Multiply by 100 for the displayed percentage. Stage reservations include modeled memory/communication stalls; this is not tensor-core utilization.",
    );
  }
  if (key === "cost_usd" || key === "cost_per_million_tokens") {
    return calc(
      title,
      key === "cost_usd"
        ? "Cost = Σ_pool G_pool × price_pool × T_ms / 3,600,000"
        : "Cost/million = Cost × 1,000,000 / (N_req × L_out)",
      [
        ...terms.flatMap((q: Obj) => [
          v(q.phase + " G", "Allocated GPUs", q.gpus),
          v(
            q.phase + " price",
            "GPU rental assumption",
            q.hourly_cost,
            "USD/GPU-hour",
          ),
        ]),
        v("T_ms", "Horizon", m.elapsed_ms, "ms"),
        v("Cost", "Total GPU rental cost", m.cost_usd, "USD"),
        v("N_req × L_out", "Completed output tokens", n),
      ],
      m[key],
      key === "cost_usd" ? "USD" : "USD/million output tokens",
      "Only allocated GPU rental is included; rates are scenario assumptions.",
    );
  }
  if (key === "energy_j") {
    const variables = [v("T_s", "Simulation horizon", t, "s")];
    for (const q of terms) {
      variables.push(
        v(q.phase + " G", "Allocated GPUs", q.gpus),
        v(q.phase + " P_idle", "Idle power", q.idle_watts, "W"),
        v(q.phase + " P_TDP", "Power cap", q.tdp_watts, "W"),
        v(
          q.phase + " B_s",
          "Weighted busy time",
          q.weighted_busy_ms / 1000,
          "GPU·s",
        ),
        v(q.phase + " E_dynamic", "Dynamic GPU energy", q.dynamic_j, "J"),
      );
      if (q.mode === "component") {
        for (const [k, value] of Object.entries(q.counters))
          variables.push(
            v(q.phase + " " + k, "Accumulated work", Number(value)),
          );
        for (const [k, value] of Object.entries(q.coefficients))
          variables.push(
            v(
              q.phase + " " + k,
              "Energy coefficient",
              Number(value),
              "pJ/unit",
            ),
          );
      } else
        variables.push(
          v(q.phase + " a", "Activity coefficient", 0.85),
          v(
            q.phase + " P_active",
            "min(P_TDP, P_idle + a × (P_TDP − P_idle))",
            q.active_watts,
            "W",
          ),
        );
    }
    return calc(
      title,
      "E = Σ_pool [P_idle × G × T_s + E_dynamic]\nActivity mode: E_dynamic = (P_active − P_idle) × B_s\nComponent mode: E_dynamic = min(Σ work_k × pJ_k × 10⁻¹², (P_TDP − P_idle) × B_s)",
      variables,
      m[key],
      "J",
      "Power and energy coefficients are scenario assumptions. Busy time includes modeled stage occupancy; no measured power-counter calibration is claimed.",
    );
  }
  const requestKey = key.startsWith("ttft")
    ? "ttft_ms"
    : key.startsWith("tpot")
      ? "tpot_ms"
      : "latency_ms";
  const values = requests
    .map((q: Obj) => q[requestKey])
    .sort((a: number, b: number) => a - b);
  if (key.endsWith("p95_ms")) {
    const x = (values.length - 1) * 0.95,
      lo = Math.floor(x),
      hi = Math.ceil(x);
    return calc(
      title,
      "P95 = a[lo] + (a[hi] − a[lo]) × (x − lo)\nx = (N − 1) × p; lo = floor(x); hi = ceil(x)",
      [
        v("p", "Quantile", 0.95),
        v("N", "Request observations", values.length),
        v("x", "Fractional index", x),
        v("a[lo]", "Lower ordered value", values[lo], "ms"),
        v("a[hi]", "Upper ordered value", values[hi], "ms"),
      ],
      m[key],
      "ms",
      "Linear interpolation of per-request latencies from the simulated timeline.",
    );
  }
  return calc(
    title,
    key === "tpot_mean_ms"
      ? "TPOT_mean = Σᵢ [(finishᵢ − first_tokenᵢ) / max(1, L_out − 1)] / N"
      : "TTFT_mean = Σᵢ (first_tokenᵢ − arrivalᵢ) / N",
    [
      v("N", "Completed requests", values.length),
      v(
        "Σ durations",
        "Sum of per-request " +
          (key === "tpot_mean_ms" ? "inter-token means" : "TTFTs"),
        values.reduce((a: number, b: number) => a + b, 0),
        "ms",
      ),
      v("L_out", "Output tokens per request", c.output_tokens),
    ],
    m[key],
    "ms",
    key === "tpot_mean_ms" && c.output_tokens === 1
      ? "One output token has no inter-token interval; TPOT is zero."
      : simulationNote,
  );
}

function memoryBase(r: Obj, phase: string, key: string): Calculation {
  const m = r.model,
    c = r.config,
    p = c[phase],
    h = p.hardware,
    a = r.accounting,
    mem = r.memory[phase],
    d = bytes[c.precision],
    dk = bytes[c.kv_precision];
  const common = [
    v("TP", "Tensor parallelism", p.tp),
    v("PP", "Pipeline parallelism", p.pp),
    v("EP", "Expert parallelism", p.ep),
  ];
  if (key === "weights_GB")
    return calc(
      phase + " weight memory",
      "Weights_GB = (P_shared + P_expert / EP) × d_weight / (TP × PP × 10⁹)",
      [
        ...common,
        v("P_shared", "Shared model parameters", a.shared_parameters),
        v("P_expert", "All expert parameters", a.expert_parameters),
        v("d_weight", "Weight bytes per element", d),
      ],
      mem[key],
      "GB",
      "Model geometry determines parameter counts; tied weights are counted once.",
    );
  if (key === "kv_GB") {
    const headWidth = Math.max(1, m.kv_heads / p.tp) * m.head_dim,
      context = c.input_tokens + c.output_tokens - 1;
    const concurrent = mem.resident_requests;
    return calc(
      phase + " KV memory",
      "KV_GB = 2 × N_resident × L_context × max(1, H_KV / TP) × D_head × d_KV × L_layers / (PP × 10⁹)",
      [
        ...common,
        v(
          "N_resident",
          "Conservatively resident requests per replica",
          Math.round(concurrent),
        ),
        v("L_context", "Maximum context", context, "tokens"),
        v("H_KV", "KV heads", m.kv_heads),
        v("D_head", "Head dimension", m.head_dim),
        v("d_KV", "KV bytes per element", dk),
        v("L_layers", "Transformer layers", m.layers),
      ],
      mem[key],
      "GB",
      "Includes queued batches assigned to a replica, not only currently active requests. Full KV payload remains in device memory.",
    );
  }
  if (key === "activation_GB") {
    const batch = Math.min(
      c.prefill.batch_size,
      c.requests,
      ...(phase === "decode" ? [c.decode.batch_size] : []),
    );
    const tokens =
      phase === "prefill" ? Math.min(c.input_tokens, c.prefill_chunk) : 1;
    return calc(
      phase + " activation memory",
      "Activation_GB = B × n × H × d_weight × k_workspace / (TP × 10⁹)",
      [
        ...common,
        v("B", "Phase batch", batch),
        v("n", "Chunk tokens", tokens),
        v("H", "Hidden width", m.hidden),
        v("d_weight", "Bytes per element", d),
        v("k_workspace", "Workspace multiplier", 8),
      ],
      mem[key],
      "GB",
      "Conservative analytical workspace estimate, not measured allocator usage.",
    );
  }
  const map: Obj = {
    total_GB: ["Resident memory", "Total = weights + KV + activation"],
    capacity_GB: [
      "Usable GPU memory",
      "Capacity = physical_GB × utilization − reserved_GB",
    ],
    headroom_GB: ["GPU memory headroom", "Headroom = capacity_GB − total_GB"],
    gpus: ["Pool GPU count", "GPUs = TP × PP × EP × replicas"],
  };
  const [title, formula] = map[key] || [key, "Configured value"];
  const explanation = calc(
    phase + " " + title,
    formula,
    [
      ...common,
      v("replicas", "Replicas", p.replicas),
      v("physical_GB", "Physical memory", h.main_memory_GB, "GB"),
      v("utilization", "Usable fraction", p.memory_utilization),
      v("reserved_GB", "Runtime reservation", p.runtime_reserved_GB, "GB"),
      v("weights", "Weight payload", mem.weights_GB, "GB"),
      v("KV", "KV payload", mem.kv_GB, "GB"),
      v("activation", "Workspace", mem.activation_GB, "GB"),
      v("capacity_GB", "Usable capacity", mem.capacity_GB, "GB"),
      v("total_GB", "Resident payload", mem.total_GB, "GB"),
    ],
    mem[key],
    key === "gpus" ? "GPUs" : "GB",
  );
  const dependencies: Obj = {
    total_GB: ["weights", "KV", "activation"],
    capacity_GB: ["physical_GB", "utilization", "reserved_GB"],
    headroom_GB: ["capacity_GB", "total_GB"],
    gpus: ["TP", "PP", "EP", "replicas"],
  };
  explanation.variables = explanation.variables.filter((row) =>
    (dependencies[key] || []).includes(row.symbol),
  );
  return explanation;
}

function operationBase(
  r: Obj,
  phase: string,
  o: Obj,
  key: string,
): Calculation {
  const p = r.config[phase],
    h = p.hardware;
  const title =
    phase +
    " · " +
    o.name +
    " · " +
    (
      {
        duration_ms: "Service time",
        compute_ms: "Nominal compute time",
        memory_ms: "Nominal memory time",
        network_ms: "Network time",
        achieved_tflops: "Attained throughput",
        arithmetic_intensity: "Arithmetic intensity",
        flops: "Compute work",
        hbm_bytes: "Modeled HBM traffic",
      } as Obj
    )[key];
  const variables = [
    v("F", "Modeled FLOPs", o.flops, "FLOP"),
    v("Q", "Modeled traffic", o.hbm_bytes, "bytes"),
  ];
  if (key === "arithmetic_intensity")
    return calc(title, "AI = F / max(1, Q)", variables, o[key], "FLOP/byte");
  if (key === "achieved_tflops")
    return calc(
      title,
      "Attained = F / max(T_ms × 10⁹, 1)",
      [
        ...variables,
        v("T_ms", "Selected operator service time", o.duration_ms, "ms"),
      ],
      o[key],
      "TFLOP/s",
    );
  if (key === "flops" || key === "hbm_bytes") {
    const row =
      o.fit_work || o.batch
        ? {
            ...o,
            batch: o.batch ?? o.fit_batch,
            tokens: o.tokens ?? o.fit_tokens,
            context: o.context ?? o.fit_context,
          }
        : r.layers.find(
            (item: Obj) => item.phase === phase && item.id === o.id,
          ) || o;
    const batch =
        row.batch || Math.min(r.config[phase].batch_size, r.config.requests),
      tokens =
        row.tokens ||
        (phase === "decode"
          ? 1
          : Math.min(r.config.prefill_chunk, r.config.input_tokens)),
      context = row.context || tokens;
    const forms: Obj = {
      "Input RMSNorm": "5 × n × H / TP",
      "FFN RMSNorm": "5 × n × H / TP",
      "QK normalization": "5 × n × (Q_heads + K_heads) / TP",
      "QKV projection": "2 × n × H × (Q_heads + 2K_heads) / TP",
      "Rotary position": "6 × n × Q_heads / TP",
      Attention: "4 × B × Q_heads × [t × max(0, context−t) + t(t+1)/2] / TP",
      "Output projection": "2 × n × Q_heads × H / TP",
      "Gate + up projection": "4 × n × H × I × top_k / (TP × EP)",
      SwiGLU: "5 × n × I × top_k / (TP × EP)",
      "Down projection": "2 × n × H × I × top_k / (TP × EP)",
      "Expert router": "2 × n × H × experts / TP",
    };
    const relevant = new Set(
      (key === "flops"
        ? (forms[o.name] || "0") + "; n = B × t"
        : "weight_bytes activation_bytes kv_hbm_bytes"
      )
        .replace(/(\d)([A-Za-z_])/g, "$1 $2")
        .match(/[A-Za-z_]+/g) || [],
    );
    return calc(
      title,
      key === "flops"
        ? "F = " + (forms[o.name] || "0") + "; n = B × t"
        : "Q = weight_bytes + activation_bytes + kv_hbm_bytes",
      [
        v("weight_bytes", "Weights read", o.weight_bytes || 0, "bytes"),
        v(
          "activation_bytes",
          "Activation traffic",
          o.activation_bytes || 0,
          "bytes",
        ),
        v("kv_hbm_bytes", "KV traffic", o.kv_hbm_bytes || 0, "bytes"),
        v("H", "Hidden width", r.model.hidden),
        v("I", "Intermediate width", r.model.intermediate),
        v(
          "Q_heads",
          "Query heads × head dimension",
          r.model.heads * r.model.head_dim,
        ),
        v(
          "K_heads",
          "KV heads × head dimension",
          r.model.kv_heads * r.model.head_dim,
        ),
        v("top_k", "Experts per token", r.model.top_k),
        v("experts", "Total experts", r.model.experts),
        v("TP", "Tensor parallelism", p.tp),
        v("EP", "Expert parallelism", p.ep),
        v("B", "Representative batch", batch),
        v("t", "Tokens per sequence", tokens),
        v("n", "Batch × tokens", batch * tokens),
        v("context", "Context length", context),
      ].filter((row) => relevant.has(row.symbol)),
      o[key],
      key === "flops" ? "FLOP" : "bytes",
      "Tensor geometry and precision determine work; traffic is nominal, not measured memory-controller transactions.",
    );
  }
  if (key === "network_ms" && o.timing_source !== "analytical-network")
    return calc(
      title,
      "T_network = 0",
      [],
      0,
      "ms",
      "This GPU operator has no separate modeled network transfer. Collective operations are listed separately.",
    );
  if (key === "compute_ms" && o.flops === 0)
    return calc(
      title,
      "C_ms = 0",
      [],
      0,
      "ms",
      "Communication is charged by its separate network formula; no analytical compute demand is assigned here.",
    );
  if (key === "memory_ms")
    return calc(
      title,
      "M_ms = Q / (BW_TBps × 10¹² × η_memory) × 1000",
      [
        ...variables,
        v("BW_TBps", "Device-memory bandwidth", h.dram_bandwidth_TBps, "TB/s"),
        v("η_memory", "Assumed memory efficiency", h.memory_efficiency),
      ],
      o[key],
      "ms",
      "Nominal resource-work estimate. This is not a decomposition of the separately fitted service time.",
    );
  if (key === "compute_ms") {
    const vector = [
      "Input RMSNorm",
      "FFN RMSNorm",
      "QK normalization",
      "Rotary position",
      "SwiGLU",
    ].includes(o.name);
    const peak = vector
      ? h.fp32_tflops
      : h[
          (r.config.precision === "fp32"
            ? "fp32"
            : r.config.precision + "_tensor") + "_tflops"
        ];
    const engine =
      p.engine === "inherit" ? r.config.inference_engine : p.engine;
    return calc(
      title,
      "C_ms = F / (P_TFLOPS × 10¹² × clock × η_compute × u_tile × η_engine) × 1000",
      [
        ...variables,
        v("P_TFLOPS", "Precision-specific peak", peak, "TFLOP/s"),
        v("clock", "Compute clock scale", h.clock_scale),
        v("η_compute", "Assumed compute efficiency", h.compute_efficiency),
        v("u_tile", "Tile utilization", o.tile_utilization),
        v(
          "η_engine",
          "Engine factor",
          ({ vllm: 0.97, trtllm: 1, eager: 0.8 } as Obj)[engine],
        ),
      ],
      o[key],
      "ms",
      "Nominal analytical compute demand; it is not a measured compute-active interval.",
    );
  }
  if (key === "network_ms" || o.timing_source === "analytical-network") {
    const group = p.tp * p.ep,
      reduce = o.name.includes("all-reduce"),
      g = o.name.startsWith("EP") ? p.ep : p.tp;
    const fabric = r.phases[phase].detail.fabric,
      eff = reduce
        ? h.collective_allreduce_fraction
        : h.collective_alltoall_fraction;
    return calc(
      title,
      "T_network_ms = Q_network / (BW_GBps × 10⁹ × η_collective) × 1000 + phases × latency_us / 1000 + offset_us / 1000",
      [
        v(
          "Q_network",
          "Ring/dispatch communication volume",
          o.network_bytes || 0,
          "bytes",
        ),
        v("BW_GBps", "Limiting fabric bandwidth", fabric.bw, "GB/s"),
        v("η_collective", "Collective efficiency", eff),
        v("phases", "Ring phases", g > 1 ? (reduce ? 2 * (g - 1) : g - 1) : 0),
        v("latency_us", "Fabric latency", fabric.latency, "µs"),
        v(
          "offset_us",
          "Dispatch overhead",
          o.name.startsWith("EP") ? h.collective_alltoall_offset : 0,
          "µs",
        ),
        v("group", "Fabric group size", group),
      ],
      o[key],
      "ms",
      o.network_ms === 0
        ? "This operator has no separate modeled network transfer."
        : "Analytical collective estimate; not calibrated by the GPU operator profiles.",
    );
  }
  if (o.fit_parameters) {
    const model = o.fit_parameters,
      family = model.family,
      x = o.shape_features_ns;
    const forms: Obj = {
      F0: "L + a C + b M",
      F1: "L + a C + b M",
      F2: "L + max(a C, b M)",
      F3: "L + √((a C)² + (b M)²)",
      F4: "L + a W + b M",
      F5: "L + a C + M × [b₀(1−g) + b₁g]",
      F6: "L + C × [a₀(1−g)+a₁g] + M × [b₀(1−g)+b₁g]",
      F7: "L + a C + b M + d M g_cache",
      F8: "L + a C + b M_correct",
      F9: "L_branch + b_branch M_correct",
    };
    const labels: Obj = {
      F0: ["L", "a", "b"],
      F1: ["L", "a", "b"],
      F2: ["L", "a", "b"],
      F3: ["L", "a", "b"],
      F4: ["L", "a", "b"],
      F5: ["L", "a", "b₀", "b₁"],
      F6: ["L", "a₀", "a₁", "b₀", "b₁"],
      F7: ["L", "a", "b", "d"],
      F8: ["L", "a", "b"],
    };
    const coeffs: Calculation["variables"] = [];
    if (family === "F9") {
      const branch =
        o.fit_token_count <= model.threshold_tokens ? "small_index" : "gather";
      coeffs.push(
        v("branch", "Selected dispatch branch", branch),
        v("threshold", "Switch token count", model.threshold_tokens),
        v(
          "L_branch",
          "Fitted launch/intercept",
          model.branches[branch][0],
          "ns",
        ),
        v("b_branch", "Fitted memory-feature slope", model.branches[branch][1]),
      );
    } else
      model.coefficients.forEach((value: number, i: number) =>
        coeffs.push(
          v(
            labels[family][i],
            i === 0
              ? "Fitted intercept"
              : "Fitted " + labels[family][i] + " coefficient",
            value,
            i === 0 ? "ns" : "dimensionless",
          ),
        ),
      );
    const featureRows: Obj = {
      C: v("C", "Nominal compute feature", x[1], "ns"),
      M: v("M", "Nominal memory feature", x[2], "ns"),
      W: v("W", "Padded tile-wave feature", x[3], "ns"),
      w: v("w", "Work tiles per SM", x[4]),
      r: v("r", "Nominal working set / physical L2", x[5]),
      M_correct: v(
        "M_correct",
        "Corrected embedding memory feature",
        x[6],
        "ns",
      ),
    };
    const keys =
      family === "F9"
        ? ["M_correct"]
        : family === "F8"
          ? ["C", "M_correct"]
          : family === "F4"
            ? ["W", "M"]
            : family === "F7"
              ? ["C", "M", "r"]
              : ["F5", "F6"].includes(family)
                ? ["C", "M", "w"]
                : ["C", "M"];
    const vars = [
      ...coeffs,
      ...keys.map((key) => featureRows[key]),
      v("C_floor", "Arithmetic floor", o.fit_compute_floor_ns, "ns"),
      v("T_base", "Fitted base service time", o.fit_base_ns, "ns"),
      v(
        "κ_engine; Δ_engine; L_extra",
        "Engine multiplier; engine and launch offsets",
        `${numberText(o.fit_engine_multiplier)}; ${numberText(o.fit_engine_offset_ns)} ns; ${numberText(o.fit_extra_launch_ns)} ns`,
      ),
    ];
    if (["F5", "F6", "F7"].includes(family))
      vars.push(
        v("s", "Fitted transition scale", model.scale),
        v(
          family === "F7" ? "g_cache" : "g",
          "Transition gate",
          family === "F7"
            ? x[5] / (x[5] + model.scale)
            : x[4] / (x[4] + model.scale),
        ),
      );
    return calc(
      title,
      `T_base = max(1 ns, C_floor, ${forms[family]})\nT_ms = max(1 ns, C_floor, κ_engine T_base + Δ_engine + L_extra) / 10⁶` +
        (["F5", "F6", "F7"].includes(family)
          ? family === "F7"
            ? "\ng_cache = r/(r+s)"
            : "\ng = w/(w+s)"
          : ""),
      vars,
      o.duration_ms,
      "ms",
      `${family} · ${o.fit_cohort.toUpperCase()} fit. ${o.fit_fallback_reason || "Coefficients are shared across GPU/model identities."} Fitted terms describe scaling, not measured compute/memory decomposition.`,
    );
  }
  return calc(
    title,
    "T_ms = max(C_ms, M_ms) + L_ms",
    [
      v("C_ms", "Nominal compute demand", o.compute_ms, "ms"),
      v("M_ms", "Nominal memory demand", o.memory_ms, "ms"),
      v("L_ms", "Launch overhead", o.launch_ms, "ms"),
    ],
    o.duration_ms,
    "ms",
    "Original roofline approximation with ideal overlap.",
  );
}

function componentBase(r: Obj, key: string): Calculation {
  const contributions = r.timeline.map((e: Obj) => {
    const p = r.config[e.phase === "decode" ? "decode" : "prefill"];
    return (
      (e.components[key] || 0) * (e.phase === "transfer" ? 1 : p.tp * p.ep)
    );
  });
  return calc(
    key + " resource service demand",
    "Demand = Σ_intervals component_ms × GPU_multiplier",
    [
      v("component", "Resource component", key),
      v("N", "Simulated intervals", r.timeline.length),
      v(
        "Σ contribution",
        "Sum of weighted interval estimates",
        contributions.reduce((a: number, b: number) => a + b, 0),
        "GPU·ms",
      ),
    ],
    r.component_work_ms[key],
    "GPU·ms",
    "Stage and pipeline contributions are multiplied by TP × EP; KV handoff links count once. These nominal resource demands overlap and do not sum to the fitted elapsed time.",
  );
}

function trafficBase(r: Obj, key: string): Calculation {
  const pre = r.accounting.phase_traffic.prefill[key],
    dec = r.accounting.phase_traffic.decode[key];
  return calc(
    key === "hbm_bytes" ? "System HBM traffic" : "System network traffic",
    "Q_system = Q_prefill + Q_decode",
    [
      v("Q_prefill", "Accumulated prefill traffic", pre, "bytes"),
      v("Q_decode", "Accumulated decode traffic", dec, "bytes"),
      v("L", "Transformer layers", r.model.layers),
      v("N_req", "Requests", r.config.requests),
      v("L_in", "Input length", r.config.input_tokens),
      v("L_out", "Output length", r.config.output_tokens),
    ],
    r.traffic[key],
    "bytes",
    key === "hbm_bytes"
      ? "Each stage accumulates nominal layer-operation bytes × layers per stage × TP × EP, plus separately counted output-head traffic. Embedding, final normalization and sampling timing do not add separate traffic counters. These are estimates, not memory-controller measurements."
      : "Includes collective bytes × layers per stage × TP × EP, pipeline payloads × TP × EP, and each KV handoff payload once. KV handoff is attributed to prefill.",
  );
}

export function intervalCalculation(e: Obj): Calculation {
  return calc(
    "Timeline interval",
    "Duration_ms = end_ms − start_ms",
    [
      v("start_ms", "Resource acquisition timestamp", e.start_ms, "ms"),
      v("end_ms", "Resource release timestamp", e.end_ms, "ms"),
      v("batch", "Requests in batch", e.batch_size),
      v("context", "Context tokens", e.context),
    ],
    e.duration_ms,
    "ms",
    "SimPy resource reservation after queueing. GPU stages sum selected per-operator costs across layers plus applicable endpoints; transfer intervals use bytes/bandwidth plus latency.",
  );
}

export function quantileCalculation(
  requests: Obj[],
  key: string,
  p: number,
  run?: Obj,
): Calculation {
  const values = requests.map((r) => r[key]).sort((a, b) => a - b),
    x = (values.length - 1) * p,
    lo = Math.floor(x),
    hi = Math.ceil(x),
    result = values[lo] + (values[hi] - values[lo]) * (x - lo);
  const explanation = calc(
    key + " quantile",
    "Q(p) = a[lo] + (a[hi] − a[lo]) × (x−lo); x = (N−1)p",
    [
      v("p", "Quantile", p),
      v("N", "Request observations", values.length),
      v("x", "Fractional ordered index", x),
      v("a[lo]", "Lower value", values[lo], "ms"),
      v("a[hi]", "Upper value", values[hi], "ms"),
    ],
    result,
    "ms",
    "Linear interpolation of per-request simulated timestamps.",
  );
  return run
    ? explainCalculation(run, explanation, { metric: key })
    : explanation;
}

function sramBase(r: Obj, phase: string): Calculation {
  const m = r.model,
    p = r.config[phase],
    detail = r.phases[phase].detail;
  return calc(
    "Ideal KV read reuse",
    "Hit fraction = min(1, reserved_bytes_per_layer / max(1, KV_read_bytes))",
    [
      v(
        "reserved_bytes_per_layer",
        "Reserved KV SRAM per layer",
        (p.hardware.sram_cache_MB * 1e6) / (m.layers / p.pp),
        "bytes",
      ),
      v(
        "KV_read_bytes",
        "Nominal KV bytes read",
        detail.kv_read_bytes,
        "bytes",
      ),
    ],
    detail.sram_hit_fraction,
    "fraction",
    "An ideal reuse assumption, not a measured physical-cache hit rate. Multiply by 100 for the displayed percent.",
  );
}

export function scaled(
  c: Calculation,
  factor: number,
  unit: string,
): Calculation {
  return {
    ...c,
    formula: c.formula + `\nDisplayed value = base_result × ${factor}`,
    variables: [
      ...c.variables,
      v("base_result", "Unscaled result", c.result, c.unit, () => c),
      v("display_scale", "Unit conversion", factor),
    ],
    result: Number(c.result) * factor,
    unit,
  };
}
export function deltaCalculation(
  current: Obj,
  pinned: Obj,
  key: string,
): Calculation {
  return calc(
    "Change versus pinned baseline",
    "Change_% = (current / pinned − 1) × 100",
    [
      v(
        "current",
        "Current completed-run value",
        current.metrics[key],
        "",
        () => metricCalculation(current, key),
      ),
      v("pinned", "Pinned completed-run value", pinned.metrics[key], "", () =>
        metricCalculation(pinned, key),
      ),
    ],
    (current.metrics[key] / pinned.metrics[key] - 1) * 100,
    "%",
  );
}

export function nativeMetricCalculation(r: Obj, key: string): Calculation {
  const m = r.metrics,
    n = r.requests.length;
  if (key === "output_tokens_per_second")
    return calc(
      "Native output throughput",
      "Throughput = N_output / (native_clocks_ns / 10⁹)",
      [
        v(
          "N_output",
          "Completed output tokens",
          m.output_tokens ??
            r.requests.reduce((sum: number, q: Obj) => sum + q.output, 0),
        ),
        v(
          "native_clocks_ns",
          "ASTRA simulator total clocks",
          m.native_total_clocks_ns,
          "ns",
        ),
      ],
      m.output_tokens_per_second,
      "tokens/s",
      "Native total-clock metric includes simulator control/idle intervals; it differs from request-span throughput.",
    );
  const field = key === "mean_ttft_ms" ? "TTFT" : "TPOT";
  return calc(
    "Native " + field,
    "Mean_ms = Σ request_" + field + "_ns / (N × 10⁶)",
    [
      v("N", "Native CSV requests", n),
      v(
        "Σ " + field,
        "Sum of request CSV values",
        r.requests.reduce((sum: number, q: Obj) => sum + q[field], 0),
        "ns",
      ),
    ],
    m[key],
    "ms",
    "Taken from the native request CSV. This aggregation has no fitted coefficient.",
  );
}

// Dependencies are lazy: opening one term builds only the requested explanation.
export const metricCalculation = (r: Obj, key: string): Calculation =>
  key === "elapsed_ms"
    ? horizonCalculation(r)
    : explainCalculation(r, metricBase(r, key), { metric: key });
export const memoryCalculation = (
  r: Obj,
  phase: string,
  key: string,
): Calculation =>
  explainCalculation(r, memoryBase(r, phase, key), { phase, memory: key });
export const operationCalculation = (
  r: Obj,
  phase: string,
  o: Obj,
  key: string,
): Calculation =>
  explainCalculation(r, operationBase(r, phase, o, key), {
    phase,
    op: o,
    operation: key,
  });
export const componentCalculation = (r: Obj, key: string): Calculation =>
  explainCalculation(r, componentBase(r, key), { component: key });
export const trafficCalculation = (r: Obj, key: string): Calculation =>
  explainCalculation(r, trafficBase(r, key), { traffic: key });
export const sramCalculation = (r: Obj, phase: string): Calculation =>
  explainCalculation(r, sramBase(r, phase), { phase });
