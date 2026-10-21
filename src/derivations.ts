/* Run-specific dependency graph. Every edge is lazy, and leaves identify provenance. */
import {
  type Calculation,
  calc,
  v,
  numberText,
  scaled,
  metricCalculation,
  memoryCalculation,
  operationCalculation,
  quantileCalculation,
} from "./calculations";
import {
  storageEventCalculation,
  storageRequestCalculation,
} from "./storageCalculations";
type Obj = Record<string, any>;
type Context = {
  metric?: string;
  phase?: string;
  op?: Obj;
  operation?: string;
  memory?: string;
  component?: string;
  traffic?: string;
};
const accuracy =
  "https://github.com/ryeii/LLM-ServeScope/blob/main/docs/accuracy-2026-10-19/README.md";
const sum = (a: Obj[], f: (x: Obj) => number) =>
  a.reduce((s, x) => s + f(x), 0);
const rows = (r: Obj, q: Obj, phase?: string) =>
  r.timeline.filter(
    (e: Obj) => e.request_ids.includes(q.id) && (!phase || e.phase === phase),
  );
const ms = (
  symbol: string,
  label: string,
  value: number,
  explain: () => Calculation,
) => v(symbol, label, value, "ms", explain);
function leaf(
  title: string,
  value: number | string,
  unit: string,
  source: string,
  summary: string,
  formula = "Value supplied by this source",
): Calculation {
  return {
    title,
    source,
    summary,
    formula,
    variables: [],
    result: value,
    unit,
  };
}
function input(
  r: Obj,
  key: string,
  label: string,
  value?: number | string,
  unit = "",
): Calculation {
  const descriptions: Obj = {
    requests:
      "Number of requests in this finite workload. Requests are numbered from 0, arrive at a fixed rate, and all must finish before the run ends.",
    output_tokens:
      "Fixed output length for every request. Prefill produces the first output token; the scheduler runs L_out − 1 subsequent decode iterations. No early stopping or sampled text changes this length.",
    input_tokens:
      "Fixed prompt length for every request. The prompt is processed in successive prefill chunks, with no cross-request prompt-length variation.",
    arrival_rate:
      "Deterministic arrival rate. Request i arrives at i / rate seconds; this is not a sampled Poisson arrival process.",
    prefill_chunk:
      "Maximum prompt tokens per sequence in each prefill work item. A final partial chunk keeps its actual size.",
    batch_size:
      "Maximum fixed prefill group size. A group waits for its last request to arrive. Decode may split this group into smaller batches, but groups never merge dynamically.",
    precision:
      "Weight/operator precision selects bytes per value and the precision-specific compute peak. FP16 uses the historical Vidur cohort; BF16 uses KAIST. Other precisions extrapolate.",
    kv_precision:
      "KV-cache precision selects bytes per key/value element. Conversion cost is not modeled.",
  };
  return leaf(
    label,
    value ?? r.config[key],
    unit,
    "Scenario input · completed-run snapshot",
    descriptions[key] ||
      `Resolved scenario setting ${key}. It is an input assumption, rather than an output inferred from the simulation.`,
  );
}
function modelInput(
  r: Obj,
  key: string,
  label: string,
  value?: number | string,
  unit = "",
): Calculation {
  const c = leaf(
    label,
    value ?? r.model[key],
    unit,
    "Model architecture · " + r.model.name,
    `This value comes from the resolved model geometry (${key}). Custom models use the geometry entered for this run; catalog models use the documented architecture source.`,
  );
  if (r.model.source_url)
    c.links = [{ label: "Architecture source", url: r.model.source_url }];
  return c;
}
function hardware(
  r: Obj,
  phase: string,
  key: string,
  label: string,
  value?: number | string,
  unit = "",
): Calculation {
  const h = r.config[phase].hardware;
  const assumed =
    [
      "compute_efficiency",
      "memory_efficiency",
      "clock_scale",
      "launch_overhead_us",
      "collective_allreduce_fraction",
      "collective_alltoall_fraction",
      "collective_alltoall_offset",
      "idle_watts",
      "tdp_watts",
      "hourly_cost",
      "sram_cache_MB",
    ].includes(key) || key.startsWith("pj_");
  return leaf(
    label,
    value ?? h[key],
    unit,
    `${phase} hardware · ${assumed ? "scenario assumption" : "configured hardware specification"}`,
    key === "clock_scale"
      ? "A multiplier on the precision-specific compute rate. It changes arithmetic features and their speed-limit floor; it does not multiply memory bandwidth. The effect on final throughput depends on fitted operator costs and scheduling waits."
      : key === "compute_efficiency" || key === "memory_efficiency"
        ? "A user-controlled efficiency assumption, not a separately measured stall fraction. Fitted features scale relative to calibration defaults (compute 0.65, memory 0.8); the learned function then predicts service time."
        : key === "sram_cache_MB"
          ? "Idealized reserved KV SRAM capacity. It affects the analytical KV traffic estimate; fitted GPU operator times already contain their historical kernel/cache behavior and do not model this hypothetical SRAM mechanistically."
          : `The resolved ${phase} hardware value used for this completed run. Hardware presets may be edited; this value is not itself a profiled latency. New hardware/software combinations require validation.`,
  );
}
function poolInput(
  r: Obj,
  phase: string,
  key: string,
  label: string,
  value?: number | string,
): Calculation {
  return leaf(
    label,
    value ?? r.config[phase][key],
    "",
    `${phase} server configuration`,
    (
      {
        tp: "Tensor parallelism shards applicable matrix/attention work and introduces explicit collectives; some normalization and sampling kernels remain replicated.",
        pp: "Transformer layers are divided uniformly over pipeline stages. Each stage has its own single-capacity FIFO resource; activation transfers connect consecutive stages.",
        ep: "Expert parallelism partitions expert work and introduces dispatch/combine communication. It is supported for modeled mixture-of-experts geometry.",
        replicas:
          "Batches are assigned round-robin to replicas. Replicas have independent stage resources and admission queues.",
        max_concurrent_batches:
          "Maximum admitted batches per replica. Admission is held across all prefill chunks or all decode tokens, while stage resources are acquired separately. It does not merge batches dynamically.",
        batch_size:
          "Maximum fixed phase batch size. Decode may split a prefill batch into smaller groups. No continuous cross-group rebatching is modeled.",
      } as Obj
    )[key] || `Resolved ${phase}.${key} input.`,
  );
}

export function horizonCalculation(r: Obj): Calculation {
  const last = r.requests.reduce((a: Obj, b: Obj) =>
    a.finish_ms >= b.finish_ms ? a : b,
  );
  const unique = [
    ...new Map(r.requests.map((q: Obj) => [q.finish_ms, q])).values(),
  ] as Obj[];
  return {
    ...calc(
      "Simulation horizon",
      "T_ms = maxᵢ finishᵢ; T_s = T_ms / 1000",
      [
        ms(
          "finish_last",
          `Last completion · request #${last.id}`,
          last.finish_ms,
          () => requestCalculation(r, last, "finish_ms"),
        ),
        ...unique
          .filter((q) => q.finish_ms !== last.finish_ms)
          .map((q) =>
            ms(
              `finish_${q.id}`,
              `Other batch completion · request #${q.id}`,
              q.finish_ms,
              () => requestCalculation(r, q, "finish_ms"),
            ),
          ),
      ],
      r.metrics.elapsed_ms,
      "ms",
    ),
    summary: `Request #${last.id} sets the horizon at ${numberText(last.finish_ms / 1000)} s. Follow its completion to see arrivals, batch formation, prefill, KV handoff, decode and queueing. Concurrent batches overlap, so adding every GPU interval would overcount elapsed time.`,
    source: "SimPy completion timestamps · actual run",
  };
}
export function requestCalculation(r: Obj, q: Obj, key: string): Calculation {
  if (q.cache) return storageRequestCalculation(r, q, key);
  const c = r.config;
  if (key === "arrival_ms")
    return {
      ...calc(
        `Request #${q.id} arrival`,
        "arrival_ms = request_id / arrival_rate × 1000",
        [
          v("request_id", "Zero-based request ID", q.id, "", () =>
            leaf(
              "Request ID",
              q.id,
              "",
              "Workload enumeration",
              "IDs are assigned sequentially from 0 to N_req − 1.",
            ),
          ),
          v(
            "arrival_rate",
            "Requests per second",
            c.arrival_rate,
            "requests/s",
            () => input(r, "arrival_rate", "Arrival rate"),
          ),
        ],
        q.arrival_ms,
        "ms",
      ),
      summary:
        "The simulator starts at 0 ms. This request's arrival is fixed by its position and the configured arrival rate.",
    };
  if (key === "batch_ready_ms") {
    const last = Math.min(
      c.requests - 1,
      (Math.floor(q.id / c.prefill.batch_size) + 1) * c.prefill.batch_size - 1,
    );
    return {
      ...calc(
        `Batch ${q.prefill_batch_id} readiness`,
        "batch_ready_ms = last_request_id / arrival_rate × 1000",
        [
          v(
            "last_request_id",
            "Last request in this fixed group",
            last,
            "",
            () =>
              leaf(
                "Batch membership",
                last,
                "",
                "Fixed-group scheduler",
                `Consecutive groups contain at most ${c.prefill.batch_size} requests. This group becomes ready only after request #${last} arrives. There is no batching timeout.`,
                `last_id = min(N_req−1, (floor(id/B_prefill)+1) B_prefill−1)`,
              ),
          ),
          v("arrival_rate", "Arrival rate", c.arrival_rate, "requests/s", () =>
            input(r, "arrival_rate", "Arrival rate"),
          ),
          v(
            "B_prefill",
            "Prefill batch limit",
            c.prefill.batch_size,
            "requests",
            () => poolInput(r, "prefill", "batch_size", "Prefill batch limit"),
          ),
        ],
        q.batch_ready_ms,
        "ms",
      ),
      summary:
        "Batch formation can delay early requests even if a GPU is idle.",
    };
  }
  if (key === "ttft_ms")
    return {
      ...calc(
        `Request #${q.id} time to first token`,
        "TTFT = first_token_ms − arrival_ms",
        [
          ms("first_token_ms", "First token release", q.first_token_ms, () =>
            requestCalculation(r, q, "first_token_ms"),
          ),
          ms("arrival_ms", "Request arrival", q.arrival_ms, () =>
            requestCalculation(r, q, "arrival_ms"),
          ),
        ],
        q.ttft_ms,
        "ms",
      ),
      summary:
        "Includes waiting to form a batch, admission/resource queueing, prefill and the KV handoff. This model releases the first token after handoff.",
    };
  if (key === "tpot_ms")
    return {
      ...calc(
        `Request #${q.id} time per output token`,
        "TPOT = decode_span / max(1, L_out − 1)",
        [
          ms(
            "decode_span",
            "Decode service and waits",
            q.finish_ms - q.first_token_ms,
            () => phaseSpan(r, q, "decode"),
          ),
          v(
            "L_out",
            "Output tokens per request",
            c.output_tokens,
            "tokens",
            () => input(r, "output_tokens", "Output tokens per request"),
          ),
        ],
        q.tpot_ms,
        "ms",
      ),
      summary:
        c.output_tokens === 1
          ? "No decode iteration is needed when output length is one; TPOT is zero."
          : "This is the mean of this request's subsequent token intervals. Each decode iteration must finish before its next iteration can start.",
    };
  if (key === "latency_ms")
    return {
      ...calc(
        `Request #${q.id} latency`,
        "Latency = finish_ms − arrival_ms",
        [
          ms("finish_ms", "Request completion", q.finish_ms, () =>
            requestCalculation(r, q, "finish_ms"),
          ),
          ms("arrival_ms", "Request arrival", q.arrival_ms, () =>
            requestCalculation(r, q, "arrival_ms"),
          ),
        ],
        q.latency_ms,
        "ms",
      ),
      summary: "End-to-end request time includes both service and waiting.",
    };
  const first = key === "first_token_ms";
  const arrival = q.arrival_ms,
    formation = q.batch_ready_ms - arrival;
  return {
    ...calc(
      `Request #${q.id} ${first ? "first token" : "completion"}`,
      first
        ? "first_token = arrival + formation_wait + prefill_span + handoff_span"
        : "finish = arrival + formation_wait + prefill_span + handoff_span + decode_span",
      [
        ms("arrival", "Request arrival", arrival, () =>
          requestCalculation(r, q, "arrival_ms"),
        ),
        ms("formation_wait", "Wait for batch formation", formation, () => ({
          ...calc(
            "Wait for batch formation",
            "formation_wait = batch_ready_ms − arrival_ms",
            [
              ms("batch_ready_ms", "Batch readiness", q.batch_ready_ms, () =>
                requestCalculation(r, q, "batch_ready_ms"),
              ),
              ms("arrival_ms", "Request arrival", arrival, () =>
                requestCalculation(r, q, "arrival_ms"),
              ),
            ],
            formation,
            "ms",
          ),
          summary:
            "The simulator groups consecutive requests; early members wait until the final member arrives.",
        })),
        ms(
          "prefill_span",
          "Prefill service and waits",
          q.prefill_end_ms - q.batch_ready_ms,
          () => phaseSpan(r, q, "prefill"),
        ),
        ms(
          "handoff_span",
          "KV handoff service and waits",
          q.first_token_ms - q.prefill_end_ms,
          () => phaseSpan(r, q, "transfer"),
        ),
        ...(!first
          ? [
              ms(
                "decode_span",
                "Decode service and waits",
                q.finish_ms - q.first_token_ms,
                () => phaseSpan(r, q, "decode"),
              ),
            ]
          : []),
      ],
      first ? q.first_token_ms : q.finish_ms,
      "ms",
    ),
    summary: `This is the actual elapsed path of request #${q.id}, not an average-batch estimate. Each span includes its own queues. Other requests can run concurrently on other resources.`,
    source: "Recorded request path · completed SimPy run",
  };
}
function phaseSpan(r: Obj, q: Obj, phase: string): Calculation {
  const events = rows(r, q, phase);
  const begin =
    phase === "prefill"
      ? q.batch_ready_ms
      : phase === "decode"
        ? q.first_token_ms
        : q.prefill_end_ms;
  const end =
    phase === "prefill"
      ? q.prefill_end_ms
      : phase === "decode"
        ? q.finish_ms
        : q.first_token_ms;
  const service = sum(events, (e) => e.duration_ms),
    total = end - begin,
    wait = Math.max(0, total - service);
  const phaseName = phase === "transfer" ? "KV handoff" : phase;
  return {
    ...calc(
      `Request #${q.id} · ${phaseName} span`,
      "Span = Σ service_intervals + Σ waiting_gaps",
      [
        ms("service", `${phaseName} service`, service, () => ({
          ...calc(
            `${phaseName} service intervals`,
            "Service = Σₑ durationₑ",
            events.map((e: Obj) =>
              ms(`event_${e.event_id}`, eventLabel(e), e.duration_ms, () =>
                eventService(r, e),
              ),
            ),
            service,
            "ms",
          ),
          summary: `${events.length} intervals on this request's serial path. Each GPU interval includes all layers assigned to that stage and applicable endpoints.`,
        })),
        ms("waiting", `${phaseName} waiting`, wait, () =>
          phaseWaits(r, q, phase, events, begin, wait),
        ),
        ...events
          .slice(-1)
          .map((e: Obj) =>
            ms(
              "last_interval",
              "Final interval in this phase",
              e.end_ms - e.start_ms,
              () => eventCalculation(r, e),
            ),
          ),
      ],
      total,
      "ms",
    ),
    summary:
      phase === "transfer"
        ? r.config.separated
          ? "A handoff channel is reserved after prefill. Payload/bandwidth plus latency determines service; a busy channel adds waiting."
          : "Prefill and decode are colocated, so no KV handoff is charged."
        : `${phase === "prefill" ? "Prompt chunks" : "Decode tokens"} run sequentially for this batch. Admission may delay its first work item; each GPU stage or pipeline link may then queue behind other work.`,
    source: "Request-specific event path",
  };
}
function phaseWaits(
  r: Obj,
  q: Obj,
  phase: string,
  events: Obj[],
  begin: number,
  total: number,
): Calculation {
  const admitted =
    phase === "prefill"
      ? q.prefill_admitted_ms
      : phase === "decode"
        ? q.decode_admitted_ms
        : begin;
  const admission = Math.max(0, (admitted ?? begin) - begin);
  const vars = [
    ...events
      .filter((e) => e.queue_ms > 1e-9)
      .map((e) =>
        ms(
          `queue_${e.event_id}`,
          `Queue before ${eventLabel(e)}`,
          e.queue_ms,
          () => queueCalculation(r, e),
        ),
      ),
  ];
  if (phase !== "transfer")
    vars.unshift(
      ms("admission_wait", "Wait for batch admission", admission, () => ({
        ...calc(
          `${phase} admission wait`,
          "Wait = admitted_ms − phase_ready_ms",
          [
            ms("phase_ready_ms", "Batch ready for this phase", begin, () =>
              requestCalculation(
                r,
                q,
                phase === "prefill" ? "batch_ready_ms" : "first_token_ms",
              ),
            ),
            v(
              "admitted_ms",
              "Admission granted timestamp",
              admitted ?? begin,
              "ms",
            ),
            v(
              "limit",
              "Admitted batch limit per replica",
              r.config[phase].max_concurrent_batches,
              "batches",
              () =>
                poolInput(
                  r,
                  phase,
                  "max_concurrent_batches",
                  "Admission limit",
                ),
            ),
            ...admissionBlockers(r, q, phase, begin, admitted ?? begin),
          ],
          admission,
          "ms",
        ),
        summary:
          "Admission is FIFO per replica. A permit is held through every chunk (prefill) or remaining output token (decode), then released. This wait is distinct from the single-capacity stage queue.",
      })),
    );
  return {
    ...calc(
      `${phase} queueing`,
      "Waiting = admission_wait + Σ resource_queue_wait",
      vars,
      total,
      "ms",
    ),
    summary:
      total < 1e-8
        ? "No queueing delay occurred on this request's phase path."
        : "These waits are measured from event readiness to resource grant in this run. Follow a resource wait to inspect the reservation that kept it busy.",
  };
}
function admissionBlockers(
  r: Obj,
  q: Obj,
  phase: string,
  begin: number,
  end: number,
) {
  if (end - begin < 1e-9) return [];
  const batchKey = phase === "prefill" ? "prefill_batch_id" : "decode_batch_id",
    replicas = r.config[phase].replicas;
  const unique = [
    ...new Map(r.requests.map((x: Obj) => [x[batchKey], x])).values(),
  ] as Obj[];
  return unique
    .filter(
      (x) =>
        x[batchKey] % replicas === q[batchKey] % replicas &&
        x[batchKey] !== q[batchKey] &&
        (phase === "prefill" ? x.prefill_admitted_ms : x.decode_admitted_ms) <=
          begin &&
        (phase === "prefill" ? x.prefill_end_ms : x.finish_ms) > begin,
    )
    .map((x) =>
      ms(
        `active_${x[batchKey]}`,
        `Admitted batch ${x[batchKey]} releases permit`,
        phase === "prefill" ? x.prefill_end_ms : x.finish_ms,
        () =>
          phase === "prefill"
            ? phaseSpan(r, x, "prefill")
            : requestCalculation(r, x, "finish_ms"),
      ),
    );
}
const eventLabel = (e: Obj) =>
  e.phase === "transfer"
    ? `KV handoff · batch ${e.batch_id}`
    : e.kind === "pipeline_transfer"
      ? `${e.phase} pipeline link · chunk ${e.chunk}`
      : `${e.phase} stage ${e.stage ?? ""} · ${e.phase === "decode" ? "token step" : "chunk"} ${e.chunk} · batch ${e.batch_id}`;
export function eventCalculation(r: Obj, e: Obj): Calculation {
  if (e.phase === "storage") return storageEventCalculation(r, e);
  return {
    ...calc(
      eventLabel(e),
      "end_ms = start_ms + service_ms",
      [
        ms("start_ms", "Resource grant time", e.start_ms, () =>
          startCalculation(r, e),
        ),
        ms("service_ms", "Interval service cost", e.duration_ms, () =>
          eventService(r, e),
        ),
      ],
      e.end_ms,
      "ms",
    ),
    summary: `Resource ${e.resource}. The scheduler waits until both this work item is ready and the previous reservation has released the resource.`,
    source: "Executed scheduling event · #" + e.event_id,
  };
}
function startCalculation(r: Obj, e: Obj): Calculation {
  const previous = r.timeline.find(
    (x: Obj) => x.event_id === e.previous_resource_event,
  );
  return {
    ...calc(
      "Resource grant time",
      "start_ms = max(ready_ms, previous_release_ms)",
      [
        ms("ready_ms", "Work item readiness", e.ready_ms ?? e.start_ms, () =>
          readyCalculation(r, e),
        ),
        ms(
          "previous_release_ms",
          "Previous reservation releases resource",
          previous?.end_ms ?? 0,
          () =>
            previous
              ? eventCalculation(r, previous)
              : leaf(
                  "Initially free resource",
                  0,
                  "ms",
                  "SimPy resource initialization",
                  "This resource has not been reserved earlier in this run; it is free at time zero.",
                ),
        ),
      ],
      e.start_ms,
      "ms",
    ),
    summary: `Single-capacity FIFO resource ${e.resource}. ${e.queue_ms > 1e-9 ? `The work item waited ${numberText(e.queue_ms)} ms.` : "The resource was available when this work item became ready."}`,
  };
}
function readyCalculation(r: Obj, e: Obj): Calculation {
  const ready = e.ready_ms ?? e.start_ms;
  const previous = r.timeline
    .filter(
      (x: Obj) =>
        x.event_id !== e.event_id &&
        x.end_ms <= ready &&
        x.request_ids.some((id: number) => e.request_ids.includes(id)),
    )
    .sort((a: Obj, b: Obj) => b.end_ms - a.end_ms)[0];
  const q = r.requests.find((x: Obj) => e.request_ids.includes(x.id));
  const base = previous?.end_ms ?? q.batch_ready_ms;
  const gap = Math.max(0, ready - base);
  return {
    ...calc(
      "Work item readiness",
      "ready_ms = predecessor_end_ms + admission_gap_ms",
      [
        ms(
          "predecessor_end_ms",
          previous
            ? "Previous work on this batch's path"
            : "Batch formation completed",
          base,
          () =>
            previous
              ? eventCalculation(r, previous)
              : requestCalculation(r, q, "batch_ready_ms"),
        ),
        v("admission_gap_ms", "Phase admission gap", gap, "ms", () =>
          leaf(
            "Phase admission gap",
            gap,
            "ms",
            "Recorded phase admission",
            gap > 1e-9
              ? "This phase could not enter its admission pool immediately. The phase waiting explanation lists the active batches and configured permit limit."
              : "No additional admission gap occurred before this interval.",
          ),
        ),
      ],
      ready,
      "ms",
    ),
    summary:
      "A batch's chunks, pipeline stages and decode steps have a serial dependency. The next item can request a resource only after its predecessor completes and any phase admission is granted.",
  };
}
function queueCalculation(r: Obj, e: Obj): Calculation {
  return {
    ...calc(
      "Resource queue wait",
      "queue_ms = start_ms − ready_ms",
      [
        ms("start_ms", "Resource grant time", e.start_ms, () =>
          startCalculation(r, e),
        ),
        ms("ready_ms", "Work item readiness", e.ready_ms ?? e.start_ms, () =>
          readyCalculation(r, e),
        ),
      ],
      e.queue_ms ?? 0,
      "ms",
    ),
    summary: `Other work owns or precedes this batch on ${e.resource}. This wait follows FIFO reservations; it is not a fitted GPU-kernel overhead.`,
  };
}
export function eventService(r: Obj, e: Obj): Calculation {
  if (e.phase === "storage") return storageEventCalculation(r, e);
  if (e.phase === "transfer" || e.kind === "pipeline_transfer")
    return {
      ...calc(
        e.phase === "transfer"
          ? "KV handoff service"
          : "Pipeline activation transfer",
        "T_ms = payload_bytes / (bandwidth_GBps × 10⁹) × 1000 + latency_us / 1000",
        [
          v(
            "payload_bytes",
            "Transfer payload",
            e.payload_bytes ?? 0,
            "bytes",
            () => transferPayload(r, e),
          ),
          v(
            "bandwidth_GBps",
            "Configured link bandwidth",
            e.link_bandwidth_GBps ?? 0,
            "GB/s",
            () => linkSource(r, e, "bw"),
          ),
          v(
            "latency_us",
            "Configured link latency",
            e.link_latency_us ?? 0,
            "µs",
            () => linkSource(r, e, "latency"),
          ),
        ],
        e.duration_ms,
        "ms",
      ),
      summary:
        "Analytical transfer service time. Channel/link queueing is charged separately; packet-level behavior and transport protocols are not simulated.",
    };
  const shape = r.execution_shapes?.[e.shape_id];
  if (!shape)
    return leaf(
      "Stage service cost",
      e.duration_ms,
      "ms",
      "Legacy run artifact",
      "This older export did not record executed operator shapes. Run the scenario again to inspect exact operator costs; representative table values are not substituted.",
    );
  const ops = shape.operations,
    layer = sum(ops, (o) => o.duration_ms),
    L = e.layers_per_stage;
  return {
    ...calc(
      `${e.phase} GPU stage service`,
      "Stage_ms = layers_per_stage × Σ operator_ms + Σ endpoint_ms + head_fallback_ms",
      [
        v("layers_per_stage", "Layers per pipeline stage", L, "layers", () => ({
          ...calc(
            "Layers per stage",
            "layers_per_stage = model_layers / PP",
            [
              v(
                "model_layers",
                "Transformer layers",
                r.model.layers,
                "layers",
                () => modelInput(r, "layers", "Transformer layers"),
              ),
              v("PP", "Pipeline parallelism", r.config[e.phase].pp, "", () =>
                poolInput(r, e.phase, "pp", "Pipeline stages"),
              ),
            ],
            L,
            "layers",
          ),
          summary:
            "Uniform layers are assigned evenly to stages. Endpoint kernels are charged separately.",
        })),
        ms("Σ operator_ms", "One transformer layer", layer, () => ({
          ...calc(
            "One transformer layer",
            "Layer_ms = Σ operator_ms",
            ops.map((o: Obj) =>
              ms(o.id, o.name, o.duration_ms, () =>
                operationCalculation(
                  r,
                  e.phase,
                  {
                    ...o,
                    batch: shape.batch,
                    tokens: shape.tokens,
                    context: shape.context,
                  },
                  "duration_ms",
                ),
              ),
            ),
            layer,
            "ms",
          ),
          summary: `Actual executed shape: batch ${shape.batch}, ${shape.tokens} query tokens per sequence, context ${shape.context}. Each operator uses its selected fitted function or disclosed analytical fallback.`,
        })),
        ...(e.endpoint_timing || []).map((o: Obj) =>
          ms(o.name, `Endpoint · ${o.name}`, o.duration_ms, () =>
            operationCalculation(r, e.phase, o, "duration_ms"),
          ),
        ),
        ...(e.head_fallback_ms
          ? [
              ms(
                "head_fallback_ms",
                "Analytical output head fallback",
                e.head_fallback_ms,
                () => headFallback(r, e, shape),
              ),
            ]
          : []),
      ],
      e.duration_ms,
      "ms",
    ),
    summary:
      "The stage reserves its GPU resource for this sum. Per-operator fitted times already include kernel memory/compute interactions. Separate nominal Compute/HBM columns are not added to these times.",
    source: "Executed operator shape · " + e.shape_id,
  };
}
function headFallback(r: Obj, e: Obj, s: Obj): Calculation {
  const p = r.config[e.phase],
    h = p.hardware,
    m = r.model,
    d = (
      { bf16: 2, fp16: 2, fp32: 4, tf32: 4, fp8: 1, int8: 1, fp4: 0.5 } as Obj
    )[r.config.precision];
  const F = (2 * s.batch * m.hidden * m.vocab) / p.tp,
    Q = (m.hidden * m.vocab * d) / p.tp;
  const peak =
    h[
      (r.config.precision === "fp32"
        ? "fp32"
        : r.config.precision + "_tensor") + "_tflops"
    ];
  const C = (F / (peak * 1e12 * h.compute_efficiency * h.clock_scale)) * 1000,
    M = (Q / (h.dram_bandwidth_TBps * 1e12 * h.memory_efficiency)) * 1000;
  return {
    ...calc(
      "Analytical output head",
      "T_head = max(C_ms, M_ms)",
      [
        v("C_ms", "Output-head arithmetic time", C, "ms", () =>
          leaf(
            "Output-head arithmetic",
            C,
            "ms",
            "Analytical fallback",
            `F = 2 × B × hidden × vocabulary / TP = ${numberText(F)} FLOP. Peak = ${peak} TFLOP/s, compute efficiency = ${h.compute_efficiency}, clock = ${h.clock_scale}.`,
            `C_ms = F / (peak × 10¹² × efficiency × clock) × 1000`,
          ),
        ),
        v("M_ms", "Output-head memory time", M, "ms", () =>
          leaf(
            "Output-head memory",
            M,
            "ms",
            "Analytical fallback",
            `Q = hidden × vocabulary × bytes / TP = ${numberText(Q)} bytes. Bandwidth = ${h.dram_bandwidth_TBps} TB/s; efficiency = ${h.memory_efficiency}.`,
            `M_ms = Q / (bandwidth × 10¹² × efficiency) × 1000`,
          ),
        ),
      ],
      e.head_fallback_ms,
      "ms",
    ),
    summary:
      "Used only when the scheduler has no fitted LM-head endpoint for this work item.",
  };
}
function transferPayload(r: Obj, e: Obj): Calculation {
  const c = r.config,
    m = r.model,
    kv = ({ bf16: 2, fp16: 2, fp8: 1 } as Obj)[c.kv_precision],
    weight = (
      { bf16: 2, fp16: 2, fp32: 4, tf32: 4, fp8: 1, int8: 1, fp4: 0.5 } as Obj
    )[c.precision];
  const pipeline = e.kind === "pipeline_transfer";
  return explainCalculation(
    r,
    {
      ...calc(
        pipeline ? "Pipeline activation payload" : "KV handoff payload",
        pipeline
          ? "Q = B × query_tokens × H × d_weight / TP"
          : "Q = 2 × B × L_in × H_KV × D_head × d_KV × L_layers",
        [
          v("B", "Actual batch size", e.batch_size),
          v(
            "query_tokens",
            "Query tokens per sequence",
            pipeline
              ? e.phase === "decode"
                ? 1
                : (e.tokens ?? Math.min(c.prefill_chunk, e.chunk))
              : 1,
          ),
          v("H", "Hidden width", m.hidden),
          v(
            "TP",
            "Tensor parallelism",
            c[e.phase === "decode" ? "decode" : "prefill"].tp,
          ),
          v("d_weight", "Weight bytes per element", weight),
          v("L_in", "Input tokens", c.input_tokens),
          v("H_KV", "KV heads", m.kv_heads),
          v("D_head", "Head dimension", m.head_dim),
          v("d_KV", "KV bytes per element", kv),
          v("L_layers", "Transformer layers", m.layers),
        ],
        e.payload_bytes,
        "bytes",
      ),
      summary: pipeline
        ? "Activations move once between each adjacent pipeline stage. This payload is sharded by TP."
        : "Both keys and values for the completed prompt are transferred for this batch across all layers. This model charges the unsharded logical KV payload once.",
    },
    { phase: e.phase === "decode" ? "decode" : "prefill" },
  );
}
function linkSource(r: Obj, e: Obj, key: string): Calculation {
  if (e.phase === "transfer")
    return leaf(
      key === "bw" ? "Handoff bandwidth" : "Handoff latency",
      key === "bw" ? e.link_bandwidth_GBps : e.link_latency_us,
      key === "bw" ? "GB/s" : "µs",
      "Network scenario input",
      `Configured spine ${key === "bw" ? "bandwidth" : "latency"} is used for KV handoff. Transfer channels are independent single-capacity FIFO resources; the model excludes packet routing and TCP/RDMA protocol effects.`,
    );
  return fabricCalculation(
    r,
    e.phase,
    key,
    e.link_bandwidth_GBps,
    e.link_latency_us,
    r.config[e.phase].tp * r.config[e.phase].ep * r.config[e.phase].pp,
  );
}
export function fabricCalculation(
  r: Obj,
  phase: string,
  key: string,
  bw?: number,
  latency?: number,
  groupOverride?: number,
): Calculation {
  const p = r.config[phase],
    h = p.hardware,
    g = groupOverride ?? p.tp * p.ep,
    n = r.config.network;
  const tier =
    g <= h.chip_cluster_size
      ? 0
      : g <= h.same_rail_gpus
        ? 1
        : g <= h.default_GPUs_per_rack
          ? 2
          : 3;
  const bws = [
    h.intra_node_bw_GBps,
    Math.min(h.intra_node_bw_GBps, h.same_rail_bw_GBps, h.scale_out_bw_GBps),
    Math.min(h.same_rail_bw_GBps, h.scale_out_bw_GBps),
    Math.min(h.scale_out_bw_GBps, n.spine_bw_GBps),
  ];
  const delays = [
    h.network_latency_us,
    h.network_latency_us * 2,
    h.network_latency_us * 3,
    n.spine_latency_us,
  ];
  return {
    ...calc(
      "Limiting network fabric",
      key === "bw"
        ? [
            "BW = intra_node_BW",
            "BW = min(intra_node_BW, same_rail_BW, scale_out_BW)",
            "BW = min(same_rail_BW, scale_out_BW)",
            "BW = min(scale_out_BW, spine_BW)",
          ][tier]
        : `latency = ${tier < 3 ? `${tier + 1} × GPU_network_latency` : "spine_latency"}`,
      [
        v("group", "GPU communication group", g, "GPUs", () =>
          poolInput(r, phase, "tp", "Communication group", g),
        ),
        ...[
          "chip_cluster_size",
          "same_rail_gpus",
          "default_GPUs_per_rack",
          "intra_node_bw_GBps",
          "same_rail_bw_GBps",
          "scale_out_bw_GBps",
          "network_latency_us",
        ].map((k) => v(k, k, h[k], "", () => hardware(r, phase, k, k))),
        v("spine_BW", "Spine bandwidth", n.spine_bw_GBps, "GB/s", () =>
          leaf(
            "Spine bandwidth",
            n.spine_bw_GBps,
            "GB/s",
            "Network scenario input",
            "User-configured inter-pool/spine bandwidth.",
          ),
        ),
        v("spine_latency", "Spine latency", n.spine_latency_us, "µs", () =>
          leaf(
            "Spine latency",
            n.spine_latency_us,
            "µs",
            "Network scenario input",
            "User-configured inter-pool/spine latency.",
          ),
        ),
      ],
      key === "bw" ? (bw ?? bws[tier]) : (latency ?? delays[tier]),
      key === "bw" ? "GB/s" : "µs",
    ),
    summary: `Selected tier: ${["intra-node", "same rail / TOR", "scale-out NIC", "spine / WAN"][tier]}. Group size is compared with nested domain capacities. This is a behavioral bottleneck-link approximation, not physical network emulation.`,
  };
}

export function ceilingCalculation(r: Obj, phase: string): Calculation {
  const h = r.config[phase].hardware;
  const key =
    r.config.precision === "fp32"
      ? "fp32_tflops"
      : r.config.precision + "_tensor_tflops";
  return {
    ...calc(
      "Nominal tensor ceiling",
      "Peak = precision_peak × clock_scale",
      [
        v("precision_peak", "Configured dense peak", h[key], "TFLOP/s", () =>
          hardware(r, phase, key, "Configured dense peak"),
        ),
        v("clock_scale", "Compute clock scale", h.clock_scale, "", () =>
          hardware(r, phase, "clock_scale", "Compute clock scale"),
        ),
      ],
      r.phases[phase].detail.peak_tflops,
      "TFLOP/s",
    ),
    summary:
      "The nominal rate is a hardware input scaled by the scenario clock multiplier. It is a feature of the service-time model; actual attained rate also depends on operator shape and fitted costs.",
  };
}

function fitFeature(r: Obj, phase: string, o: Obj, key: string): Calculation {
  const w = o.fit_work,
    h = r.config[phase].hardware,
    m = r.model,
    x = o.shape_features_ns;
  if (!w)
    return leaf(
      "Fitted feature",
      x?.[key === "C" ? 1 : 2] ?? 0,
      "ns",
      "Legacy run export",
      "Run the scenario again to record feature inputs. Missing provenance is not reconstructed from representative shapes.",
    );
  const F = v(
    "F_fit",
    "Arithmetic work for fitted function",
    w.flops,
    "FLOP",
    () => fitWork(r, phase, o, "flops"),
  );
  const Q = v(
    "Q_fit",
    "Nominal traffic for fitted function",
    w.traffic_bytes,
    "bytes",
    () => fitWork(r, phase, o, "traffic_bytes"),
  );
  const P = v("P", "Precision-specific peak", w.peak_tflops, "TFLOP/s", () =>
    hardware(
      r,
      phase,
      [
        "layernorm",
        "final_layernorm",
        "rotary_emb",
        "qk_norm",
        "act_fn",
        "sampler",
        "embedding",
      ].includes(o.fit_key)
        ? "fp32_tflops"
        : r.config.precision === "fp32"
          ? "fp32_tflops"
          : r.config.precision + "_tensor_tflops",
      "Compute peak",
      w.peak_tflops,
      "TFLOP/s",
    ),
  );
  const clock = v("clock", "Compute clock scale", h.clock_scale, "", () =>
    hardware(r, phase, "clock_scale", "Compute clock scale"),
  );
  if (key === "C" || key === "C_floor")
    return {
      ...calc(
        key === "C"
          ? "Nominal compute feature"
          : "Arithmetic speed-limit floor",
        key === "C"
          ? "C_ns = F_fit / (P × 1000 × clock) × 0.65 / η_compute"
          : "C_floor_ns = F_fit / (P × 1000 × clock)",
        [
          F,
          P,
          clock,
          ...(key === "C"
            ? [
                v(
                  "η_compute",
                  "Assumed compute efficiency",
                  h.compute_efficiency,
                  "",
                  () =>
                    hardware(
                      r,
                      phase,
                      "compute_efficiency",
                      "Compute efficiency",
                    ),
                ),
                v(
                  "0.65",
                  "Calibration reference compute efficiency",
                  0.65,
                  "",
                  () =>
                    leaf(
                      "Reference compute efficiency",
                      0.65,
                      "",
                      "Feature normalization constant",
                      "Scenario compute efficiency scales features relative to 0.65, the calibration default. This is a modeling convention; fitted coefficients supply empirical timing scale.",
                    ),
                ),
              ]
            : []),
        ],
        key === "C" ? x[1] : o.fit_compute_floor_ns,
        "ns",
      ),
      summary:
        key === "C"
          ? "Geometry supplies arithmetic work, hardware supplies nominal rate, and the selected fit learns how this feature scales to kernel service time. This is not the separate analytical Compute-column estimate."
          : "The fitted time is never allowed below the arithmetic work divided by the nominal hardware rate. This floor excludes the user efficiency multiplier.",
    };
  if (key === "M")
    return {
      ...calc(
        "Nominal memory feature",
        "M_ns = Q_fit / (BW_TBps × 1000) × 0.8 / η_memory",
        [
          Q,
          v(
            "BW_TBps",
            "Device-memory bandwidth",
            h.dram_bandwidth_TBps,
            "TB/s",
            () =>
              hardware(
                r,
                phase,
                "dram_bandwidth_TBps",
                "Device-memory bandwidth",
              ),
          ),
          v(
            "η_memory",
            "Assumed memory efficiency",
            h.memory_efficiency,
            "",
            () => hardware(r, phase, "memory_efficiency", "Memory efficiency"),
          ),
          v("0.8", "Calibration reference memory efficiency", 0.8, "", () =>
            leaf(
              "Reference memory efficiency",
              0.8,
              "",
              "Feature normalization constant",
              "Scenario memory efficiency scales features relative to 0.8. The actual fitted coefficient is learned from total kernel times; it is not an independently measured DRAM efficiency.",
            ),
          ),
        ],
        x[2],
        "ns",
      ),
      summary:
        "Nominal bytes divided by bandwidth forms an interpretable feature. Kernel measurements contain caching/stalls/overlap; the function learns their aggregate effect without claiming a measured decomposition.",
    };
  if (key === "M_correct")
    return {
      ...calc(
        "Corrected embedding memory feature",
        "M_correct_ns = n × (2 × d × H + 8) / (BW_TBps × 1000) × 0.8 / η_memory",
        [
          v("n", "Token items handled by kernel", w.n, "tokens", () =>
            kernelN(r, phase, o),
          ),
          v("d", "Bytes per weight value", w.dtype_bytes, "bytes", () =>
            input(r, "precision", "Weight precision", r.config.precision),
          ),
          v("H", "Hidden width", m.hidden, "", () =>
            modelInput(r, "hidden", "Hidden width"),
          ),
          v("8", "Embedding index bytes per token", 8, "bytes", () =>
            leaf(
              "Embedding index width",
              8,
              "bytes",
              "Corrected embedding traffic convention",
              "The F8/F9 embedding feature counts an 8-byte index plus one full hidden-vector read and write per token. Vocabulary sharding does not shard that hidden-vector payload.",
            ),
          ),
          v(
            "BW_TBps",
            "Device-memory bandwidth",
            h.dram_bandwidth_TBps,
            "TB/s",
            () =>
              hardware(
                r,
                phase,
                "dram_bandwidth_TBps",
                "Device-memory bandwidth",
              ),
          ),
          v(
            "η_memory",
            "Assumed memory efficiency",
            h.memory_efficiency,
            "",
            () => hardware(r, phase, "memory_efficiency", "Memory efficiency"),
          ),
        ],
        x[6],
        "ns",
      ),
      summary:
        "This feature corrects the historical embedding byte convention. It is the feature selected by the embedding fit, including its observed dispatch branch when F9 is used.",
    };
  if (key === "r")
    return {
      ...calc(
        "Working-set pressure on L2",
        "r = Q_fit / (L2_MiB × 1024²)",
        [
          Q,
          v("L2_MiB", "Physical L2 capacity", h.l2_cache_MiB, "MiB", () =>
            hardware(r, phase, "l2_cache_MiB", "Physical L2 capacity"),
          ),
        ],
        x[5],
        "ratio",
      ),
      summary:
        "A shape-informed proxy for cache pressure. It does not assert that the entire nominal working set resides in L2 or predict individual cache misses.",
    };
  const name = o.fit_key.split(":")[0],
    n = w.n,
    S = h.num_sms,
    H = m.hidden,
    I = m.intermediate,
    Qwidth = m.heads * m.head_dim,
    Kwidth = m.kv_heads * m.head_dim,
    tp = w.effective_tp;
  const widths: Obj = {
    qkv_proj: (Qwidth + 2 * Kwidth) / tp,
    o_proj: H,
    gate_up_proj: (2 * I) / tp,
    down_proj: H,
    lm_head: m.vocab / tp,
  };
  let width, blocks, padded, waves, formula, blockFormula;
  if (name in widths) {
    width = widths[name];
    blocks = Math.ceil(n / 128) * Math.ceil(width / 128);
    padded =
      ((Math.ceil(n / 128) * 128) / n) *
      ((Math.ceil(width / 128) * 128) / width);
    waves = (n * width) / (128 * 128 * S);
    formula = "W_ns = C_ns × padding × max(1, SMs / blocks)";
    blockFormula =
      "blocks = ceil(n/128) × ceil(width/128); padding = ceil(n/128)×128/n × ceil(width/128)×128/width";
  } else if (name === "attention") {
    width = m.heads / tp;
    const query =
        o.fit_cohort === "vidur" && phase === "prefill"
          ? o.fit_tokens
          : w.pc || 1,
      seq = w.pc ? w.pc / query : w.nd;
    blocks = seq * Math.ceil(query / 128) * width;
    padded = w.pc ? (Math.ceil(query / 128) * 128) / query : 1;
    waves = (seq * width * Math.max(1, query / 128)) / S;
    formula = "W_ns = C_ns × padding × max(1, SMs / max(1, blocks))";
    blockFormula = `blocks = sequences × ceil(query/128) × heads_per_shard; query=${query}; sequences=${seq}; padding=${numberText(padded)}`;
  } else {
    width =
      name === "act_fn"
        ? I / tp
        : ["rotary_emb", "qk_norm"].includes(name)
          ? (Qwidth + Kwidth) / tp
          : name === "sampler"
            ? m.vocab
            : H / (name === "embedding" ? tp : 1);
    blocks = Math.ceil((n * width) / 1024);
    padded = 1;
    waves = (n * width) / (1024 * S);
    formula = "W_ns = M_ns × max(1, SMs / blocks)";
    blockFormula = "blocks = ceil(n × width / 1024)";
  }
  return {
    ...calc(
      key === "W" ? "Padded tile-wave feature" : "Work tiles per SM",
      key === "W"
        ? formula
        : name in widths
          ? "w = n × width / (128 × 128 × SMs)"
          : name === "attention"
            ? "w = sequences × heads_per_shard × max(1, query/128) / SMs"
            : "w = n × width / (1024 × SMs)",
      [
        v("n", "Token items handled by kernel", n, "", () =>
          kernelN(r, phase, o),
        ),
        v("width", "Per-kernel feature width", width, "elements", () =>
          leaf(
            "Shape feature width",
            width,
            "elements",
            "Shape-informed feature definition",
            `Operator ${name}; effective TP ${tp}, hidden ${H}, intermediate ${I}, query width ${Qwidth}, KV width ${Kwidth}, vocabulary ${m.vocab}. Matrix outputs use their per-kernel width; o_proj/down_proj keep the full output hidden width.`,
            `width = ${name in widths ? ({ qkv_proj: "(query_width+2×KV_width)/TP", o_proj: "hidden", gate_up_proj: "2×intermediate/TP", down_proj: "hidden", lm_head: "vocab/TP" } as Obj)[name] : name === "attention" ? "heads/TP" : name === "act_fn" ? "intermediate/TP" : ["rotary_emb", "qk_norm"].includes(name) ? "(query_width+KV_width)/TP" : name === "sampler" ? "vocab" : name === "embedding" ? "hidden/TP" : "hidden"}`,
          ),
        ),
        v("SMs", "Streaming multiprocessors", S, "", () =>
          hardware(r, phase, "num_sms", "Streaming multiprocessors"),
        ),
        v("blocks", "Work blocks", blocks, "", () =>
          leaf(
            "Feature work blocks",
            blocks,
            "blocks",
            "Shape-informed feature definition",
            blockFormula +
              `. Token items=${n}, feature width=${width}. Tile constants are model features selected by grouped cross-validation, not measured CUDA launch metadata.`,
          ),
        ),
        v("padding", "Feature padding multiplier", padded, "", () =>
          leaf(
            "Tile padding multiplier",
            padded,
            "",
            "Shape-informed feature definition",
            blockFormula,
          ),
        ),
        v("C_ns", "Nominal compute feature", x[1], "ns", () =>
          fitFeature(r, phase, o, "C"),
        ),
        v("M_ns", "Nominal memory feature", x[2], "ns", () =>
          fitFeature(r, phase, o, "M"),
        ),
      ],
      key === "W" ? x[3] : waves,
      key === "W" ? "ns" : "tiles/SM",
    ),
    summary:
      "A compact shape proxy for underfilled/padded work. It was selected using profile validation; it is not a GPU instruction or cycle simulation.",
  };
}
function kernelN(r: Obj, phase: string, o: Obj): Calculation {
  const endpoint = ["lm_head", "sampler"].includes(o.fit_key);
  return {
    ...calc(
      "Token items handled by kernel",
      endpoint ? "n = B" : "n = B × query_tokens",
      [
        v("B", "Actual executed batch", o.fit_batch, "requests", () =>
          leaf(
            "Executed batch size",
            o.fit_batch,
            "requests",
            "Recorded work item",
            `This work item belongs to ${phase}. Prefill groups use the configured prefill batch limit; decode may split a group. A final partial group keeps its actual size.`,
          ),
        ),
        v(
          "query_tokens",
          "Query tokens per sequence",
          o.fit_tokens,
          "tokens",
          () =>
            leaf(
              "Executed query length",
              o.fit_tokens,
              "tokens",
              "Recorded chunk/token step",
              phase === "decode"
                ? "Each decode work item processes one new token per sequence."
                : `The prompt is processed in chunks of at most ${r.config.prefill_chunk} tokens per sequence. This item's actual context is ${o.fit_context}.`,
            ),
        ),
      ],
      o.fit_work.n,
      "items",
    ),
    summary: endpoint
      ? "The LM head and sampler act on one next-token position per sequence in this scheduler."
      : "Most kernels process batch size multiplied by the number of query tokens in this work item.",
  };
}
function fitWork(r: Obj, phase: string, o: Obj, key: string): Calculation {
  const w = o.fit_work,
    m = r.model,
    name = o.fit_key.split(":")[0],
    c = r.config;
  const f: Obj = {
    layernorm: "5*n*H",
    final_layernorm: "5*n*H",
    qkv_proj: "2*n*H*(Q+2*K)/TP_eff",
    qk_norm: "5*n*(Q+K)/TP_eff",
    rotary_emb: "6*n*Q/TP_eff",
    attention: "4*Q*[pc*(kp+0.5)+nd*(kv+1)]/TP_eff",
    o_proj: "2*n*Q*H/TP_eff",
    gate_up_proj: "4*n*H*I *top_k/TP_eff",
    act_fn: "5*n*I *top_k/TP_eff",
    down_proj: "2*n*H*I*top_k/TP_eff",
    embedding: "0",
    lm_head: "2*n*H*V/TP_eff",
    sampler: "5*n*V",
  };
  const q: Obj = {
    layernorm: "(2*n*H+H)*d",
    final_layernorm: "(2*n*H+H)*d",
    qkv_proj: "[H*(Q+2*K)/TP_eff+n*H+n*(Q+2*K)/TP_eff]*d",
    qk_norm: "[2*n*(Q+K)/TP_eff+2*D_head]*d",
    rotary_emb: "2*n*(Q+K)*d/TP_eff",
    attention: "[2*d*(pc+nd)*Q+2*d_KV*K*(pc+kp+nd*(kv+1))]/TP_eff",
    o_proj: "[Q*H/TP_eff+n*Q/TP_eff+n*H]*d",
    gate_up_proj: "[2*H*I *active/TP_eff+n*H+2*n*I *top_k/TP_eff]*d",
    act_fn: "3*n*I *top_k*d/TP_eff",
    down_proj: "[H*I*active/TP_eff+n*I*top_k/TP_eff+n*H]*d",
    embedding: "n*(2*H*d/TP_eff+4)",
    lm_head: "[H*V/TP_eff+n*(H+V/TP_eff)]*d",
    sampler: "4*n*V",
  };
  const relevant = new Set(
    (key === "flops" ? f[name] : q[name]).match(/[A-Za-z_]+/g),
  );
  const vars = [
    v("n", "Token items handled by kernel", w.n, "", () =>
      kernelN(r, phase, o),
    ),
    v("H", "Hidden width", m.hidden, "", () =>
      modelInput(r, "hidden", "Hidden width"),
    ),
    v("I", "Intermediate width", m.intermediate, "", () =>
      modelInput(r, "intermediate", "Intermediate width"),
    ),
    v(
      "Q",
      "Query heads × head dimension",
      m.heads * m.head_dim,
      "elements",
      () => headWidth(r, "heads"),
    ),
    v(
      "K",
      "KV heads × head dimension",
      m.kv_heads * m.head_dim,
      "elements",
      () => headWidth(r, "kv_heads"),
    ),
    v("V", "Vocabulary size", m.vocab, "", () =>
      modelInput(r, "vocab", "Vocabulary size"),
    ),
    v("D_head", "Head dimension", m.head_dim, "", () =>
      modelInput(r, "head_dim", "Head dimension"),
    ),
    v("TP_eff", "Effective sharding degree", w.effective_tp, "", () =>
      leaf(
        "Effective kernel sharding",
        w.effective_tp,
        "",
        "Kernel mapping rule",
        `Server TP=${r.config[phase].tp}, EP=${r.config[phase].ep}. MLP operations use TP×EP. Norm/final-norm/sampler remain replicated (TP_eff=1); other kernels use TP. This mapping is an explicit behavioral assumption.`,
      ),
    ),
    v("top_k", "Experts per token", m.top_k, "", () =>
      modelInput(r, "top_k", "Experts per token"),
    ),
    v("active", "Expected experts touched", w.active_experts, "", () => ({
      ...calc(
        "Expected experts touched",
        m.experts === 1 ? "active = 1" : "active = E × [1 − (1 − top_k / E)ⁿ]",
        [
          v("E", "Total experts", m.experts, "", () =>
            modelInput(r, "experts", "Total experts"),
          ),
          v("top_k", "Experts per token", m.top_k, "", () =>
            modelInput(r, "top_k", "Experts per token"),
          ),
          v("n", "Token items", w.n, "", () => kernelN(r, phase, o)),
        ],
        w.active_experts,
        "experts",
      ),
      summary:
        "Uniform independent expert selection approximates the number of distinct weight sets read. Actual routing imbalance/fused MoE execution requires separate validation.",
    })),
    v("d", "Bytes per weight value", w.dtype_bytes, "bytes", () =>
      input(r, "precision", "Operator precision", c.precision),
    ),
    v(
      "d_KV",
      "Bytes per KV value",
      ({ bf16: 2, fp16: 2, fp8: 1 } as Obj)[c.kv_precision],
      "bytes",
      () => input(r, "kv_precision", "KV precision", c.kv_precision),
    ),
    ...["pc", "kp", "nd", "kv"].map((k) =>
      v(
        k,
        (
          {
            pc: "Prefill query-token items",
            kp: "Prefill context-position proxy",
            nd: "Decode sequences",
            kv: "Prior decode context",
          } as Obj
        )[k],
        w[k],
        "tokens",
        () =>
          leaf(
            "Attention shape · " + k,
            w[k],
            "tokens",
            "Executed attention shape mapping",
            `Batch=${o.fit_batch}, query tokens=${o.fit_tokens}, context=${o.fit_context}. ${phase === "prefill" ? "pc=B×query_tokens; kp=max(0,context−query_tokens)+query_tokens/2; nd=kv=0." : "pc=kp=0; nd=B; kv=max(0,context−1)."} This is the same mapping used by the fitted timing feature.`,
          ),
      ),
    ),
  ].filter((row) => relevant.has(row.symbol));
  return {
    ...calc(
      key === "flops"
        ? "Fitted-function arithmetic work"
        : "Fitted-function nominal traffic",
      `${key === "flops" ? "F_fit" : "Q_fit"} = ${(key === "flops" ? f : q)[name].replaceAll("*", " × ")}`,
      vars,
      w[key],
      key === "flops" ? "FLOP" : "bytes",
    ),
    summary:
      "These are the geometry-based features used by the production fit. They deliberately retain the profiling convention, which can differ from the separate analytical resource-accounting columns. Nominal bytes are not measured memory-controller traffic.",
    source: "Production feature equation · " + name,
  };
}
function headWidth(r: Obj, key: string): Calculation {
  return {
    ...calc(
      key === "heads" ? "Query vector width" : "KV vector width",
      "width = heads × head_dim",
      [
        v("heads", "Head count", r.model[key], "", () =>
          modelInput(r, key, "Head count"),
        ),
        v("head_dim", "Head dimension", r.model.head_dim, "", () =>
          modelInput(r, "head_dim", "Head dimension"),
        ),
      ],
      r.model[key] * r.model.head_dim,
      "elements",
    ),
    summary:
      "Model architecture determines the vector width before parallel sharding.",
  };
}
function fittedSource(
  o: Obj,
  row: Calculation["variables"][number],
): Calculation {
  const c = leaf(
    row.label,
    row.value,
    row.unit || "",
    `Profile-fitted parameter · ${o.fit_cohort?.toUpperCase()} · ${o.fit_key} · ${o.fit_family}`,
    `This value is learned from ${o.fit_cohort === "vidur" ? "Vidur FP16" : "KAIST BF16"} operator timing profiles. Grouped nested cross-validation selects a simple function family; the deployed coefficients are refitted on all compatible observations in that cohort. This parameter is shared across GPU/model identities. It is not a new measurement of this GPU or a separately measured physical stall coefficient.`,
    "Parameter from the full-data fitted model",
  );
  c.links = [
    {
      label: "Fit selection, held-out accuracy and data sources",
      url: accuracy,
    },
  ];
  return c;
}

/* Attach interpretable dependencies to the accounting cards used throughout the UI. */
export function explainCalculation(
  r: Obj,
  c: Calculation,
  ctx: Context = {},
): Calculation {
  return {
    ...c,
    summary: c.summary || overview(c, ctx),
    variables: c.variables.map((row) => ({
      ...row,
      explain: row.explain || (() => term(r, c, row, ctx)),
    })),
  };
}
function overview(c: Calculation, ctx: Context): string {
  if (ctx.metric === "throughput_tps")
    return "Generation throughput depends on the total completion horizon. Follow Simulation horizon to the last-finishing request, then through its queues, stage costs and fitted hardware-dependent predictions.";
  if (ctx.op)
    return "Follow a feature to its arithmetic/byte counts and hardware rates, or a coefficient to its profiling source. Fitted service time and separate nominal resource demands have different meanings.";
  if (ctx.memory)
    return "This capacity estimate comes from model geometry, workload residency and server sharding. Follow a term to its formula or configured source.";
  return "Follow the marked terms to see their derivation or original input. All values belong to the completed run shown on the page.";
}
function term(
  r: Obj,
  c: Calculation,
  row: Calculation["variables"][number],
  ctx: Context,
): Calculation {
  const s = row.symbol,
    label = row.label,
    unit = row.unit || "",
    value = row.value,
    phase = ctx.phase || "prefill",
    o = ctx.op,
    h = r.config[phase].hardware;
  if (o) {
    if (
      ["C", "M", "W", "w", "r", "M_correct", "C_floor"].includes(s) &&
      o.fit_work
    )
      return fitFeature(r, phase, o, s);
    if (
      [
        "L",
        "a",
        "b",
        "b₀",
        "b₁",
        "a₀",
        "a₁",
        "d",
        "L_branch",
        "b_branch",
        "s",
        "threshold",
      ].includes(s) &&
      o.fit_parameters
    )
      return fittedSource(o, row);
    if (s === "T_base") {
      const parent = operationCalculation(r, phase, o, "duration_ms");
      return {
        ...parent,
        title: "Fitted base service time",
        formula: parent.formula.split("\n")[0],
        variables: parent.variables.filter(
          (x) => !["T_base", "κ_engine; Δ_engine; L_extra"].includes(x.symbol),
        ),
        result: o.fit_base_ns,
        unit: "ns",
        summary:
          "Evaluate the selected shape-informed function, then enforce a 1 ns minimum and the physical arithmetic floor. Engine/extra-launch adjustments are applied after this base value.",
      };
    }
    if (s === "κ_engine; Δ_engine; L_extra")
      return {
        ...calc(
          "Engine and launch adjustments",
          "κ = 0.97 / engine_efficiency; Δ_ns = (engine_launch_us − 2.5) × 1000; L_extra_ns = configured_launch_us × 1000",
          [
            v("κ", "Engine multiplier", o.fit_engine_multiplier),
            v("Δ_ns", "Engine offset", o.fit_engine_offset_ns, "ns"),
            v(
              "L_extra_ns",
              "Additional launch offset",
              o.fit_extra_launch_ns,
              "ns",
              () =>
                hardware(
                  r,
                  phase,
                  "launch_overhead_us",
                  "Additional launch overhead",
                ),
            ),
          ],
          o.fit_engine_multiplier,
          "multiplier",
        ),
        summary:
          "Explicit scenario assumptions relative to the calibration policy: vLLM (0.97, 2.5 µs), TensorRT-LLM (1, 2 µs), eager (0.8, 8 µs). These are not fitted or measured engine-version speedups. The final arithmetic floor is enforced again.",
        source: "Engine policy constants · disclosed assumption",
      };
    if (s === "branch")
      return leaf(
        label,
        value,
        unit,
        "Observed embedding dispatch boundary",
        `Select small_index if n ≤ ${o.fit_parameters.threshold_tokens}; otherwise select gather. Actual n=${o.fit_token_count}. Each branch uses its own fitted intercept and memory slope. The boundary reflects two observed profile regimes, not simulated CUDA dispatch instructions.`,
      );
    if (s === "g" || s === "g_cache")
      return {
        ...calc(
          label,
          s === "g" ? "g = w / (w+s)" : "g_cache = r / (r+s)",
          [
            v(
              s === "g" ? "w" : "r",
              s === "g" ? "Work tiles per SM" : "Working set / L2",
              o.shape_features_ns[s === "g" ? 4 : 5],
              "",
              () => fitFeature(r, phase, o, s === "g" ? "w" : "r"),
            ),
            v("s", "Fitted transition scale", o.fit_parameters.scale, "", () =>
              fittedSource(
                o,
                v("s", "Fitted transition scale", o.fit_parameters.scale),
              ),
            ),
          ],
          Number(value),
          "fraction",
        ),
        summary:
          "A smooth bounded transition between small/large work or cache-pressure regimes. The transition scale is learned from profile timings; no event-by-event cache behavior is implied.",
      };
    const operationKey: Obj = {
      F: "flops",
      Q: "hbm_bytes",
      T_ms: "duration_ms",
      C_ms: "compute_ms",
      M_ms: "memory_ms",
    };
    if (s in operationKey && !(ctx.operation === operationKey[s]))
      return operationCalculation(r, phase, o, operationKey[s]);
    if (s === "BW_GBps" || s === "latency_us")
      return fabricCalculation(r, phase, s === "BW_GBps" ? "bw" : "latency");
    if (s === "Q_network") {
      const reduce = o.name.includes("all-reduce"),
        g = o.name.startsWith("EP") ? r.config[phase].ep : r.config[phase].tp;
      return {
        ...calc(
          "Collective communication volume",
          reduce
            ? "Q_network = B × t × H × d × 2(g−1)/g"
            : "Q_network = B × t × H × d × top_k / TP × (g−1)/g",
          [
            v(
              "B",
              "Executed batch",
              o.batch ??
                o.fit_batch ??
                Math.min(r.config[phase].batch_size, r.config.requests),
            ),
            v(
              "t",
              "Query tokens per sequence",
              o.tokens ??
                o.fit_tokens ??
                (phase === "decode"
                  ? 1
                  : Math.min(r.config.prefill_chunk, r.config.input_tokens)),
            ),
            v("H", "Hidden width", r.model.hidden, "", () =>
              modelInput(r, "hidden", "Hidden width"),
            ),
            v(
              "d",
              "Bytes per weight value",
              (
                {
                  bf16: 2,
                  fp16: 2,
                  fp32: 4,
                  tf32: 4,
                  fp8: 1,
                  int8: 1,
                  fp4: 0.5,
                } as Obj
              )[r.config.precision],
            ),
            v("g", "Collective group", g),
            v("TP", "Tensor parallelism", r.config[phase].tp),
            v("top_k", "Experts per token", r.model.top_k),
          ],
          Number(value),
          "bytes",
        ),
        summary: reduce
          ? "Ring all-reduce moves two passes of (g−1)/g of the logical activation payload per rank."
          : "Expert dispatch/combine uses a uniform-routing payload and the (g−1)/g all-to-all fraction.",
      };
    }
    if (["phases", "group", "offset_us", "η_collective"].includes(s))
      return leaf(
        label,
        value,
        unit,
        "Analytical collective model",
        s === "phases"
          ? "Ring all-reduce uses 2(g−1) phases; all-to-all uses g−1. The per-phase link latency is charged explicitly."
          : s === "η_collective"
            ? "Configured collective bandwidth efficiency, not a coefficient fitted to KAIST/Vidur GPU-kernel profiles."
            : "Group size and dispatch overhead are derived from phase parallelism and its configured network assumptions.",
      );
    if (["weight_bytes", "activation_bytes", "kv_hbm_bytes"].includes(s))
      return analyticalBytes(r, phase, o, s);
    if (s === "u_tile")
      return leaf(
        label,
        value,
        unit,
        "Analytical resource-accounting heuristic",
        `Vector kernels use 1. Matrix kernels use min(1,max(0.08,tiles/(2×SMs))), where tiles=max(1,ceil(B×t/64))×ceil(output_width/(TP×tensor_width)). Actual SMs=${h.num_sms}, tensor width=${h.tensor_width}. This affects the nominal Compute column, not the fitted feature C or fitted latency.`,
      );
    if (s === "η_engine")
      return leaf(
        label,
        value,
        unit,
        "Engine policy assumption",
        "Nominal compute demand uses vLLM=0.97, TensorRT-LLM=1, eager=0.8. These are disclosed modeling assumptions, not separately profiled engine efficiencies.",
      );
  }
  if (s === "query_tokens")
    return leaf(
      label,
      value,
      unit,
      "Executed work-item geometry",
      `Each decode item uses one query token per sequence; prefill items contain at most the configured chunk size (${r.config.prefill_chunk}). A final partial chunk keeps its actual size.`,
    );
  if (s === "component")
    return leaf(
      label,
      value,
      unit,
      "Analytical resource accounting category",
      "Compute and Memory are nominal work/bandwidth demands; Network is modeled communication; Launch is an engine/hardware overhead assumption. These resource demands overlap and are not measured pieces of total latency.",
    );
  if (s === "base_result")
    return {
      ...c,
      title: "Before unit conversion",
      formula: c.formula.split("\nDisplayed value")[0],
      variables: c.variables.filter(
        (x) => !["base_result", "display_scale"].includes(x.symbol),
      ),
      result: Number(value),
      unit,
    };
  if (s === "reserved_bytes_per_layer")
    return explainCalculation(
      r,
      {
        ...calc(
          "Reserved KV SRAM per layer",
          "Reserved bytes = SRAM_MB × 10⁶ / (L_layers / PP)",
          [
            v("SRAM_MB", "Reserved KV SRAM", h.sram_cache_MB, "MB", () =>
              hardware(r, phase, "sram_cache_MB", "Reserved KV SRAM"),
            ),
            v("L_layers", "Transformer layers", r.model.layers),
            v("PP", "Pipeline parallelism", r.config[phase].pp),
          ],
          value,
          unit,
        ),
        summary:
          "The configured ideal KV reuse budget is divided equally among the layers on this stage. This is a scenario assumption, not a measured cache allocation.",
      },
      { phase },
    );
  if (s === "KV_read_bytes") {
    const attention = r.layers.find(
      (x: Obj) => x.phase === phase && x.name === "Attention",
    );
    return explainCalculation(
      r,
      {
        ...calc(
          "Representative KV read payload",
          "KV_read = 2 × B × context × max(1, H_KV / TP) × D_head × d_KV",
          [
            v("B", "Representative batch", attention.batch),
            v("context", "Representative context", attention.context),
            v("H_KV", "KV heads", r.model.kv_heads),
            v("TP", "Tensor parallelism", r.config[phase].tp),
            v("D_head", "Head dimension", r.model.head_dim),
            v(
              "d_KV",
              "KV bytes per value",
              ({ bf16: 2, fp16: 2, fp8: 1 } as Obj)[r.config.kv_precision],
            ),
          ],
          value,
          unit,
        ),
        summary:
          "This reuse indicator uses the phase's representative shape. Executed token/chunk shapes can have different context lengths; fitted operator time is not decomposed by this ideal SRAM assumption.",
      },
      { phase },
    );
  }
  if (s === "max_latency")
    return quantileCalculation(r.requests, "latency_ms", 1, r);
  if (s === "display_scale")
    return leaf(
      label,
      value,
      unit,
      "Unit conversion",
      "Only units change; the simulation value does not. Seconds↔milliseconds uses 1000, milliseconds↔microseconds uses 1000, decimal bytes↔GB uses 10⁹.",
    );
  if (["T_s", "T_ms"].includes(s) && !o)
    return s === "T_s"
      ? scaled(horizonCalculation(r), 0.001, "s")
      : horizonCalculation(r);
  if (["N_req", "N"].includes(s))
    return input(r, "requests", label, Number(value), unit);
  if (s === "L_out") return input(r, "output_tokens", label, undefined, unit);
  if (["L_in"].includes(s))
    return input(r, "input_tokens", label, undefined, unit);
  if (s === "N_req × L_out")
    return {
      ...calc(
        "Completed output tokens",
        "N_output = N_req × L_out",
        [
          v("N_req", "Completed requests", r.config.requests, "", () =>
            input(r, "requests", "Completed requests"),
          ),
          v(
            "L_out",
            "Output tokens per request",
            r.config.output_tokens,
            "",
            () => input(r, "output_tokens", "Output tokens per request"),
          ),
        ],
        Number(value),
        "tokens",
      ),
      summary:
        "All requests use the same fixed generation length; no early stopping is modeled.",
    };
  if (s === "E") return metricCalculation(r, "energy_j");
  if (s === "G" && !ctx.memory) return metricCalculation(r, "gpu_count");
  if (s === "Cost" && ctx.metric !== "cost_usd")
    return metricCalculation(r, "cost_usd");
  if (s === "Cost")
    return leaf(
      label,
      value,
      unit,
      "Redundant aggregate",
      "This is the same total given by the current pool-cost formula. Each pool's GPU count, price and simulation horizon are the independent inputs.",
    );
  if (s === "Σ durations" || s === "a[lo]" || s === "a[hi]") {
    const k = ctx.metric?.startsWith("tpot")
      ? "tpot_ms"
      : ctx.metric?.startsWith("ttft")
        ? "ttft_ms"
        : "latency_ms";
    const req = s.startsWith("a[")
      ? r.requests.filter((q: Obj) => q[k] === value)
      : r.requests;
    return {
      ...calc(
        s.startsWith("a[")
          ? "Ordered request observation"
          : "Per-request timing observations",
        s.startsWith("a[")
          ? "Selected sorted request value"
          : "Sum = Σ request_value",
        req.map((q: Obj) =>
          ms(`request_${q.id}`, `Request #${q.id} ${k}`, q[k], () =>
            requestCalculation(r, q, k),
          ),
        ),
        Number(value),
        "ms",
      ),
      summary:
        "Each observation is derived from this request's own arrival, first-token and completion path. Requests in a fixed batch can share completion times while having different arrival times.",
    };
  }
  const modelMap: Obj = {
    H: "hidden",
    I: "intermediate",
    H_KV: "kv_heads",
    D_head: "head_dim",
    L_layers: "layers",
    L: "layers",
    top_k: "top_k",
    experts: "experts",
  };
  if (s in modelMap)
    return modelInput(r, modelMap[s], label, Number(value), unit);
  if (s === "Q_heads" || s === "K_heads")
    return headWidth(r, s === "Q_heads" ? "heads" : "kv_heads");
  const hardwareMap: Obj = {
    BW_TBps: "dram_bandwidth_TBps",
    P_TFLOPS:
      r.config.precision === "fp32"
        ? "fp32_tflops"
        : r.config.precision + "_tensor_tflops",
    clock: "clock_scale",
    η_compute: "compute_efficiency",
    η_memory: "memory_efficiency",
    physical_GB: "main_memory_GB",
  };
  if (s in hardwareMap)
    return hardware(r, phase, hardwareMap[s], label, Number(value), unit);
  const poolMap: Obj = {
    TP: "tp",
    PP: "pp",
    EP: "ep",
    replicas: "replicas",
    reserved_GB: "runtime_reserved_GB",
    utilization: "memory_utilization",
  };
  if (s in poolMap)
    return poolInput(r, phase, poolMap[s], label, Number(value));
  if (
    ["weights", "KV", "activation", "capacity_GB", "total_GB"].includes(s) &&
    ctx.memory
  )
    return memoryCalculation(
      r,
      phase,
      (
        {
          weights: "weights_GB",
          KV: "kv_GB",
          activation: "activation_GB",
          capacity_GB: "capacity_GB",
          total_GB: "total_GB",
        } as Obj
      )[s],
    );
  if (s === "P_shared" || s === "P_expert")
    return parameterCount(r, s === "P_shared");
  if (s === "L_context")
    return explainCalculation(
      r,
      calc(
        "Maximum resident context",
        "L_context = L_in + L_out − 1",
        [
          v("L_in", "Input tokens", r.config.input_tokens),
          v("L_out", "Output tokens", r.config.output_tokens),
        ],
        Number(value),
        "tokens",
      ),
    );
  if (s === "N_resident") return residentCount(r, phase);
  if (s === "d_weight" || s === "d_KV")
    return leaf(
      label,
      value,
      unit,
      "Precision byte convention",
      `FP4=0.5, FP8/INT8=1, BF16/FP16=2, TF32/FP32=4 bytes per value. Weight precision=${r.config.precision}; KV precision=${r.config.kv_precision}. Metadata, alignment and quantization-conversion overhead are excluded.`,
    );
  if (s === "k_workspace")
    return leaf(
      label,
      value,
      unit,
      "Conservative workspace assumption",
      "Activation residency uses eight hidden-vector-sized buffers per batch/query position. This is a compact behavioral estimate, not a profiled allocator peak.",
    );
  if (s === "B" && ctx.metric === "gpu_utilization") return busyCalculation(r);
  if (s === "n" && o)
    return {
      ...calc(
        "Token items",
        "n = B × t",
        [
          v(
            "B",
            "Executed batch",
            o.batch ??
              o.fit_batch ??
              Math.min(r.config[phase].batch_size, r.config.requests),
          ),
          v(
            "t",
            "Query tokens per sequence",
            o.tokens ??
              o.fit_tokens ??
              (phase === "decode"
                ? 1
                : Math.min(r.config.prefill_chunk, r.config.input_tokens)),
          ),
        ],
        Number(value),
        "items",
      ),
      source: "Executed work-item geometry",
      summary: "Actual batch size multiplied by query tokens per sequence.",
    };
  if (s === "B" || s === "t" || s === "n" || s === "context")
    return leaf(
      label,
      value,
      unit,
      "Executed or representative operator geometry",
      `The operator table shows a representative phase shape; timeline drill-down uses the actual executed batch, chunk and context. Prefill chunk limit=${r.config.prefill_chunk}, phase batch limit=${r.config[phase].batch_size}, output length=${r.config.output_tokens}.`,
      s === "n" ? "n = B × t" : "Recorded work-item geometry",
    );
  if (s === "Q_prefill" || s === "Q_decode")
    return trafficPhase(
      r,
      s === "Q_prefill" ? "prefill" : "decode",
      ctx.traffic!,
    );
  if (ctx.component && s === "Σ contribution")
    return componentIntervals(r, ctx.component);
  if (s === "B" && ctx.metric === "gpu_utilization") return busyCalculation(r);
  if (s.includes(" ")) {
    const [ph, field] = s.split(" ");
    if (["prefill", "decode"].includes(ph)) {
      if (["TP", "PP", "EP", "replicas"].includes(field))
        return poolInput(r, ph, field.toLowerCase(), label, Number(value));
      if (field === "G") return memoryCalculation(r, ph, "gpus");
      if (field === "price")
        return hardware(r, ph, "hourly_cost", label, Number(value), unit);
      if (field === "B_s")
        return scaled(busyCalculation(r, ph), 0.001, "GPU·s");
      if (field === "E_dynamic") return dynamicEnergy(r, ph);
      if (field === "P_idle" || field === "P_TDP")
        return hardware(
          r,
          ph,
          field === "P_idle" ? "idle_watts" : "tdp_watts",
          label,
          Number(value),
          unit,
        );
      if (field === "P_active") return activePower(r, ph);
      if (field === "a")
        return leaf(
          label,
          value,
          unit,
          "Activity power assumption",
          "The modeled active power uses idle + 0.85×(TDP−idle), capped at TDP. This coefficient is an assumption, not a measurement from KAIST/Vidur timings.",
        );
      if (field.startsWith("pj_"))
        return hardware(r, ph, field, label, Number(value), unit);
      if (
        ["tensor_flops", "vector_flops", "hbm_bytes", "network_bytes"].includes(
          field,
        )
      )
        return trafficPhase(r, ph, field);
    }
  }
  if (["p", "x"].includes(s))
    return leaf(
      label,
      value,
      unit,
      "Quantile convention",
      s === "p"
        ? "P95 uses p=0.95. It describes the finite simulated request set, not a statistical confidence interval."
        : `The ordered fractional index is (N−1)×p. For this run N=${r.requests.length}; linear interpolation is applied between floor(x) and ceil(x).`,
    );
  return leaf(
    label,
    value,
    unit,
    "Completed-run accounting / model convention",
    c.note ||
      `This is an intermediate value of ${c.title}. The equation in its parent states how it is used; it is not a separate hardware measurement.`,
  );
}
function parameterCount(r: Obj, shared: boolean): Calculation {
  const m = r.model,
    H = m.hidden,
    Q = m.heads * m.head_dim,
    K = m.kv_heads * m.head_dim;
  return explainCalculation(r, {
    ...calc(
      shared ? "Shared parameters" : "Expert parameters",
      shared
        ? "P_shared = L[H(Q+2K)+QH+2H] + embedding_copies×H×V + H + qk_norm_extra + router_extra"
        : "P_expert = L × 3 × H × I × experts",
      [
        v("L_layers", "Transformer layers", m.layers),
        v("H", "Hidden width", H),
        v("Q_heads", "Query vector width", Q),
        v("K_heads", "KV vector width", K),
        v("I", "Intermediate width", m.intermediate),
        v("experts", "Total experts", m.experts),
        v("V", "Vocabulary size", m.vocab, "", () =>
          modelInput(r, "vocab", "Vocabulary size"),
        ),
        ...(shared
          ? [
              v(
                "embedding_copies",
                "Embedding/output-head weight copies",
                m.tie_embeddings ? 1 : 2,
                "",
                () =>
                  modelInput(
                    r,
                    "tie_embeddings",
                    "Tied embeddings",
                    String(m.tie_embeddings),
                  ),
              ),
              v(
                "qk_norm_extra",
                "Optional QK normalization parameters",
                m.qk_norm ? m.layers * 2 * m.head_dim : 0,
                "parameters",
                () =>
                  leaf(
                    "QK normalization weights",
                    m.qk_norm ? m.layers * 2 * m.head_dim : 0,
                    "parameters",
                    "Architecture equation",
                    `L × 2 × head_dim if QK normalization is enabled; otherwise 0. L=${m.layers}, head_dim=${m.head_dim}, enabled=${m.qk_norm}.`,
                  ),
              ),
              v(
                "router_extra",
                "Optional expert-router parameters",
                m.experts > 1 ? m.layers * H * m.experts : 0,
                "parameters",
                () =>
                  leaf(
                    "Expert-router weights",
                    m.experts > 1 ? m.layers * H * m.experts : 0,
                    "parameters",
                    "Architecture equation",
                    `L × H × experts for MoE; otherwise 0. L=${m.layers}, H=${H}, experts=${m.experts}.`,
                  ),
              ),
            ]
          : []),
      ],
      shared ? r.accounting.shared_parameters : r.accounting.expert_parameters,
      "parameters",
    ),
    summary:
      "Uniform decoder geometry is counted analytically. Shared attention/norm/embedding/router weights and expert MLP weights are separated to apply EP correctly. This is not a checkpoint-file-size measurement.",
  });
}
function residentCount(r: Obj, phase: string): Calculation {
  const c = r.config,
    p = c[phase],
    B = r.memory[phase].representative_batch;
  let groups = 0;
  for (let i = 0; i < c.requests; i += c.prefill.batch_size)
    groups +=
      phase === "prefill"
        ? 1
        : Math.ceil(Math.min(c.prefill.batch_size, c.requests - i) / B);
  return {
    ...calc(
      "Conservatively resident requests",
      "N_resident = min(N_req, B × ceil(phase_batches / replicas))",
      [
        v("N_req", "Total requests", c.requests, "", () =>
          input(r, "requests", "Total requests"),
        ),
        v("B", "Representative batch", B, "requests", () =>
          leaf(
            "Memory accounting batch",
            B,
            "requests",
            "Resolved phase batching",
            `Prefill B=min(prefill batch limit,N_req); decode B=min(prefill limit,decode limit,N_req). Here prefill limit=${c.prefill.batch_size}, decode limit=${c.decode.batch_size}, N_req=${c.requests}.`,
          ),
        ),
        v("phase_batches", "Fixed phase groups", groups, "batches", () =>
          leaf(
            "Fixed phase batch count",
            groups,
            "batches",
            "Fixed-group scheduler",
            `Prefill groups are ceil(N_req/B_prefill). Decode splits each prefill group into ceil(group_size/B_decode) groups; it never merges groups. N_req=${c.requests}, B_prefill=${c.prefill.batch_size}, B_decode=${c.decode.batch_size}.`,
          ),
        ),
        v("replicas", "Phase replicas", p.replicas, "", () =>
          poolInput(r, phase, "replicas", "Phase replicas"),
        ),
      ],
      r.memory[phase].resident_requests,
      "requests",
    ),
    summary:
      "Capacity reservation includes queued requests assigned to a replica, not just the currently executing batch. This deliberately conservative rule is independent of the admission permit limit.",
  };
}
function busyCalculation(r: Obj, phase?: string): Calculation {
  const entries = Object.entries(r.accounting.busy_ms).filter(
    ([k]) =>
      !phase || k.startsWith((r.config.separated ? phase : "shared") + ":"),
  );
  const contributions = entries.map(([resource, busy]) => {
    const ph = resource.startsWith("decode:") ? "decode" : "prefill",
      p = r.config[ph],
      mult = p.tp * p.ep;
    return v(
      resource,
      "Stage GPU busy time",
      Number(busy) * mult,
      "GPU·ms",
      () => ({
        ...calc(
          resource + " busy time",
          "Busy_GPU_ms = Σ stage_service_ms × TP × EP",
          [
            ...r.timeline
              .filter((e: Obj) => e.resource === resource)
              .map((e: Obj) =>
                ms(`event_${e.event_id}`, eventLabel(e), e.duration_ms, () =>
                  eventService(r, e),
                ),
              ),
            v("TP", "Tensor parallelism", p.tp, "", () =>
              poolInput(r, ph, "tp", "Tensor parallelism"),
            ),
            v("EP", "Expert parallelism", p.ep, "", () =>
              poolInput(r, ph, "ep", "Expert parallelism"),
            ),
          ],
          Number(busy) * mult,
          "GPU·ms",
        ),
        summary:
          "Only GPU-stage reservations contribute to GPU busy time. Separate pipeline/handoff links are not GPU resources. Memory and communication inside a stage remain part of its occupancy.",
      }),
    );
  });
  return {
    ...calc(
      "Weighted GPU busy time",
      "B = Σ GPU_resource busy_ms × TP × EP",
      contributions,
      contributions.reduce((s, q) => s + Number(q.value), 0),
      "GPU·ms",
    ),
    summary:
      "Parallel GPU work is accumulated across resources; this sum can exceed elapsed time. GPU resource occupancy divides it by horizon × allocated GPU count.",
  };
}
function activePower(r: Obj, phase: string): Calculation {
  const q = r.accounting.energy_terms.find((x: Obj) => x.phase === phase);
  return {
    ...calc(
      "Modeled active GPU power",
      "P_active = min(P_TDP, P_idle + 0.85 × (P_TDP−P_idle))",
      [
        v("P_idle", "Idle GPU power", q.idle_watts, "W", () =>
          hardware(r, phase, "idle_watts", "Idle GPU power"),
        ),
        v("P_TDP", "GPU power cap", q.tdp_watts, "W", () =>
          hardware(r, phase, "tdp_watts", "GPU power cap"),
        ),
        v("0.85", "Activity coefficient", 0.85, "", () =>
          leaf(
            "Activity coefficient",
            0.85,
            "",
            "Behavioral energy assumption",
            "A disclosed activity fraction of the idle-to-TDP range. It has not been fitted to measured power traces.",
          ),
        ),
      ],
      q.active_watts,
      "W",
    ),
    summary:
      "This power estimate is applied during weighted stage busy time. Idle power is charged for every allocated GPU across the whole run.",
  };
}
function dynamicEnergy(r: Obj, phase: string): Calculation {
  const q = r.accounting.energy_terms.find((x: Obj) => x.phase === phase),
    busy = q.weighted_busy_ms / 1000;
  const vars = [
    v("B_s", "Weighted GPU busy time", busy, "GPU·s", () =>
      scaled(busyCalculation(r, phase), 0.001, "GPU·s"),
    ),
    v("P_idle", "Idle power", q.idle_watts, "W", () =>
      hardware(r, phase, "idle_watts", "Idle power"),
    ),
    v("P_TDP", "Power cap", q.tdp_watts, "W", () =>
      hardware(r, phase, "tdp_watts", "Power cap"),
    ),
  ];
  if (q.mode === "activity")
    vars.push(
      v("P_active", "Modeled active GPU power", q.active_watts, "W", () =>
        activePower(r, phase),
      ),
    );
  else
    for (const [key, value] of Object.entries(q.counters))
      vars.push(
        v(key, "Accumulated " + key, Number(value), "", () =>
          trafficPhase(r, phase, key),
        ),
      );
  if (q.mode === "component")
    for (const [key, value] of Object.entries(q.coefficients))
      vars.push(
        v(key, "Energy coefficient", Number(value), "pJ/unit", () =>
          hardware(
            r,
            phase,
            key,
            "Energy coefficient",
            Number(value),
            "pJ/unit",
          ),
        ),
      );
  return {
    ...calc(
      `${phase} dynamic energy`,
      q.mode === "activity"
        ? "E_dynamic = (P_active − P_idle) × B_s"
        : "E_dynamic = min(Σ work_k × pJ_k × 10⁻¹², (P_TDP−P_idle) × B_s)",
      vars,
      q.dynamic_j,
      "J",
    ),
    summary:
      q.mode === "activity"
        ? "Dynamic energy is the power increment above idle, integrated over weighted stage occupancy. It inherits the fitted stage costs and scheduling workload."
        : "Explicit work/byte counters and pJ coefficients replace the activity increment. The result is capped by the available idle-to-TDP power budget. No independent measured energy calibration is claimed.",
  };
}
function componentIntervals(r: Obj, key: string): Calculation {
  const contributions = r.timeline
    .filter((e: Obj) => e.components[key] > 0)
    .map((e: Obj) => {
      const ph = e.phase === "decode" ? "decode" : "prefill",
        p = r.config[ph],
        mult = e.phase === "transfer" ? 1 : p.tp * p.ep;
      return v(
        `event_${e.event_id}`,
        eventLabel(e),
        e.components[key] * mult,
        "GPU·ms",
        () =>
          leaf(
            "Nominal " + key + " contribution",
            e.components[key] * mult,
            "GPU·ms",
            "Executed interval resource accounting",
            `Interval component=${numberText(e.components[key])} ms, multiplier=${mult}. Stage components sum the separate analytical column over assigned layers, with applicable output-head work. They overlap and are not additive pieces of fitted service time.`,
          ),
      );
    });
  return {
    ...calc(
      key + " resource demand",
      "Demand = Σ interval_component_ms × GPU_multiplier",
      contributions,
      r.component_work_ms[key],
      "GPU·ms",
    ),
    summary:
      "This is nominal work accumulated over executed intervals, not a measured stall-time decomposition.",
  };
}
function trafficPhase(r: Obj, phase: string, key: string): Calculation {
  const p = r.config[phase],
    m = r.model,
    c = r.config,
    d = (
      { bf16: 2, fp16: 2, fp32: 4, tf32: 4, fp8: 1, int8: 1, fp4: 0.5 } as Obj
    )[c.precision];
  const vectors = [
    "Input RMSNorm",
    "FFN RMSNorm",
    "QK normalization",
    "Rotary position",
    "SwiGLU",
  ];
  const contributions = r.timeline
    .filter(
      (e: Obj) =>
        e.phase === phase || (phase === "prefill" && e.phase === "transfer"),
    )
    .map((e: Obj) => {
      let value = 0,
        vars: Calculation["variables"] = [],
        formula = "";
      if (e.phase === "transfer" || e.kind === "pipeline_transfer") {
        if (key !== "network_bytes") return null;
        const mult = e.phase === "transfer" ? 1 : p.tp * p.ep;
        value = e.payload_bytes * mult;
        formula = "Network bytes = payload × GPU_multiplier";
        vars = [
          v("payload", "Transfer payload", e.payload_bytes, "bytes", () =>
            transferPayload(r, e),
          ),
          v("GPU_multiplier", "Accounting multiplier", mult),
        ];
      } else {
        const shape = r.execution_shapes?.[e.shape_id];
        if (!shape) return null;
        const flops = key === "tensor_flops" || key === "vector_flops";
        const ops = shape.operations.filter(
          (o: Obj) =>
            !flops || vectors.includes(o.name) === (key === "vector_flops"),
        );
        const metric = flops ? "flops" : key;
        const L = e.layers_per_stage,
          mult = p.tp * p.ep;
        let head = 0;
        if (
          e.stage === p.pp - 1 &&
          (phase === "decode" ||
            e.context >= c.input_tokens ||
            c.timing_mode === "calibrated")
        )
          head =
            key === "tensor_flops"
              ? (2 * shape.batch * m.hidden * m.vocab) / p.tp
              : key === "hbm_bytes"
                ? (m.hidden * m.vocab * d) / p.tp
                : 0;
        value = (sum(ops, (o) => o[metric] || 0) * L + head) * mult;
        formula =
          "Counter = (Σ operation_work × layers_per_stage + head_work) × TP × EP";
        vars = ops
          .filter((o: Obj) => (o[metric] || 0) > 0)
          .map((o: Obj) =>
            v(o.id, o.name, o[metric], flops ? "FLOP" : "bytes", () =>
              metric === "network_bytes"
                ? term(
                    r,
                    calc("Network", "", [], 0, ""),
                    v("Q_network", "Communication volume", o[metric]),
                    {
                      phase,
                      op: { ...o, batch: shape.batch, tokens: shape.tokens },
                      operation: "network_ms",
                    },
                  )
                : operationCalculation(
                    r,
                    phase,
                    {
                      ...o,
                      batch: shape.batch,
                      tokens: shape.tokens,
                      context: shape.context,
                    },
                    metric,
                  ),
            ),
          );
        vars.push(
          v("layers_per_stage", "Assigned layers", L),
          v("TP × EP", "GPU accounting multiplier", mult),
          v(
            "head_work",
            "Additional output-head counter",
            head,
            flops ? "FLOP" : "bytes",
            () =>
              leaf(
                "Output-head counter",
                head,
                flops ? "FLOP" : "bytes",
                "Output-head accounting convention",
                `The last stage adds 2×B×H×V/TP FLOP and H×V×d/TP bytes when applicable; this interval has B=${shape.batch}, H=${m.hidden}, V=${m.vocab}, TP=${p.tp}, d=${d}. Endpoint kernel timing and counter accounting are separate conventions.`,
              ),
          ),
        );
      }
      if (value === 0) return null;
      return v(
        `event_${e.event_id}`,
        eventLabel(e),
        value,
        key.endsWith("flops") ? "FLOP" : "bytes",
        () => ({
          ...calc(
            "Interval " + key,
            formula,
            vars,
            value,
            key.endsWith("flops") ? "FLOP" : "bytes",
          ),
          summary:
            "Counters are accumulated from modeled work, not hardware performance counters. Embedding/final-normalization/sampling costs do not add separate traffic counters.",
        }),
      );
    })
    .filter(Boolean) as Calculation["variables"];
  return {
    ...calc(
      `${phase} accumulated ${key}`,
      "Phase counter = Σ executed interval contributions",
      contributions,
      r.accounting.phase_traffic[phase][key],
      key.endsWith("flops") ? "FLOP" : "bytes",
    ),
    summary:
      "This list uses actual executed shapes, not one representative layer multiplied by an assumed token count. Each entry explains its layer and GPU multipliers.",
  };
}
function analyticalBytes(
  r: Obj,
  phase: string,
  o: Obj,
  key: string,
): Calculation {
  const c = r.config,
    m = r.model,
    p = c[phase],
    B = o.batch ?? o.fit_batch ?? Math.min(p.batch_size, c.requests),
    t =
      o.tokens ??
      o.fit_tokens ??
      (phase === "decode" ? 1 : Math.min(c.prefill_chunk, c.input_tokens)),
    context = o.context ?? o.fit_context ?? t,
    n = B * t;
  const name = o.name,
    H = m.hidden,
    I = m.intermediate,
    Q = m.heads * m.head_dim,
    K = m.kv_heads * m.head_dim,
    d = (
      { bf16: 2, fp16: 2, fp32: 4, tf32: 4, fp8: 1, int8: 1, fp4: 0.5 } as Obj
    )[c.precision],
    dk = ({ bf16: 2, fp16: 2, fp8: 1 } as Obj)[c.kv_precision];
  const active =
    m.experts === 1 ? 1 : m.experts * (1 - (1 - m.top_k / m.experts) ** n);
  const weights: Obj = {
    "Input RMSNorm": "H d/TP",
    "FFN RMSNorm": "H d/TP",
    "QK normalization": "2 D_head d",
    "QKV projection": "H(Q+2K)d/TP",
    "Output projection": "Q H d/TP",
    "Expert router": "H experts d/TP",
    "Gate + up projection": "2 H I active d/(TP EP)",
    "Down projection": "H I active d/(TP EP)",
  };
  const activation: Obj = {
    "Input RMSNorm": "2nH d/TP",
    "FFN RMSNorm": "2nH d/TP",
    "QK normalization": "2n(Q+K)d/TP",
    "QKV projection": "2nH d/TP",
    "Rotary position": "2nQ d/TP",
    Attention: "2nQ d/TP",
    "Output projection": "2nH d/TP",
    "Expert router": "n experts d/TP",
    "Gate + up projection": "3nI d top_k/(TP EP)",
    SwiGLU: "3nI d top_k/(TP EP)",
    "Down projection": "2nH d/TP",
  };
  const formula =
    key === "weight_bytes"
      ? weights[name] || "0"
      : key === "activation_bytes"
        ? activation[name] || "0"
        : name === "Attention"
          ? "max(0,KV_read−SRAM_per_layer) + KV_write"
          : "0";
  const kvRead =
      2 * B * context * Math.max(1, m.kv_heads / p.tp) * m.head_dim * dk,
    kvWrite = 2 * B * t * Math.max(1, m.kv_heads / p.tp) * m.head_dim * dk,
    sram = (p.hardware.sram_cache_MB * 1e6) / (m.layers / p.pp);
  return explainCalculation(
    r,
    {
      ...calc(
        labelForBytes(key),
        "Bytes = " + formula,
        [
          v("B", "Actual batch", B),
          v("t", "Query tokens", t),
          v("n", "Batch × query tokens", n),
          v("context", "Context length", context),
          v("H", "Hidden width", H),
          v("I", "Intermediate width", I),
          v("Q_heads", "Query vector width", Q),
          v("K_heads", "KV vector width", K),
          v("TP", "Tensor parallelism", p.tp),
          v("EP", "Expert parallelism", p.ep),
          v("d_weight", "Bytes per weight value", d),
          v("d_KV", "Bytes per KV value", dk),
          v("D_head", "Head dimension", m.head_dim),
          v("experts", "Total experts", m.experts),
          v("top_k", "Experts per token", m.top_k),
          v("active", "Expected experts touched", active, "", () =>
            leaf(
              "Expected distinct experts",
              active,
              "experts",
              "Uniform routing approximation",
              `active=E[1−(1−top_k/E)ⁿ], except active=1 for dense models. E=${m.experts}, top_k=${m.top_k}, n=${n}.`,
            ),
          ),
          ...(key === "kv_hbm_bytes"
            ? [
                v("KV_read", "Nominal KV read", kvRead, "bytes", () =>
                  leaf(
                    "Nominal KV read",
                    kvRead,
                    "bytes",
                    "KV geometry",
                    `2×B×context×max(1,KV_heads/TP)×head_dim×d_KV with B=${B}, context=${context}, KV_heads=${m.kv_heads}, TP=${p.tp}, head_dim=${m.head_dim}, d_KV=${dk}.`,
                  ),
                ),
                v("KV_write", "KV write", kvWrite, "bytes", () =>
                  leaf(
                    "KV write",
                    kvWrite,
                    "bytes",
                    "KV geometry",
                    `2×B×query_tokens×max(1,KV_heads/TP)×head_dim×d_KV; query_tokens=${t}. Writes always reach HBM.`,
                  ),
                ),
                v(
                  "SRAM_per_layer",
                  "Reserved KV SRAM per layer",
                  sram,
                  "bytes",
                  () =>
                    leaf(
                      "SRAM allocation per layer",
                      sram,
                      "bytes",
                      "Ideal KV reuse assumption",
                      `SRAM_MB×10⁶/(model_layers/PP), with SRAM=${p.hardware.sram_cache_MB} MB, layers=${m.layers}, PP=${p.pp}. This is not physical L2 capacity.`,
                    ),
                ),
              ]
            : []),
        ],
        o[key] || 0,
        "bytes",
      ),
      summary:
        "Analytical resource-accounting bytes. Weights/activation/KV estimates are separate from the profiling-convention features used by fitted kernel timings.",
    },
    { phase },
  );
}
const labelForBytes = (key: string) =>
  (
    ({
      weight_bytes: "Analytical weight reads",
      activation_bytes: "Analytical activation traffic",
      kv_hbm_bytes: "Analytical KV HBM traffic",
    }) as Obj
  )[key];

export const intervalDerivation = (r: Obj, e: Obj): Calculation => ({
  ...eventService(r, e),
  variables: [
    ...eventService(r, e).variables,
    ms("schedule", "How this interval was scheduled", e.end_ms, () =>
      eventCalculation(r, e),
    ),
  ],
});
