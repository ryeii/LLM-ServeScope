import { calc, v, type Calculation } from "./calculations";
import { storageEventCalculation } from "./storageCalculations";
type Obj = Record<string, any>;
const sum = (rows: Obj[], f: (row: Obj) => number) =>
  rows.reduce((n, row) => n + f(row), 0);
export function activeAllocationCalculation(r: Obj, q: Obj): Calculation {
  const p = r.config.decode,
    m = r.model,
    request = r.requests.find((row: Obj) => row.id === q.request_id);
  const dtype = ({ bf16: 2, fp16: 2, fp8: 1 } as Obj)[r.config.kv_precision];
  return {
    ...calc(
      "Active KV remote fraction",
      "Remote_fraction = max_KV > 0 ? 1 − resident_bytes_per_gpu / max_KV_bytes_per_gpu : 0",
      [
        v(
          "resident_bytes_per_gpu",
          "Pinned active HBM allocation",
          q.resident_bytes_per_gpu,
          "bytes",
        ),
        v(
          "max_KV_bytes_per_gpu",
          "Maximum active payload per GPU",
          q.max_kv_bytes_per_gpu,
          "bytes",
          () =>
            calc(
              "Maximum active KV allocation",
              "Q_GPU = 2 × (input+output−1) × max(1,KV_heads/TP) × head_dim × dtype_bytes × layers/PP",
              [
                v("input", "Prompt tokens", request.input_tokens),
                v("output", "Output tokens", request.output_tokens),
                v("KV_heads", "KV heads", m.kv_heads),
                v("TP", "Tensor parallelism", p.tp),
                v("head_dim", "Head dimension", m.head_dim),
                v("dtype_bytes", "KV bytes/element", dtype),
                v("layers", "Transformer layers", m.layers),
                v("PP", "Pipeline stages", p.pp),
              ],
              q.max_kv_bytes_per_gpu,
              "bytes",
              "No decode working allocation is required for a one-token output.",
            ),
        ),
      ],
      q.remote_fraction,
      "fraction",
    ),
    summary:
      "Residency is assigned atomically using the free active HBM budget and backing capacity. The fraction remains pinned for the request lifetime. Remote bytes are striped uniformly across tokens/layers; this is a behavioral placement assumption.",
  };
}
function activePayload(r: Obj, e: Obj): Calculation {
  const p = r.config.decode,
    m = r.model;
  return {
    ...calc(
      "Active KV transfer payload",
      e.payload_kind === "resident_migration"
        ? "Q = resident_fraction × prompt_KV_payload"
        : "Q = Σ remote_fraction_i × 2 × KV_tokens × max(1,KV_heads/TP) × head_dim × dtype × layers/PP × TP × EP × (PP for initialization)",
      [
        v("KV_tokens", "KV tokens transferred", e.kv_tokens || 0),
        v("KV_heads", "KV heads", m.kv_heads),
        v("head_dim", "Head width", m.head_dim),
        v("layers", "Layers", m.layers),
        v("TP", "Tensor parallelism", p.tp),
        v("PP", "Pipeline stages", p.pp),
        v("EP", "Expert replicas", p.ep),
        v(
          "dtype",
          "KV bytes per element",
          ({ bf16: 2, fp16: 2, fp8: 1 } as Obj)[r.config.kv_precision],
        ),
        ...(e.request_ids || []).map((id: number) => {
          const row = r.active_kv.allocations.find(
            (q: Obj) => q.request_id === id,
          );
          return v(
            "request_" + id,
            `Request #${id} remote fraction`,
            row.remote_fraction,
            "fraction",
            () => activeAllocationCalculation(r, row),
          );
        }),
      ],
      e.payload_bytes,
      "bytes",
    ),
    summary:
      "Reads fetch pre-existing context before each decode query. Appends write only newly produced KV. Initialization covers the prompt across all pipeline stages. SSD→host and host→GPU intervals carry the same payload; they are distinct physical hops, not independent logical KV reads.",
  };
}
function ioElapsed(r: Obj, e: Obj): Calculation {
  const previous = r.timeline.find(
    (p: Obj) => p.event_id === e.previous_resource_event,
  );
  return {
    ...calc(
      `${e.tier} ${e.operation} · ${e.cause}`,
      "Elapsed_ms = FIFO_wait_ms + service_ms",
      [
        v(
          "payload_bytes",
          "Active transfer geometry",
          e.payload_bytes,
          "bytes",
          () => activePayload(r, e),
        ),
        v("FIFO_wait_ms", "Shared path queue", e.queue_ms, "ms", () =>
          calc(
            "Shared I/O queue",
            "Wait = grant_ms − ready_ms",
            [
              v("ready_ms", "Transfer requested", e.ready_ms, "ms"),
              v("grant_ms", "Resource granted", e.start_ms, "ms"),
              ...(previous
                ? [
                    v(
                      "previous_end",
                      "Previous shared-path reservation",
                      previous.end_ms,
                      "ms",
                      () => storageEventCalculation(r, previous),
                    ),
                  ]
                : []),
            ],
            e.queue_ms,
            "ms",
          ),
        ),
        v("service_ms", "I/O service function", e.duration_ms, "ms", () =>
          storageEventCalculation(r, e, true),
        ),
      ],
      e.queue_ms + e.duration_ms,
      "ms",
    ),
    summary:
      "Active KV and retained prefixes share this FIFO resource. Staging is serial and bounded by the GPU tile buffer; this interval aggregates tile service, not full-context simultaneous GPU residency.",
  };
}
export function activeStageCalculation(
  r: Obj,
  e: Obj,
  reference: () => Calculation,
): Calculation {
  const events = (e.active_io_event_ids || [])
    .map((id: number) => r.timeline.find((row: Obj) => row.event_id === id))
    .filter(Boolean);
  const staging = e.kind === "active_staging";
  return {
    ...calc(
      staging
        ? "Active KV initialization · GPU staging reservation"
        : "Decode GPU stage with active-KV streaming",
      staging
        ? "Hold_ms = wait_for_other_stages + wire_ms + initialization_IO_ms"
        : "Stage_ms = remote_read_elapsed_ms + reference_GPU_ms + append_write_elapsed_ms",
      [
        ...(staging
          ? [
              v(
                "wait_for_other_stages",
                "Wait for all GPU staging buffers",
                e.active_group_wait_ms || 0,
                "ms",
              ),
              v("wire_ms", "Handoff wire service", e.active_wire_ms || 0, "ms"),
            ]
          : [
              v(
                "reference_GPU_ms",
                "HBM-reference GPU operators",
                e.gpu_service_ms,
                "ms",
                reference,
              ),
            ]),
        ...events.map((io: Obj) =>
          v(
            "io_" + io.event_id,
            `${io.tier} ${io.operation} · ${io.cause}`,
            io.queue_ms + io.duration_ms,
            "ms",
            () => ioElapsed(r, io),
          ),
        ),
      ],
      e.duration_ms,
      "ms",
    ),
    summary: staging
      ? "The buffers are held until destination backing is ready. This reservation can overlap the handoff-channel reservation; do not add nested reservations twice."
      : "GPU resources are held during serial remote-I/O stalls. HBM-resident operator profiles provide reference compute/service; transfer stalls are added once. Streaming-kernel overhead and overlap remain unvalidated.",
  };
}
export function activeHandoffCalculation(r: Obj, e: Obj): Calculation {
  const events = (e.active_stream_event_ids || [])
    .map((id: number) => r.timeline.find((row: Obj) => row.event_id === id))
    .filter(Boolean);
  return {
    ...calc(
      "Staged active KV handoff",
      "Handoff_hold_ms = staging_buffer_wait_ms + wire_service_ms + backing_initialization_IO_ms",
      [
        v(
          "staging_buffer_wait_ms",
          "Wait for decode GPU staging",
          e.active_staging_wait_ms,
          "ms",
        ),
        v(
          "wire_service_ms",
          "Network wire service",
          e.active_wire_ms,
          "ms",
          () =>
            calc(
              "KV wire service",
              "Wire_ms = payload/(link_GBps×10⁹)×1000 + latency_us/1000",
              [
                v("payload", "Prompt KV payload", e.payload_bytes, "bytes"),
                v(
                  "link_GBps",
                  "Configured handoff bandwidth",
                  e.link_bandwidth_GBps,
                  "GB/s",
                ),
                v("latency_us", "Handoff latency", e.link_latency_us, "µs"),
              ],
              e.active_wire_ms,
              "ms",
            ),
        ),
        ...events.map((io: Obj) =>
          v(
            "io_" + io.event_id,
            `${io.tier} ${io.operation}`,
            io.duration_ms + io.queue_ms,
            "ms",
            () => ioElapsed(r, io),
          ),
        ),
      ],
      e.duration_ms,
      "ms",
    ),
    summary:
      "The handoff channel stays reserved until active backing is ready. Wire and device costs aggregate serial tile cycles; the intermediate wire-end timestamp does not imply a full remote context was materialized in HBM.",
  };
}
export function activeMetricCalculation(r: Obj, key: string): Calculation {
  const a = r.active_kv;
  if (key === "stage_io_stall_ms") {
    const stages = r.timeline.filter(
      (e: Obj) =>
        e.shape_id && e.phase === "decode" && e.gpu_service_ms !== undefined,
    );
    return {
      ...calc(
        "Decode-stage active I/O stalls",
        "Stalls = Σ(stage remote_read_elapsed + append_write_elapsed)",
        stages.map((e: Obj) =>
          v(
            "stage_" + e.event_id,
            `Decode batch ${e.batch_id} stage ${e.stage}`,
            e.active_read_ms + e.active_append_ms,
            "ms",
            () =>
              activeStageCalculation(r, e, () =>
                calc(
                  "Reference GPU service",
                  "Reference GPU service",
                  [],
                  e.gpu_service_ms,
                  "ms",
                ),
              ),
          ),
        ),
        a.stats[key] || 0,
        "ms",
      ),
      summary:
        "Sum across GPU-stage reservations; it includes I/O queueing and can exceed wall-clock elapsed time. It excludes initialization and pure reference GPU operator service.",
    };
  }
  const events = r.timeline.filter(
    (e: Obj) =>
      e.kind === "storage_io" &&
      e.scope === "active" &&
      (key === "ssd_service_ms"
        ? e.tier === "ssd"
        : e.cause ===
            (key === "attention_read_bytes"
              ? "active_read"
              : "active_append") &&
          e.tier === (a.backing === "ssd" ? "ssd" : "pcie")),
  );
  return {
    ...calc(
      key,
      key === "ssd_service_ms"
        ? "Service_ms = Σ active SSD operation service_ms"
        : "Logical_bytes = Σ canonical active backing transfer payload",
      events.map((e: Obj) =>
        v(
          "io_" + e.event_id,
          `${e.cause} · batch ${e.batch_id}`,
          key === "ssd_service_ms" ? e.duration_ms : e.payload_bytes,
          key === "ssd_service_ms" ? "ms" : "bytes",
          () => ioElapsed(r, e),
        ),
      ),
      a.stats[key] || 0,
      key === "ssd_service_ms" ? "ms" : "bytes",
    ),
    summary:
      "These counters include only active decode KV. Retained-prefix counters remain separate. SSD and PCIe copies are not counted twice as logical bytes.",
  };
}
export function activeRequestPath(
  r: Obj,
  q: Obj,
  key: string,
  explainEvent: (e: Obj) => Calculation,
): Calculation {
  const k = q.cache,
    release = key === "prefill_release_ms",
    finish = key === "finish_ms";
  const parts: [string, string, number, number, (e: Obj) => boolean][] = [
    [
      "batch",
      "Fixed microbatch formation",
      q.arrival_ms,
      q.batch_ready_ms,
      () => false,
    ],
    [
      "prefill_admission",
      "Prefill active-HBM admission",
      q.batch_ready_ms,
      k.memory_ready_ms,
      (e) => e.kind === "kv_memory_wait",
    ],
    [
      "restore",
      "Retained-prefix restoration",
      k.memory_ready_ms,
      k.cache_ready_ms,
      (e) =>
        e.kind === "storage_io" &&
        e.scope === "prefix" &&
        e.cause === "restore",
    ],
    [
      "prefill",
      "Prefill GPU work and queues",
      k.cache_ready_ms,
      k.gpu_prefill_end_ms,
      (e) => e.phase === "prefill",
    ],
    [
      "publish",
      "Retained-prefix publication",
      k.gpu_prefill_end_ms,
      k.publication_end_ms,
      (e) =>
        e.kind === "storage_io" &&
        e.scope === "prefix" &&
        e.cause !== "restore",
    ],
    [
      "active_admission",
      "Active decode backing admission",
      k.publication_end_ms,
      k.active_admission_end_ms,
      () => false,
    ],
    [
      "initialize",
      "Active KV handoff and backing initialization",
      k.active_admission_end_ms,
      k.handoff_end_ms,
      (e) =>
        r.config.separated
          ? e.kind === "active_handoff"
          : e.kind === "active_staging",
    ],
    ...(finish
      ? [
          [
            "decode",
            "Decode steps with active KV I/O",
            q.first_token_ms,
            q.finish_ms,
            (e: Obj) => e.phase === "decode" && e.kind !== "active_staging",
          ] as [string, string, number, number, (e: Obj) => boolean],
        ]
      : []),
  ];
  return {
    ...calc(
      `Request #${q.id} ${release ? "prefill KV release" : finish ? "completion" : "first token"}`,
      "Time_ms = arrival_ms + Σ ordered elapsed spans",
      [
        v("arrival_ms", "Request arrival", q.arrival_ms, "ms"),
        ...parts.map(([symbol, label, begin, end, filter]) =>
          v(symbol, label, end - begin, "ms", () => {
            const events = r.timeline.filter(
              (e: Obj) => e.request_ids.includes(q.id) && filter(e),
            );
            return {
              ...calc(
                label,
                "Elapsed_span = end_ms − begin_ms",
                [
                  v("begin_ms", "Span begins", begin, "ms"),
                  v("end_ms", "Span completes", end, "ms"),
                  ...events.map((e: Obj) =>
                    v(
                      "event_" + e.event_id,
                      `${e.kind || e.phase} · ${e.resource}`,
                      e.duration_ms,
                      "ms",
                      () => explainEvent(e),
                    ),
                  ),
                ],
                end - begin,
                "ms",
              ),
              summary:
                "Child reservations explain this span; concurrent or nested reservations are not summed again. Waits can include GPU/staging queues, backing capacity and microbatch barriers.",
            };
          }),
        ),
      ],
      finish ? q.finish_ms : release ? k.prefill_release_ms : q.first_token_ms,
      "ms",
    ),
    summary:
      "Active decode backing is separate from retained prefixes. Per-token active I/O is already inside each decode GPU-stage reservation; it is not added a second time to the request span.",
  };
}
