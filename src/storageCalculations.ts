import { calc, v, type Calculation, numberText } from "./calculations";
import { eventService, eventCalculation } from "./derivations";
type Obj = Record<string, any>;
export const storageReport =
  "https://github.com/ryeii/LLM-ServeScope/blob/main/docs/storage-2026-10-20/README.md";
const total = (xs: Obj[], f: (x: Obj) => number) =>
  xs.reduce((s, x) => s + f(x), 0);
export function storageEventCalculation(r: Obj, e: Obj): Calculation {
  if (e.kind === "kv_memory_wait")
    return {
      ...calc(
        "Active KV memory admission wait",
        "Wait_ms = grant_ms − ready_ms",
        [
          v("ready_ms", "Request ready", e.start_ms, "ms"),
          v("grant_ms", "Active KV memory granted", e.end_ms, "ms"),
          v(
            "bytes",
            "Pinned per-GPU request KV allocation",
            e.payload_bytes,
            "bytes",
          ),
          v(
            "capacity_bytes",
            "Active KV memory budget",
            e.capacity_bytes,
            "bytes",
          ),
          ...(e.blocking_allocations || []).map((holder: Obj) =>
            v(
              "holder_" + holder.request_id,
              `Active ${holder.phase} request #${holder.request_id}`,
              holder.bytes,
              "bytes",
              () =>
                storageRequestCalculation(
                  r,
                  r.requests.find((q: Obj) => q.id === holder.request_id),
                  holder.phase === "decode"
                    ? "finish_ms"
                    : "prefill_release_ms",
                ),
            ),
          ),
        ],
        e.duration_ms,
        "ms",
      ),
      summary:
        "The request waits until the active-KV container can reserve its maximum context payload. Retained-prefix cache space is reserved separately. This interval represents waiting, not I/O service.",
      source: "SimPy memory admission · recorded wait",
    };
  const f = e.service_model,
    c = f.coefficients,
    chunk = e.chunk_bytes;
  const terms = [
    v("L", "Path intercept", c[0], "ms"),
    v("b", "Linear size coefficient", c[1], "ms/MB"),
  ];
  if (c.length > 2)
    terms.push(
      v(
        "a",
        f.family === "sqrt-affine"
          ? "Square-root coefficient"
          : "Additional slope above 0.25 MB",
        c[2],
        f.family === "sqrt-affine" ? "ms/√MB" : "ms/MB",
      ),
    );
  const equation =
    f.family === "sqrt-affine"
      ? "f(x)=L+b x+a √x"
      : f.family === "two-slope"
        ? "f(x)=L+b x+a max(0,x−0.25)"
        : "f(x)=L+b x";
  const service: Calculation = {
    ...calc(
      e.tier + " " + e.operation + " service",
      `${equation}; x=bytes/10⁶\nService_ms = full_chunks × f(chunk_bytes/10⁶) + f(tail_bytes/10⁶) when tail>0`,
      [
        ...terms,
        v(
          "payload_bytes",
          "Aggregate cache-object payload",
          e.payload_bytes,
          "bytes",
        ),
        v("chunk_bytes", "I/O chunk size", chunk, "bytes"),
        v("full_chunks", "Full chunks", e.full_chunks),
        v("tail_bytes", "Remainder", e.tail_bytes, "bytes"),
      ],
      e.duration_ms,
      "ms",
    ),
    summary:
      e.profile === "local-mooncake-path"
        ? "Coefficients fitted to local Mooncake DiskHashTable measurements: QD1, warm buffered reads, fsync writes. This is a file-path fit, not measured performance of the selected GPU server or a customer SSD."
        : "Configured bandwidth/latency assumptions expressed as an affine service function. PCIe/HBM paths and manual SSD profiles have no measured calibration.",
    source: e.profile,
    links: [
      {
        label: "Data coverage, selected functions and held-out error",
        url: storageReport,
      },
    ],
  };
  if (e.extrapolated)
    service.note =
      "At least one chunk size lies outside the measured I/O-size range.";
  return {
    ...calc(
      `${e.tier} ${e.operation} · ${e.cause}`,
      "End_ms = ready_ms + queue_ms + service_ms",
      [
        v("ready_ms", "Transfer requested", e.ready_ms, "ms"),
        v("queue_ms", "Wait for FIFO transfer path", e.queue_ms, "ms", () => ({
          ...calc(
            "Transfer queue wait",
            "Wait = start_ms − ready_ms",
            [
              v("ready_ms", "Ready time", e.ready_ms, "ms"),
              v("start_ms", "Resource grant", e.start_ms, "ms"),
            ],
            e.queue_ms,
            "ms",
          ),
          summary: `Shared resource ${e.resource}; reads and writes compete for the same path. Queue delay is added once outside the service function.`,
        })),
        v(
          "service_ms",
          "I/O service function",
          e.duration_ms,
          "ms",
          () => service,
        ),
      ],
      e.end_ms,
      "ms",
    ),
    summary: `${e.payload_bytes.toLocaleString()} bytes. ${e.cause === "restore" ? "The request cannot compute its suffix until restoration finishes." : e.cause === "spill" ? "An LRU victim moves to its spill tier. The destination becomes ready only after this transfer completes; the triggering publication waits." : "Cache publication completes before this request hands off to decode."}`,
    source: "Executed storage reservation · " + e.resource,
  };
}
function span(
  r: Obj,
  q: Obj,
  title: string,
  begin: number,
  end: number,
  predicate: (e: Obj) => boolean,
): Calculation {
  const events = r.timeline.filter(
    (e: Obj) => e.request_ids.includes(q.id) && predicate(e),
  );
  const service = total(events, (e) => e.duration_ms);
  const wait = Math.max(0, end - begin - service);
  return {
    ...calc(
      title,
      "Elapsed_ms = Σ interval_ms + remaining_wait_ms",
      [
        ...events.map((e: Obj) =>
          v(
            "event_" + e.event_id,
            e.kind === "storage_io"
              ? `${e.tier} ${e.operation}`
              : e.kind === "kv_memory_wait"
                ? "KV admission wait"
                : `${e.phase} stage/link`,
            e.duration_ms,
            "ms",
            () =>
              e.phase === "storage"
                ? storageEventCalculation(r, e)
                : eventService(r, e),
          ),
        ),
        v(
          "remaining_wait_ms",
          "Resource, placement admission or shared-fill wait",
          wait,
          "ms",
          () => ({
            ...calc(
              "Waiting inside this span",
              "Wait_ms = elapsed_span − own_intervals",
              [
                ...events
                  .filter((e: Obj) => e.queue_ms > 0)
                  .map((e: Obj) =>
                    v(
                      "queue_" + e.event_id,
                      e.resource + " FIFO queue",
                      e.queue_ms,
                      "ms",
                      () =>
                        e.phase === "storage"
                          ? storageEventCalculation(r, e)
                          : eventCalculation(r, e),
                    ),
                  ),
              ],
              wait,
              "ms",
            ),
            summary:
              "This is the recorded elapsed gap. It can include phase/placement admission and waiting for another request to finish a coalesced cache publication. Microbatch members can also wait for one another’s restore/publication barrier. Parallel requests are not summed into this request’s latency.",
          }),
        ),
      ],
      end - begin,
      "ms",
    ),
    summary:
      "Uses this completed request’s actual intervals and timestamps. Overlap with other requests remains in the global event schedule.",
  };
}
export function storageRequestCalculation(
  r: Obj,
  q: Obj,
  key: string,
): Calculation {
  const k = q.cache,
    L = q.output_tokens;
  if (key === "batch_ready_ms")
    return calc(
      `Request #${q.id} microbatch readiness`,
      "Batch_ready_ms = max(arrival_ms of microbatch members)",
      (k.batch_request_ids || [q.id]).map((id: number) =>
        v(
          "request_" + id,
          `Request #${id} arrival`,
          r.requests.find((row: Obj) => row.id === id).arrival_ms,
          "ms",
        ),
      ),
      q.batch_ready_ms,
      "ms",
      "Fixed groups wait for their last member before memory admission and prefix restoration.",
    );
  if (key === "arrival_ms")
    return {
      ...calc(
        `Request #${q.id} arrival`,
        "Arrival_ms = recorded_timestamp − window_origin",
        [],
        q.arrival_ms,
        "ms",
      ),
      summary: r.config.storage.workload.length
        ? "Timestamp preserved from the selected workload window. The first selected arrival defines time zero; tied records retain source order."
        : "Generated deterministic arrivals with a documented repeated-prefix pattern. This workload is synthetic.",
      source: r.storage.workload_source.label,
    };
  if (["ttft_ms", "latency_ms", "tpot_ms"].includes(key)) {
    const first = key === "ttft_ms",
      tpot = key === "tpot_ms";
    return {
      ...calc(
        `Request #${q.id} ${first ? "TTFT" : tpot ? "TPOT" : "latency"}`,
        tpot
          ? "TPOT = (finish − first_token) / max(1,L_out−1)"
          : `${first ? "TTFT" : "Latency"} = ${first ? "first_token" : "finish"} − arrival`,
        [
          v("finish", "Completion", q.finish_ms, "ms", () =>
            storageRequestCalculation(r, q, "finish_ms"),
          ),
          v("first_token", "First token release", q.first_token_ms, "ms", () =>
            storageRequestCalculation(r, q, "first_token_ms"),
          ),
          v("arrival", "Arrival", q.arrival_ms, "ms", () =>
            storageRequestCalculation(r, q, "arrival_ms"),
          ),
          v("L_out", "This request’s output length", L, "tokens"),
        ].filter((row) =>
          tpot
            ? ["finish", "first_token", "L_out"].includes(row.symbol)
            : first
              ? ["first_token", "arrival"].includes(row.symbol)
              : ["finish", "arrival"].includes(row.symbol),
        ),
        q[key],
        "ms",
      ),
      summary:
        "The first token is released after cache publication and any required spill, handoff and decode-memory admission. Only subsequent token iterations contribute to TPOT.",
    };
  }
  const release = key === "prefill_release_ms";
  const first = key !== "finish_ms";
  const parts: [string, string, number, number, (e: Obj) => boolean][] = [
    [
      "batch_wait",
      "Fixed microbatch formation",
      q.arrival_ms,
      q.batch_ready_ms,
      () => false,
    ],
    [
      "memory_wait",
      "Active HBM admission",
      q.batch_ready_ms,
      k.memory_ready_ms,
      (e) => e.kind === "kv_memory_wait" && e.resource.includes("prefill"),
    ],
    [
      "restore",
      "Prefix restoration",
      k.memory_ready_ms,
      k.cache_ready_ms,
      (e) => e.kind === "storage_io" && e.cause === "restore",
    ],
    [
      "prefill",
      "Suffix prefill and GPU queues",
      k.cache_ready_ms,
      k.gpu_prefill_end_ms,
      (e) => e.phase === "prefill",
    ],
    [
      "publication",
      "Cache publication and eviction spill",
      k.gpu_prefill_end_ms,
      k.publication_end_ms,
      (e) => e.kind === "storage_io" && ["publish", "spill"].includes(e.cause),
    ],
    [
      "handoff",
      "Prefill/decode KV handoff",
      k.publication_end_ms,
      k.handoff_end_ms,
      (e) => e.phase === "transfer",
    ],
    [
      "decode_memory",
      "Decode HBM admission",
      k.handoff_end_ms,
      k.decode_memory_ready_ms,
      (e) => e.kind === "kv_memory_wait" && e.resource.includes("decode"),
    ],
    ...(!first
      ? [
          [
            "decode",
            "Decode service and queues",
            q.first_token_ms,
            q.finish_ms,
            (e: Obj) => e.phase === "decode",
          ] as (typeof parts)[number],
        ]
      : []),
  ];
  const visibleParts = release
    ? parts.filter(([symbol]) => symbol !== "decode_memory")
    : parts;
  return {
    ...calc(
      `Request #${q.id} ${release ? "prefill KV release" : first ? "first token" : "completion"}`,
      `${release ? "handoff_end" : first ? "first_token" : "finish"} = arrival + batch_wait + memory_wait + restore + prefill + publication + handoff${release ? "" : " + decode_memory"}${first ? "" : " + decode"}`,
      [
        v("arrival", "Request arrival", q.arrival_ms, "ms", () =>
          storageRequestCalculation(r, q, "arrival_ms"),
        ),
        ...visibleParts.map(([symbol, label, begin, end, filter]) =>
          v(symbol, label, end - begin, "ms", () =>
            span(r, q, label, begin, end, filter),
          ),
        ),
      ],
      release ? k.handoff_end_ms : first ? q.first_token_ms : q.finish_ms,
      "ms",
    ),
    summary: `${q.input_tokens} prompt tokens; ${k.reused_tokens} reused, ${k.computed_tokens} computed. Reuse is determined by ready contiguous prefixes, not by a random hit-rate probability. The last prompt token is recomputed for logits.`,
    source: "Executed KV-tier request path",
  };
}
export function cacheDecisionCalculation(r: Obj, q: Obj): Calculation {
  const k = q.cache;
  return {
    ...calc(
      "Prefix reuse decision",
      "Available_prefix = min(matched_blocks × 512, input_tokens−1)\nReused_tokens = min(available_prefix of fixed microbatch members)\nComputed_tokens = input_tokens − Reused_tokens",
      [
        v("matched_blocks", "Ready contiguous prefix blocks", k.matched_blocks),
        v(
          "available_prefix",
          "This request’s restored prefix",
          k.available_reused_tokens ?? k.reused_tokens,
          "tokens",
        ),
        v(
          "batch_prefix_floor",
          "Uniform microbatch prefix floor",
          k.batch_prefix_floor_tokens ?? k.reused_tokens,
          "tokens",
          () =>
            calc(
              "Fixed microbatch prefix floor",
              "Used_prefix = min(restored_prefix_i)",
              Object.entries(
                k.batch_available_prefixes || { [q.id]: k.reused_tokens },
              ).map(([id, n]) =>
                v(
                  "request_" + id,
                  `Request #${id} restored prefix`,
                  Number(n),
                  "tokens",
                ),
              ),
              k.reused_tokens,
              "tokens",
              "A fixed rectangular prefill batch uses one query/context shape. Members with more cached tokens recompute the excess; ragged batching is outside this scheduler.",
            ),
        ),
        v(
          "input_tokens",
          "This request’s prompt length",
          q.input_tokens,
          "tokens",
        ),
        v("hbm_hits", "Blocks from retained HBM", k.hit_tiers.hbm || 0),
        v("dram_hits", "Blocks restored from DRAM", k.hit_tiers.dram || 0),
        v("ssd_hits", "Blocks restored from SSD", k.hit_tiers.ssd || 0),
        v(
          "first_token",
          "Follow this request’s complete path",
          q.first_token_ms,
          "ms",
          () => storageRequestCalculation(r, q, "first_token_ms"),
        ),
      ],
      k.reused_tokens,
      "tokens",
    ),
    summary:
      "A prefix stops at its first missing or unfinished block. Data selected for restore is pinned against eviction until the read finishes; current compute uses the full context and the query suffix after the batch-common restored prefix. Higher-hit members can recompute extra tokens to preserve a uniform fixed batch.",
  };
}
export function storageMetricCalculation(r: Obj, key: string): Calculation {
  const s = r.storage.stats;
  if (key === "hit_fraction")
    return {
      ...calc(
        "Observed simulated prefix hit rate",
        "Hit_fraction = Σ matched_blocks / Σ eligible_full_blocks",
        r.requests.map((q: Obj) =>
          v(
            "request_" + q.id,
            `Request #${q.id}: ${q.cache.matched_blocks}/${q.cache.eligible_blocks} prefix blocks`,
            q.cache.matched_blocks,
            "blocks",
            () => cacheDecisionCalculation(r, q),
          ),
        ),
        s.hit_fraction,
        "fraction",
      ),
      summary:
        "An exact outcome of this simulation’s capacity, LRU state, readiness and request order. It is not sampled from the separately fitted hit-curve forecast.",
    };
  const events = r.timeline.filter(
    (e: Obj) =>
      e.kind === "storage_io" &&
      (key === "ssd_read_bytes"
        ? e.tier === "ssd" && e.operation === "read"
        : key === "ssd_write_bytes"
          ? e.tier === "ssd" && e.operation === "write"
          : key === "ssd_queue_ms"
            ? e.tier === "ssd"
            : false),
  );
  if (key === "reused_tokens")
    return {
      ...calc(
        "Reused prompt tokens",
        "Reused = Σ request reused_tokens",
        r.requests.map((q: Obj) =>
          v(
            "request_" + q.id,
            `Request #${q.id}`,
            q.cache.reused_tokens,
            "tokens",
            () => cacheDecisionCalculation(r, q),
          ),
        ),
        s.reused_tokens || 0,
        "tokens",
      ),
      summary:
        "Saved prompt query work; attention still processes the retained context.",
    };
  return {
    ...calc(
      key,
      key === "ssd_queue_ms"
        ? "Queue_ms = Σ SSD resource waits"
        : "Bytes = Σ executed transfer payload",
      events.map((e: Obj) =>
        v(
          "event_" + e.event_id,
          `Request #${e.request_ids[0]} · SSD ${e.operation}`,
          key === "ssd_queue_ms" ? e.queue_ms : e.payload_bytes,
          key === "ssd_queue_ms" ? "ms" : "bytes",
          () => storageEventCalculation(r, e),
        ),
      ),
      s[key] || 0,
      key === "ssd_queue_ms" ? "ms" : "bytes",
    ),
    summary:
      "Counters are taken from executed storage reservations. Aggregate queue time across requests can exceed wall-clock time.",
  };
}
