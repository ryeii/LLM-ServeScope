import FormulaValue from "./FormulaValue";
import { calc, v, scaled } from "./calculations";
import { eventService } from "./derivations";
import { storageEventCalculation } from "./storageCalculations";
type Obj = Record<string, any>;

/** Aggregate resource occupancy, not a sum along one request's critical path. */
export default function ResourceService({ r }: { r: Obj }) {
  const groups: Record<string, { event: Obj; value: number }[]> = {
    "GPU operators + collectives": [],
    "SSD ↔ GPU · GDS": [],
    "SSD ↔ host": [],
    "Host ↔ GPU": [],
    "HBM cache copies": [],
    "Pipeline + KV handoff": [],
  };
  let queue = 0;
  for (const e of r.timeline as Obj[]) {
    if (e.kind === "kv_memory_wait") {
      queue += e.duration_ms;
      continue;
    }
    queue += e.queue_ms || 0;
    let key: string, value: number;
    if (e.kind === "storage_io") {
      key = e.gds
        ? "SSD ↔ GPU · GDS"
        : e.tier === "ssd"
          ? "SSD ↔ host"
          : e.tier === "pcie"
            ? "Host ↔ GPU"
            : "HBM cache copies";
      value = e.duration_ms;
    } else if (e.kind === "pipeline_transfer" || e.phase === "transfer") {
      key = "Pipeline + KV handoff";
      value = e.active_wire_ms ?? e.duration_ms;
    } else if (e.shape_id) {
      key = "GPU operators + collectives";
      value =
        (e.gpu_service_ms ?? e.duration_ms) *
        r.config[e.phase].tp *
        r.config[e.phase].ep;
    } else continue;
    groups[key].push({ event: e, value });
  }
  const rows = Object.entries(groups).map(([key, events]) => ({
    key,
    events,
    value: events.reduce((sum, e) => sum + e.value, 0),
  }));
  const max = Math.max(1, ...rows.map((row) => row.value));
  return (
    <section
      className="card resource-service"
      aria-label="Executed resource service"
    >
      <div className="card-head">
        <div>
          <h3>Executed resource service</h3>
          <p>Aggregate resource time, including storage and transfers</p>
        </div>
      </div>
      {rows.map(({ key, value, events }) => (
        <div className="work-row" key={key}>
          <div>
            <span>{key}</span>
            <FormulaValue
              formulaKey={"service." + key}
              calculation={{
                ...calc(
                  key,
                  "Aggregate_resource_ms = Σ executed service_ms × resource_count",
                  events.map(({ event: e, value: n }) =>
                    v(
                      "event_" + e.event_id,
                      `Event #${e.event_id} · ${e.resource}`,
                      n,
                      "resource·ms",
                      () =>
                        e.kind === "storage_io"
                          ? storageEventCalculation(r, e, true)
                          : e.shape_id
                            ? scaled(
                                eventService(r, {
                                  ...e,
                                  active_io_event_ids: undefined,
                                  duration_ms:
                                    e.gpu_service_ms ?? e.duration_ms,
                                }),
                                r.config[e.phase].tp * r.config[e.phase].ep,
                                "resource·ms",
                              )
                            : e.active_wire_ms !== undefined
                              ? calc(
                                  "KV handoff wire service",
                                  "Wire_ms = bytes / (bandwidth_GBps × 10⁶) + latency_us/1000",
                                  [
                                    v(
                                      "bytes",
                                      "KV payload",
                                      e.payload_bytes,
                                      "bytes",
                                    ),
                                    v(
                                      "bandwidth_GBps",
                                      "Handoff bandwidth",
                                      e.link_bandwidth_GBps,
                                      "GB/s",
                                    ),
                                    v(
                                      "latency_us",
                                      "Handoff startup",
                                      e.link_latency_us,
                                      "µs",
                                    ),
                                  ],
                                  e.active_wire_ms,
                                  "ms",
                                  "Nested SSD and staging reservations are counted in their own rows, without adding the parent span again.",
                                )
                              : eventService(r, e),
                    ),
                  ),
                  value,
                  "resource·ms",
                ),
                summary:
                  "GPU service is multiplied by TP × EP and includes fitted operator timing plus collectives. Explicit storage/copy service is counted once; GDS combines storage and direct-link service. Active-offload parent stalls are excluded from GPU service. Background publication is included even if it finishes after the last response. Different resources can overlap, so these bars do not add up to request latency.",
              }}
            >
              <b>
                {value.toLocaleString("en-US", { maximumFractionDigits: 2 })}{" "}
                resource·ms
              </b>
            </FormulaValue>
          </div>
          <div className="work-track">
            <i
              style={{
                width: `${(value / max) * 100}%`,
                background: key.includes("SSD") ? "#a16a26" : "#2c796d",
              }}
            />
          </div>
        </div>
      ))}
      <p className="storage-note">
        Resource and memory-admission waits:{" "}
        <FormulaValue
          formulaKey="service.queue"
          calculation={calc(
            "Recorded reservation waits",
            "Wait_ms = Σ resource_queue_ms + Σ memory_admission_wait_ms",
            r.timeline
              .filter((e: Obj) => e.queue_ms > 0 || e.kind === "kv_memory_wait")
              .map((e: Obj) =>
                v(
                  "wait_" + e.event_id,
                  `Event #${e.event_id} · ${e.resource}`,
                  e.kind === "kv_memory_wait" ? e.duration_ms : e.queue_ms,
                  "ms",
                ),
              ),
            queue,
            "ms",
            "Aggregate across recorded reservations, not the critical path of one request. Publication-worker/placement waits are visible in publication records; this is not a complete latency decomposition.",
          )}
        >
          {queue.toLocaleString("en-US", { maximumFractionDigits: 2 })}
        </FormulaValue>{" "}
        ms across reservations. Waits are separate from service. Parallel
        resource times are not request latency.
      </p>
    </section>
  );
}
