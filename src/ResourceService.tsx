import FormulaValue from "./FormulaValue";
import { calc, v, scaled } from "./calculations";
import { eventService } from "./derivations";
type Obj = Record<string, any>;

/** Aggregate resource occupancy, not a sum along one request's critical path. */
export default function ResourceService({ r }: { r: Obj }) {
  const groups: Record<string, { event: Obj; value: number }[]> = {
    "GPU operators + collectives": [],
    "Pipeline + KV handoff": [],
  };
  let queue = 0;
  for (const e of r.timeline as Obj[]) {
    queue += e.queue_ms || 0;
    let key: string, value: number;
    if (e.kind === "pipeline_transfer" || e.phase === "transfer") {
      key = "Pipeline + KV handoff";
      value = e.duration_ms;
    } else if (e.shape_id) {
      key = "GPU operators + collectives";
      value = e.duration_ms * r.config[e.phase].tp * r.config[e.phase].ep;
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
          <p>Aggregate GPU and inter-GPU transfer service</p>
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
                        e.shape_id
                          ? scaled(
                              eventService(r, {
                                ...e,
                                duration_ms: e.duration_ms,
                              }),
                              r.config[e.phase].tp * r.config[e.phase].ep,
                              "resource·ms",
                            )
                          : eventService(r, e),
                    ),
                  ),
                  value,
                  "resource·ms",
                ),
                summary:
                  "GPU service is multiplied by TP × EP and includes fitted operator timing plus analytical collectives. Pipeline and full-context KV handoff use configured network estimates. Different resources can overlap; these bars do not add up to request latency.",
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
                background: "#2c796d",
              }}
            />
          </div>
        </div>
      ))}
      <p className="resource-note">
        Resource queue waits:{" "}
        <FormulaValue
          formulaKey="service.queue"
          calculation={calc(
            "Recorded reservation waits",
            "Wait_ms = Σ resource_queue_ms",
            r.timeline
              .filter((e: Obj) => e.queue_ms > 0)
              .map((e: Obj) =>
                v(
                  "wait_" + e.event_id,
                  `Event #${e.event_id} · ${e.resource}`,
                  e.queue_ms,
                  "ms",
                ),
              ),
            queue,
            "ms",
            "Aggregate across recorded resource reservations; follow a request’s timestamps for its latency decomposition.",
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
