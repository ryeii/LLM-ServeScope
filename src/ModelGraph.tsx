import { useState } from "react";
import FormulaValue from "./FormulaValue";
import { operationCalculation, scaled } from "./calculations";
type Obj = Record<string, any>;
const COLORS: Obj = {
  Compute: "#35a58a",
  Memory: "#ed9c55",
  Network: "#6389d0",
};
const f = (n: number) =>
  n.toLocaleString(undefined, { maximumFractionDigits: 2 });
export default function ModelGraph({
  result: r,
  phase,
}: {
  result: Obj;
  phase: string;
}) {
  const [selected, setSelected] = useState("attention");
  const ops = r.phases[phase].operations as Obj[];
  const boundary = ops.findIndex((o) => o.id === "ffn_rmsnorm");
  const positions = new Map(
    ops.map((o, i) => [
      o.id,
      {
        x: i < boundary ? 30 : 405,
        y: 35 + (i < boundary ? i : i - boundary) * 85,
      },
    ]),
  );
  const height = Math.max(boundary, ops.length - boundary) * 85 + 80;
  const cacheY = positions.get("attention")!.y + 32;
  const found = ops.find((o) => o.id === selected) || ops[0];
  const edges = ops
    .slice(1)
    .map((o, i) => ({ a: positions.get(ops[i].id)!, b: positions.get(o.id)! }));
  return (
    <section className="card chart" id="model-graph">
      <div className="card-head">
        <div>
          <h3>Transformer computation graph</h3>
          <p>
            {r.model.name} · one representative block × {r.model.layers} layers
          </p>
        </div>
        <span className="label">BEHAVIORAL DAG</span>
      </div>
      <div className="model-graph-layout">
        <div className="graph-scroll">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox={`0 0 710 ${height}`}
            role="img"
            aria-label="Behavioral transformer computation graph"
          >
            <defs>
              <marker
                id="behavioral-arrow"
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="5"
                markerHeight="5"
                orient="auto"
              >
                <path d="M0 0L10 5L0 10Z" fill="#9aacb9" />
              </marker>
            </defs>
            <rect width="710" height={height} fill="white" />
            {edges.map(({ a, b }, i) => (
              <path
                key={i}
                d={
                  a.x === b.x
                    ? `M${a.x + 115},${a.y + 64} L${b.x + 115},${b.y}`
                    : `M${a.x + 230},${a.y + 32} L365,${a.y + 32} L365,${b.y - 15} L${b.x + 115},${b.y - 15} L${b.x + 115},${b.y}`
                }
                fill="none"
                stroke="#9fb1bd"
                markerEnd="url(#behavioral-arrow)"
              />
            ))}
            <path
              d={`M30,45 L10,45 L10,${positions.get(ops[boundary - 1].id)!.y + 75} L365,${positions.get(ops[boundary - 1].id)!.y + 75} L365,25 L520,25 L520,35`}
              fill="none"
              stroke="#c6d2d9"
              strokeDasharray="4 4"
              markerEnd="url(#behavioral-arrow)"
            />
            <text
              x="280"
              y={height - 14}
              fill="#9aadb8"
              fontSize="8"
              textAnchor="middle"
            >
              Dashed branch: residual connection
            </text>
            <rect
              x="275"
              y={cacheY - 25}
              width="90"
              height="50"
              rx="7"
              fill="#f0f7f3"
              stroke="#bbdace"
            />
            <text
              x="320"
              y={cacheY - 4}
              textAnchor="middle"
              fontSize="9"
              fill="#6b9c8e"
            >
              KV cache
            </text>
            <text
              x="320"
              y={cacheY + 12}
              textAnchor="middle"
              fontSize="8"
              fill="#9ab4a9"
            >
              SRAM / HBM
            </text>
            <path
              d={`M275,${cacheY} L260,${cacheY}`}
              stroke="#8fb9aa"
              markerEnd="url(#behavioral-arrow)"
            />
            {ops.map((o) => {
              const p = positions.get(o.id)!;
              return (
                <g
                  key={o.id}
                  role="button"
                  tabIndex={0}
                  aria-label={"Inspect " + o.name}
                  onClick={() => setSelected(o.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") setSelected(o.id);
                  }}
                  style={{ cursor: "pointer" }}
                >
                  <rect
                    x={p.x}
                    y={p.y}
                    width="230"
                    height="64"
                    rx="8"
                    fill={found.id === o.id ? "#eff8f3" : "#fafcfd"}
                    stroke={found.id === o.id ? "#42a68b" : "#dce6ec"}
                  />
                  <circle
                    cx={p.x + 17}
                    cy={p.y + 23}
                    r="4"
                    fill={COLORS[o.bound]}
                  />
                  <text x={p.x + 29} y={p.y + 27} fill="#4f7485" fontSize="10">
                    {o.name}
                  </text>
                  <text x={p.x + 17} y={p.y + 49} fill="#8ba4b0" fontSize="9">
                    <FormulaValue
                      as="tspan"
                      calculation={scaled(
                        operationCalculation(r, phase, o, "duration_ms"),
                        1000,
                        "µs",
                      )}
                    >
                      {f(o.duration_ms * 1000)} µs
                    </FormulaValue>{" "}
                    · {o.bound.toLowerCase()}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
        <aside className="node-detail">
          <span className="label">SELECTED OPERATION</span>
          <h3>{found.name}</h3>
          <span className={"bound " + found.bound.toLowerCase()}>
            {found.bound} limited
          </span>
          <dl>
            <div>
              <dt>Time / layer / GPU</dt>
              <dd>
                <FormulaValue
                  calculation={scaled(
                    operationCalculation(r, phase, found, "duration_ms"),
                    1000,
                    "µs",
                  )}
                  formulaKey={
                    "graph." + phase + "." + found.id + ".duration_ms"
                  }
                >
                  {f(found.duration_ms * 1000)} µs
                </FormulaValue>
              </dd>
            </div>
            <div>
              <dt>Timing source</dt>
              <dd title={found.fit_key}>{found.timing_source}</dd>
            </div>
            <div>
              <dt>Compute work</dt>
              <dd>
                <FormulaValue
                  calculation={scaled(
                    operationCalculation(r, phase, found, "flops"),
                    1e-9,
                    "GFLOP",
                  )}
                  formulaKey={"graph." + phase + "." + found.id + ".flops"}
                >
                  {f(found.flops / 1e9)} GFLOP
                </FormulaValue>
              </dd>
            </div>
            <div>
              <dt>HBM traffic</dt>
              <dd>
                <FormulaValue
                  calculation={scaled(
                    operationCalculation(r, phase, found, "hbm_bytes"),
                    1e-6,
                    "MB",
                  )}
                  formulaKey={"graph." + phase + "." + found.id + ".hbm_bytes"}
                >
                  {f(found.hbm_bytes / 1e6)} MB
                </FormulaValue>
              </dd>
            </div>
            <div>
              <dt>Arithmetic intensity</dt>
              <dd>
                <FormulaValue
                  calculation={scaled(
                    operationCalculation(
                      r,
                      phase,
                      found,
                      "arithmetic_intensity",
                    ),
                    1,
                    "FLOP/B",
                  )}
                  formulaKey={
                    "graph." + phase + "." + found.id + ".arithmetic_intensity"
                  }
                >
                  {f(found.arithmetic_intensity)} FLOP/B
                </FormulaValue>
              </dd>
            </div>
          </dl>
          <p>
            Edges follow transformer data flow. Each operation uses the fitted
            service-time function and matches the layer table. Shared embedding,
            head and sampling costs are charged separately in the timeline.
            Native graphs are available in the LLMServingSim tab.
          </p>
        </aside>
      </div>
      <button
        className="text-button"
        onClick={() => {
          const svg = document.querySelector("#model-graph svg");
          if (!svg) return;
          const url = URL.createObjectURL(
            new Blob([new XMLSerializer().serializeToString(svg)], {
              type: "image/svg+xml",
            }),
          );
          const a = document.createElement("a");
          a.href = url;
          a.download = "computation-graph.svg";
          a.click();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        }}
      >
        Download computation graph SVG
      </button>
    </section>
  );
}
