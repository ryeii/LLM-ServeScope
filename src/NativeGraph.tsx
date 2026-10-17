import { useState } from "react";
type Obj = Record<string, any>;
export default function NativeGraph({ graph }: { graph: Obj }) {
  const [selected, setSelected] = useState<Obj | null>(null);
  const [limit, setLimit] = useState(48);
  const all = graph.nodes as Obj[],
    edges = graph.edges as Obj[];
  const incoming = new Map(all.map((n) => [n.id, 0]));
  for (const e of edges)
    incoming.set(e.target, (incoming.get(e.target) || 0) + 1);
  const queue = all.filter((n) => incoming.get(n.id) === 0),
    ordered: Obj[] = [];
  while (queue.length) {
    const n = queue.shift()!;
    ordered.push(n);
    for (const e of edges.filter((e) => e.source === n.id)) {
      incoming.set(e.target, incoming.get(e.target)! - 1);
      if (incoming.get(e.target) === 0) {
        const child = all.find((c) => c.id === e.target);
        if (child) queue.push(child);
      }
    }
  }
  const nodes = (ordered.length === all.length ? ordered : all).slice(0, limit);
  const pos = new Map(
    nodes.map((n, i) => [
      n.id,
      { x: 18 + (i % 4) * 220, y: 25 + Math.floor(i / 4) * 105 },
    ]),
  );
  const visible = edges.filter((e) => pos.has(e.source) && pos.has(e.target));
  return (
    <div className="native-dag">
      <div className="native-dag-tools">
        <span>
          {nodes.length} of {all.length} nodes · {visible.length} visible
          dependency edges
        </span>
        <select
          aria-label="Native graph node limit"
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
        >
          <option value={48}>48 nodes</option>
          <option value={120}>120 nodes</option>
          <option value={1000}>All nodes</option>
        </select>
      </div>
      <div className="native-dag-scroll">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox={`0 0 910 ${Math.ceil(nodes.length / 4) * 105 + 30}`}
          role="img"
          aria-label="Actual native Chakra computation DAG"
        >
          <defs>
            <marker
              id="native-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto"
            >
              <path d="M0 0L10 5L0 10Z" fill="#93a9b7" />
            </marker>
          </defs>
          <rect width="910" height="100%" fill="#fbfdfe" />
          {visible.map((e, i) => {
            const a = pos.get(e.source)!,
              b = pos.get(e.target)!;
            const same = a.y === b.y;
            return (
              <path
                key={i}
                d={
                  same
                    ? `M${a.x + 194},${a.y + 32} L${b.x},${b.y + 32}`
                    : `M${a.x + 97},${a.y + 66} C${a.x + 97},${a.y + 88} ${b.x + 97},${b.y - 20} ${b.x + 97},${b.y}`
                }
                fill="none"
                stroke={e.kind === "control" ? "#c5d2db" : "#9fb9b0"}
                strokeDasharray={e.kind === "control" ? "3 3" : undefined}
                strokeWidth="1"
                markerEnd="url(#native-arrow)"
              />
            );
          })}
          {nodes.map((n) => {
            const p = pos.get(n.id)!;
            const color = n.type.includes("COMM")
              ? "#7896c8"
              : n.type.includes("MEM")
                ? "#d4a06d"
                : "#5eab95";
            return (
              <g
                key={n.id}
                role="button"
                tabIndex={0}
                aria-label={`Native node ${n.id} ${n.name}`}
                onClick={() => setSelected(n)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") setSelected(n);
                }}
                style={{ cursor: "pointer" }}
              >
                <rect
                  x={p.x}
                  y={p.y}
                  width="194"
                  height="66"
                  rx="7"
                  fill={selected?.id === n.id ? "#eff7f3" : "white"}
                  stroke={selected?.id === n.id ? "#40a181" : "#dce6ec"}
                />
                <rect
                  x={p.x}
                  y={p.y + 10}
                  width="3"
                  height="46"
                  rx="2"
                  fill={color}
                />
                <text x={p.x + 12} y={p.y + 17} fontSize="7" fill="#98aebc">
                  {n.id} · {n.type}
                </text>
                <text x={p.x + 12} y={p.y + 35} fontSize="10" fill="#547889">
                  {n.name.length > 27 ? n.name.slice(0, 26) + "…" : n.name}
                </text>
                <text x={p.x + 12} y={p.y + 53} fontSize="8" fill="#8ba4b1">
                  {(n.profile_cost_ns / 1000).toFixed(2)} µs profile cost
                </text>
                <title>{n.name}</title>
              </g>
            );
          })}
        </svg>
      </div>
      {selected && (
        <pre className="source-json">{JSON.stringify(selected, null, 2)}</pre>
      )}
      <p className="fine">
        Edges are actual data/control dependencies from the native Chakra file.
        Nodes are topologically ordered for display; positions do not represent
        execution timing.
      </p>
    </div>
  );
}
