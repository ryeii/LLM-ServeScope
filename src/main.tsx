import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  CircleHelp,
  Clock3,
  Cpu,
  Database,
  Download,
  ExternalLink,
  FileJson,
  FlaskConical,
  GitBranch,
  Layers3,
  LoaderCircle,
  Network,
  Pin,
  Play,
  Plus,
  RotateCcw,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Upload,
  X,
  Zap,
} from "lucide-react";
import "./styles.css";
import NativeGraph from "./NativeGraph";
import BehavioralGraph from "./ModelGraph";
type Obj = Record<string, any>;
const colors = {
  Compute: "#35a58a",
  Memory: "#ed9c55",
  Network: "#6389d0",
  Launch: "#a7b0bb",
};
const fmt = (n: number, d = 1) =>
  Number.isFinite(n)
    ? n.toLocaleString(undefined, { maximumFractionDigits: d })
    : "—";
const clone = (o: any) => structuredClone(o);
async function api(path: string, body?: any) {
  const r = await fetch(
    "/api/" + path,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {},
  );
  const data = await r.json();
  if (!r.ok) throw Error(data.error || "Request failed");
  return data;
}
function download(name: string, data: any, type = "application/json") {
  const url = URL.createObjectURL(
    new Blob(
      [typeof data === "string" ? data : JSON.stringify(data, null, 2)],
      { type },
    ),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function saveSvg(id: string) {
  const el = document.querySelector("#" + id + " svg");
  if (el)
    download(
      id + ".svg",
      new XMLSerializer().serializeToString(el),
      "image/svg+xml",
    );
}
const tabs = [
  ["overview", "Overview"],
  ["roofline", "Roofline"],
  ["layers", "Layer / chunk"],
  ["boxes", "Box graph"],
  ["graph", "Model graph"],
  ["configuration", "Configuration"],
  ["native", "LLMServingSim 2.0"],
];
function Field({
  label,
  value,
  onChange,
  step = 1,
  min = 0,
  max,
  unit = "",
  tip = "",
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  step?: number;
  min?: number;
  max?: number;
  unit?: string;
  tip?: string;
}) {
  return (
    <label className="field" title={tip}>
      <span>{label}</span>
      <div>
        <input
          type="number"
          aria-label={label}
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) =>
            onChange(e.target.value === "" ? 0 : Number(e.target.value))
          }
        />
        {unit && <small>{unit}</small>}
      </div>
    </label>
  );
}
function Section({
  title,
  children,
  initial = false,
}: {
  title: string;
  children: React.ReactNode;
  initial?: boolean;
}) {
  return (
    <details className="config-section" open={initial}>
      <summary>
        {title}
        <ChevronDown size={14} />
      </summary>
      <div className="fields">{children}</div>
    </details>
  );
}
function Chart({
  id,
  title,
  note,
  children,
}: {
  id: string;
  title: string;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="card chart" id={id}>
      <div className="card-head">
        <div>
          <h3>{title}</h3>
          {note && <p>{note}</p>}
        </div>
        <button
          className="icon-button"
          aria-label={"Download " + title + " SVG"}
          title="Download SVG"
          onClick={() => saveSvg(id)}
        >
          <Download size={15} />
        </button>
      </div>
      {children}
    </section>
  );
}
function Legend() {
  return (
    <div className="legend">
      {Object.entries(colors).map(([name, c]) => (
        <span key={name}>
          <i style={{ background: c }} />
          {name}
        </span>
      ))}
    </div>
  );
}
function Kpi({
  label,
  value,
  unit,
  note,
  icon,
  delta,
}: {
  label: string;
  value: string;
  unit: string;
  note: string;
  icon: React.ReactNode;
  delta?: string;
}) {
  return (
    <div className="kpi">
      <div className="kpi-label">
        {label}
        {icon}
      </div>
      <div className="kpi-value">
        {value}
        <small>{unit}</small>
      </div>
      <div className="kpi-note">{delta ? <b>{delta} vs pinned</b> : note}</div>
    </div>
  );
}
function Pipeline({ r }: { r: Obj }) {
  const c = r.config;
  return (
    <div className="pipeline">
      <div className="pipeline-title">
        <span className="label">SERVING DESIGN</span>
        <span>
          {c.separated
            ? "Disaggregated prefill + decode"
            : "Colocated serving pool"}
        </span>
      </div>
      <div className="pipeline-flow">
        <div className="pool">
          <span className="pool-icon">
            <Cpu size={20} />
          </span>
          <div>
            <h4>Prefill pool</h4>
            <p>
              {c.prefill.hardware.name}
              {c.prefill.hardware_modified ? " · edited" : ""}
            </p>
            <span>
              {c.prefill.tp} TP · {c.prefill.pp} PP · {c.prefill.ep} EP ·{" "}
              {c.prefill.replicas} replica{c.prefill.replicas > 1 ? "s" : ""}
              <br />
              Batch {c.prefill.batch_size} ·{" "}
              {c.prefill.engine === "inherit"
                ? c.inference_engine
                : c.prefill.engine}
            </span>
          </div>
          <b>
            {r.memory.prefill.gpus}
            <small>GPUs</small>
          </b>
        </div>
        <div className="handoff">
          <ArrowRight size={20} />
          <span>{c.separated ? "KV handoff" : "Shared KV"}</span>
          <small>
            {c.separated
              ? `${fmt(c.network.spine_bw_GBps)} GB/s × ${c.network.handoff_channels} link${c.network.handoff_channels > 1 ? "s" : ""}`
              : "No inter-pool copy"}
          </small>
        </div>
        <div className="pool decode">
          <span className="pool-icon">
            <Layers3 size={20} />
          </span>
          <div>
            <h4>Decode pool</h4>
            <p>
              {c.decode.hardware.name}
              {c.decode.hardware_modified ? " · edited" : ""}
            </p>
            <span>
              {c.decode.tp} TP · {c.decode.pp} PP · {c.decode.ep} EP ·{" "}
              {c.decode.replicas} replica{c.decode.replicas > 1 ? "s" : ""}
              <br />
              Batch cap {c.decode.batch_size} ·{" "}
              {c.decode.engine === "inherit"
                ? c.inference_engine
                : c.decode.engine}
            </span>
          </div>
          <b>
            {c.separated ? r.memory.decode.gpus : "↳"}
            <small>{c.separated ? "GPUs" : "shared"}</small>
          </b>
        </div>
      </div>
    </div>
  );
}
function MemoryChart({ r }: { r: Obj }) {
  return (
    <Chart
      id="memory-chart"
      title="Memory budget"
      note="Resident payload vs usable capacity per GPU, at peak context"
    >
      <svg
        viewBox="0 0 560 200"
        role="img"
        aria-label="Prefill and decode per-GPU memory capacity"
      >
        <rect width="560" height="200" fill="white" />
        {["prefill", "decode"].map((ph, i) => {
          const m = r.memory[ph],
            y = 42 + i * 72;
          let x = 100;
          const scale = 375 / m.capacity_GB;
          return (
            <g key={ph}>
              <text x="20" y={y + 17} fill="#607380" fontSize="12">
                {ph === "prefill" ? "Prefill" : "Decode"}
              </text>
              <rect
                x="100"
                y={y}
                width="375"
                height="28"
                rx="4"
                fill="#eff3f5"
              />
              {[
                ["weights_GB", "#738bd2"],
                ["kv_GB", "#34a58a"],
                ["activation_GB", "#e8a15d"],
              ].map(([key, color]) => {
                const width = m[key] * scale;
                const el = (
                  <rect
                    key={key}
                    x={x}
                    y={y}
                    width={width}
                    height="28"
                    fill={color}
                  />
                );
                x += width;
                return el;
              })}
              <text x="488" y={y + 18} fill="#18394c" fontSize="12">
                {fmt(m.total_GB)}/{fmt(m.capacity_GB)} GB
              </text>
            </g>
          );
        })}
        {[
          [100, "Weights", "#738bd2"],
          [190, "KV cache", "#34a58a"],
          [285, "Activations", "#e8a15d"],
          [395, "Headroom", "#eff3f5"],
        ].map(([xx, label, color]) => (
          <g key={String(label)}>
            <rect
              x={Number(xx)}
              y="177"
              width="6"
              height="6"
              rx="1"
              fill={String(color)}
            />
            <text x={Number(xx) + 12} y="184" fill="#84959e" fontSize="9">
              {label}
            </text>
          </g>
        ))}
      </svg>
      <div className="subnote">
        KV remains resident in HBM; SRAM reduces modeled read traffic. Runtime
        reservations reduce usable capacity; allocator fragmentation is
        excluded.
      </div>
    </Chart>
  );
}
function Distribution({ requests }: { requests: Obj[] }) {
  const fields = [
    ["ttft_ms", "Time to first token"],
    ["tpot_ms", "Time per output token"],
    ["latency_ms", "End-to-end latency"],
  ];
  const quant = (a: number[], p: number) => {
    const x = (a.length - 1) * p;
    return (
      a[Math.floor(x)] +
      (a[Math.ceil(x)] - a[Math.floor(x)]) * (x - Math.floor(x))
    );
  };
  const max = Math.max(1, ...requests.map((x) => x.latency_ms));
  return (
    <Chart
      id="latency-distribution"
      title="Latency distribution"
      note="Min / 25th / median / 75th / max; one observation per request"
    >
      <svg
        viewBox="0 0 650 250"
        role="img"
        aria-label="Latency box and whisker plots"
      >
        <rect width="650" height="250" fill="white" />
        {fields.map(([key, name], i) => {
          const a = requests.map((r) => r[key]).sort((a, b) => a - b),
            y = 42 + i * 62;
          const x = (v: number) => 165 + (v / max) * 420;
          return (
            <g key={key}>
              <text x="18" y={y + 5} fill="#627988" fontSize="11">
                {name}
              </text>
              <line
                x1={x(a[0])}
                x2={x(a[a.length - 1])}
                y1={y}
                y2={y}
                stroke="#71899a"
              />
              {[a[0], a.at(-1)!].map((v, k) => (
                <line
                  key={k}
                  x1={x(v)}
                  x2={x(v)}
                  y1={y - 8}
                  y2={y + 8}
                  stroke="#71899a"
                />
              ))}
              <rect
                x={x(quant(a, 0.25))}
                y={y - 13}
                width={Math.max(2, x(quant(a, 0.75)) - x(quant(a, 0.25)))}
                height="26"
                rx="3"
                fill="#dcefe9"
                stroke="#57a78f"
              />
              <line
                x1={x(quant(a, 0.5))}
                x2={x(quant(a, 0.5))}
                y1={y - 13}
                y2={y + 13}
                stroke="#19826b"
                strokeWidth="2"
              />
              <text
                x={Math.min(590, x(a.at(-1)!) + 8)}
                y={y + 4}
                fill="#496577"
                fontSize="10"
              >
                {fmt(a.at(-1)!, 0)}
              </text>
            </g>
          );
        })}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1={165 + t * 420}
              x2={165 + t * 420}
              y1="218"
              y2="224"
              stroke="#b7c4cd"
            />
            <text
              x={165 + t * 420}
              y="242"
              textAnchor="middle"
              fontSize="10"
              fill="#84959e"
            >
              {fmt(max * t, 0)} ms
            </text>
          </g>
        ))}
      </svg>
    </Chart>
  );
}
function Roofline({ r, phase }: { r: Obj; phase: string }) {
  const [focus, setFocus] = useState<string | null>(null);
  const hw = r.config[phase].hardware;
  const peak = r.phases[phase].detail.peak_tflops * hw.compute_efficiency;
  const bw = hw.dram_bandwidth_TBps * hw.memory_efficiency;
  const x = (v: number) => 64 + ((Math.log10(Math.max(0.1, v)) + 1) / 6) * 620;
  const y = (v: number) =>
    315 - ((Math.log10(Math.max(0.001, v)) + 3) / 7) * 275;
  const ops = r.phases[phase].operations.filter((o: Obj) => o.flops > 0);
  const path = Array.from({ length: 121 }, (_, i) => {
    const ai = 10 ** (-1 + i / 20);
    return `${i ? "L" : "M"}${x(ai)},${y(Math.min(peak, bw * ai))}`;
  }).join(" ");
  return (
    <Chart
      id="roofline-chart"
      title={`${phase === "prefill" ? "Prefill" : "Decode"} performance roofline`}
      note="Per-GPU operation estimates; decimal TFLOP/s and bytes"
    >
      <svg
        viewBox="0 0 740 380"
        role="img"
        aria-label="Compute and memory roofline with operation markers"
      >
        <rect width="740" height="380" fill="white" />
        {[-1, 0, 1, 2, 3, 4, 5].map((t) => (
          <g key={t}>
            <line
              x1={x(10 ** t)}
              x2={x(10 ** t)}
              y1="38"
              y2="315"
              stroke="#edf1f4"
            />
            <text
              x={x(10 ** t)}
              y="337"
              textAnchor="middle"
              fontSize="10"
              fill="#81939e"
            >
              {fmt(10 ** t, 1)}
            </text>
          </g>
        ))}
        {[-3, -2, -1, 0, 1, 2, 3, 4].map((t) => (
          <g key={t}>
            <line
              x1="64"
              x2="684"
              y1={y(10 ** t)}
              y2={y(10 ** t)}
              stroke="#edf1f4"
            />
            <text
              x="54"
              y={y(10 ** t) + 4}
              textAnchor="end"
              fontSize="10"
              fill="#81939e"
            >
              {fmt(10 ** t, 3)}
            </text>
          </g>
        ))}
        <path d={path} fill="none" stroke="#2c9d82" strokeWidth="3" />
        <path d={`${path} L684,315 L64,315 Z`} fill="#35a58a" opacity=".04" />
        {ops.map((o: Obj) => (
          <g key={o.id}>
            <circle
              cx={x(o.arithmetic_intensity)}
              cy={y(o.achieved_tflops)}
              r={focus === o.id ? 7 : 5}
              fill={colors[o.bound as keyof typeof colors]}
              stroke={focus === o.id ? "#234e5f" : "white"}
              strokeWidth={focus === o.id ? 1.5 : 2}
              opacity={focus && focus !== o.id ? 0.35 : 1}
            >
              <title>
                {o.name}: {fmt(o.achieved_tflops, 3)} TFLOP/s;{" "}
                {fmt(o.arithmetic_intensity)} FLOP/B
              </title>
            </circle>
          </g>
        ))}
        <text x="375" y="372" textAnchor="middle" fontSize="11" fill="#708694">
          Arithmetic intensity · FLOP / HBM byte
        </text>
        <text
          x="15"
          y="180"
          transform="rotate(-90 15 180)"
          textAnchor="middle"
          fontSize="11"
          fill="#708694"
        >
          Attained TFLOP/s (log)
        </text>
        <text
          x="675"
          y={Math.max(28, y(peak) - 10)}
          textAnchor="end"
          fontSize="10"
          fill="#218b71"
        >
          Effective tensor ceiling {fmt(peak)} TFLOP/s
        </text>
      </svg>
      <div className="roofline-legend">
        {ops.map((o: Obj) => (
          <button
            key={o.id}
            className={focus === o.id ? "active" : ""}
            onMouseEnter={() => setFocus(o.id)}
            onFocus={() => setFocus(o.id)}
            onClick={() => setFocus(o.id)}
          >
            <i style={{ background: colors[o.bound as keyof typeof colors] }} />
            {o.name}
            <span>{fmt(o.achieved_tflops, 3)} TFLOP/s</span>
          </button>
        ))}
      </div>
      <div className="subnote">
        Roof = min(effective compute peak, HBM bandwidth × arithmetic
        intensity). Vector operations use FP32 throughput; launch and tiling
        costs can place them below the tensor roof.
      </div>
    </Chart>
  );
}
function BoxGraph({ r }: { r: Obj }) {
  const events = r.timeline;
  const resources = [
    ...new Set(events.map((e: Obj) => e.resource)),
  ] as string[];
  const max = r.metrics.elapsed_ms;
  const visible =
    events.length > 1200
      ? events.filter(
          (_: Obj, i: number) => i % Math.ceil(events.length / 1200) === 0,
        )
      : events;
  return (
    <Chart
      id="box-graph"
      title="Execution box graph"
      note={`SimPy stage reservations · ${events.length.toLocaleString()} intervals${visible.length < events.length ? " · sampled for display; export retains all" : ""}`}
    >
      <div className="graph-scroll">
        <svg
          viewBox={`0 0 1000 ${Math.max(200, resources.length * 46 + 80)}`}
          style={{ minWidth: 780 }}
          role="img"
          aria-label="Pipeline resource timeline with prefill, decode and transfer boxes"
        >
          <rect width="1000" height="100%" fill="white" />
          {resources.map((res, i) => (
            <g key={res}>
              <text x="12" y={48 + i * 46} fill="#5d7483" fontSize="11">
                {res}
              </text>
              <line
                x1="160"
                x2="972"
                y1={60 + i * 46}
                y2={60 + i * 46}
                stroke="#eef2f5"
              />
            </g>
          ))}
          {visible.map((e: Obj, i: number) => (
            <rect
              key={i}
              x={160 + (e.start_ms / max) * 800}
              y={30 + resources.indexOf(e.resource) * 46}
              width={Math.max(1, (e.duration_ms / max) * 800)}
              height="26"
              rx="3"
              fill={
                e.phase === "prefill"
                  ? "#35a58a"
                  : e.phase === "decode"
                    ? "#6389d0"
                    : "#ed9c55"
              }
              opacity=".8"
            >
              <title>
                {e.phase} · batch {e.batch_id + 1} · {fmt(e.start_ms)}–
                {fmt(e.end_ms)} ms · {e.batch_size} requests
              </title>
            </rect>
          ))}
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <text
              key={v}
              x={160 + v * 800}
              y={resources.length * 46 + 68}
              textAnchor="middle"
              fill="#8b9ca6"
              fontSize="11"
            >
              {fmt(v * max, 0)} ms
            </text>
          ))}
        </svg>
      </div>
      <div className="legend">
        <span>
          <i style={{ background: "#35a58a" }} />
          Prefill
        </span>
        <span>
          <i style={{ background: "#6389d0" }} />
          Decode
        </span>
        <span>
          <i style={{ background: "#ed9c55" }} />
          KV handoff
        </span>
      </div>
      <div className="subnote">
        Boxes show modeled resource occupancy, including memory and
        communication stalls. Hover to inspect an interval.
      </div>
    </Chart>
  );
}
function Native({
  health,
  onRefresh,
}: {
  health: Obj | null;
  onRefresh: () => void;
}) {
  const [form, setForm] = useState({
    requests: 4,
    input_tokens: 128,
    output_tokens: 8,
    tp: 1,
    hardware: "RTXPRO6000",
    memory_gib: 96,
    memory_bandwidth_gbps: 1597,
    link_bandwidth_gibps: 16,
    link_latency_ns: 1000,
  });
  const [job, setJob] = useState<Obj | null>(null);
  const [error, setError] = useState("");
  const [graphPhase, setGraphPhase] = useState("First batch");
  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = setTimeout(
      () =>
        api("jobs/" + job.id)
          .then(setJob)
          .catch((e) => setError(e.message)),
      1000,
    );
    return () => clearTimeout(timer);
  }, [job]);
  async function run() {
    setError("");
    try {
      const j = await api("native/run", form);
      setJob(j);
    } catch (e: any) {
      setError(e.message);
    }
  }
  const ready = health?.native?.available;
  const r = job?.result;
  const graphs = Array.isArray(r?.graphs)
    ? Object.fromEntries(
        r.graphs.map((g: Obj, i: number) => [
          i === 0 ? "First batch" : "Next batch",
          g,
        ]),
      )
    : r?.graphs || {};
  const graph =
    graphs[graphPhase] || (Object.values(graphs)[0] as Obj | undefined);
  return (
    <div className="native-page">
      <div className="native-intro">
        <div className="engine-icon">
          <FlaskConical size={26} />
        </div>
        <div>
          <div className="label">PROFILE-DRIVEN REFERENCE ENGINE</div>
          <h2>LLMServingSim 2.0</h2>
          <p>
            Execute its native scheduler, Chakra graph generator and ASTRA-Sim
            analytical backend on this machine. Uses published operation
            profiles; no GPU or model weights required.
          </p>
        </div>
        <span className={"status-pill " + (ready ? "ready" : "")}>
          <i />
          {ready ? "Native backend ready" : "Backend preparing"}
        </span>
      </div>
      <div className="native-layout">
        <section className="card native-form">
          <h3>Native experiment</h3>
          <label className="field">
            <span>Hardware profile</span>
            <select
              aria-label="Native hardware"
              value={form.hardware}
              onChange={(e) =>
                setForm({
                  ...form,
                  hardware: e.target.value,
                  tp: 1,
                  memory_gib: e.target.value === "RTX4090" ? 24 : 96,
                  memory_bandwidth_gbps:
                    e.target.value === "RTX4090" ? 1008 : 1597,
                })
              }
            >
              <option>RTXPRO6000</option>
              <option>RTX4090</option>
            </select>
          </label>
          <div className="read-only">
            <span>Model</span>
            <b>Llama 3.1 8B · BF16</b>
          </div>
          <Field
            label="Native requests"
            value={form.requests}
            min={1}
            max={8}
            onChange={(n) => setForm({ ...form, requests: n })}
          />
          <Field
            label="Native input tokens"
            value={form.input_tokens}
            min={1}
            max={2048}
            onChange={(n) => setForm({ ...form, input_tokens: n })}
          />
          <Field
            label="Native output tokens"
            value={form.output_tokens}
            min={2}
            max={32}
            onChange={(n) => setForm({ ...form, output_tokens: n })}
          />
          <label className="field">
            <span>Tensor parallel</span>
            <select
              aria-label="Native tensor parallel"
              value={form.tp}
              onChange={(e) => setForm({ ...form, tp: Number(e.target.value) })}
            >
              <option value={1}>1 GPU</option>
              {form.hardware === "RTXPRO6000" && (
                <option value={2}>2 GPUs</option>
              )}
            </select>
          </label>
          <Field
            label="Native memory capacity"
            value={form.memory_gib}
            min={1}
            unit="GiB"
            onChange={(n) => setForm({ ...form, memory_gib: n })}
          />
          <Field
            label="Native memory bandwidth"
            value={form.memory_bandwidth_gbps}
            min={1}
            unit="GB/s"
            onChange={(n) => setForm({ ...form, memory_bandwidth_gbps: n })}
          />
          <Field
            label="Native link bandwidth"
            value={form.link_bandwidth_gibps}
            min={0.01}
            step={1}
            unit="GiB/s"
            onChange={(n) => setForm({ ...form, link_bandwidth_gibps: n })}
          />
          <Field
            label="Native link latency"
            value={form.link_latency_ns}
            min={0}
            unit="ns"
            onChange={(n) => setForm({ ...form, link_latency_ns: n })}
          />
          <p className="fine">
            Changing memory or link specifications preserves the recorded kernel
            profiles. It does not reprofile compute.
          </p>
          <button
            className="primary"
            onClick={run}
            disabled={!ready || ["queued", "running"].includes(job?.status)}
          >
            {["queued", "running"].includes(job?.status) ? (
              <LoaderCircle size={16} className="spin" />
            ) : (
              <Play size={16} />
            )}
            Run native simulator
          </button>
          <button className="text-button" onClick={onRefresh}>
            <RotateCcw size={13} />
            Refresh engine status
          </button>
          <p className="fine">
            Length-only workload with deterministic token identity markers. This
            run models serving behavior; it does not generate model text.
          </p>
          {error && (
            <div role="alert" className="inline-error">
              {error}
            </div>
          )}
        </section>
        <section className="card native-result">
          {!job ? (
            <div className="empty-native">
              <GitBranch size={44} />
              <h3>No simulation results</h3>
              <p>
                Run the small reference experiment to inspect native request
                metrics and its actual computation DAG. Behavioral hardware
                edits on the other tabs do not change these recorded profiles.
              </p>
              <div className="native-steps">
                <span>Profile lookup</span>
                <ArrowRight size={16} />
                <span>Scheduler</span>
                <ArrowRight size={16} />
                <span>Chakra DAG</span>
                <ArrowRight size={16} />
                <span>ASTRA-Sim</span>
              </div>
            </div>
          ) : job.status === "failed" ? (
            <div role="alert" className="inline-error">
              <h3>Native run failed</h3>
              <p>{job.error}</p>
              <a
                href={"/api/native/artifact?job=" + job.id + "&file=runner.log"}
                target="_blank"
              >
                Open retained runner log
              </a>
              <br />
              <a
                href={
                  "/api/native/artifact?job=" + job.id + "&file=simulator.log"
                }
                target="_blank"
              >
                Open simulator log
              </a>
            </div>
          ) : !r ? (
            <div className="empty-native">
              <LoaderCircle className="spin" size={40} />
              <h3>
                {job.status === "queued"
                  ? "Experiment queued"
                  : "Running native simulator"}
              </h3>
              <p>
                The CPU backend is processing real profile-driven execution
                graphs. Runs have a time and memory limit.
              </p>
            </div>
          ) : (
            <>
              <div className="card-head">
                <div>
                  <span className="label">COMPLETED NATIVE RUN</span>
                  <h3>LLMServingSim results</h3>
                </div>
                <button
                  className="secondary"
                  onClick={() => download("native-result.json", r)}
                >
                  <Download size={14} />
                  Export JSON
                </button>
              </div>
              <div className="native-metrics">
                <Kpi
                  label="Output throughput"
                  value={fmt(
                    r.metrics?.output_tokens_per_second ??
                      r.metrics?.throughput_tps,
                  )}
                  unit="tok/s"
                  note="Native serving result"
                  icon={<Activity size={15} />}
                />
                <Kpi
                  label="Mean TTFT"
                  value={fmt(
                    r.metrics?.mean_ttft_ms ?? r.metrics?.ttft_mean_ms,
                  )}
                  unit="ms"
                  note="Native request CSV"
                  icon={<Clock3 size={15} />}
                />
                <Kpi
                  label="Mean TPOT"
                  value={fmt(
                    r.metrics?.mean_tpot_ms ?? r.metrics?.tpot_mean_ms,
                  )}
                  unit="ms"
                  note="Native request CSV"
                  icon={<Zap size={15} />}
                />
              </div>
              <p className="fine">
                Actual native clocks and graph dependencies. Profile costs are
                not per-node observed start/end times.
              </p>
              <div className="native-request-box">
                <h4>Native request box graph</h4>
                <p className="fine">
                  CSV boundaries: arrival → first token → completion. TTFT
                  includes queueing.
                </p>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox={`0 0 720 ${r.requests.length * 30 + 55}`}
                  role="img"
                  aria-label="Native request latency box graph"
                >
                  <rect width="720" height="100%" fill="white" />
                  {r.requests.map((q: Obj, i: number) => {
                    const maximum = Math.max(
                      ...r.requests.map((x: Obj) => x.end_time),
                    );
                    const scale = 580 / Math.max(1, maximum);
                    const start = q.arrival * scale,
                      first = (q.arrival + q.TTFT) * scale,
                      end = q.end_time * scale;
                    return (
                      <g key={i}>
                        <text
                          x="10"
                          y={i * 30 + 28}
                          fill="#8aa2af"
                          fontSize="10"
                        >
                          Request {q["request id"]}
                        </text>
                        <rect
                          x={95 + start}
                          y={i * 30 + 13}
                          width={Math.max(1, first - start)}
                          height="21"
                          rx="3"
                          fill="#77bba6"
                        />
                        <rect
                          x={95 + first}
                          y={i * 30 + 13}
                          width={Math.max(1, end - first)}
                          height="21"
                          rx="3"
                          fill="#8ea8d2"
                        />
                      </g>
                    );
                  })}
                  <text
                    x="95"
                    y={r.requests.length * 30 + 46}
                    fill="#9aaeba"
                    fontSize="9"
                  >
                    Green: time to first token · Blue: generation after first
                    token
                  </text>
                </svg>
              </div>
              <div className="native-graph-toolbar">
                <h4>Actual Chakra computation graph</h4>
                <select
                  aria-label="Native graph phase"
                  value={graphPhase}
                  onChange={(e) => {
                    setGraphPhase(e.target.value);
                  }}
                >
                  {Object.keys(graphs).map((k) => (
                    <option key={k}>{k}</option>
                  ))}
                </select>
              </div>
              <p className="fine">
                {graph?.nodes?.length || 0} nodes · {graph?.edges?.length || 0}{" "}
                dependency edges. Full graph and profile costs are included in
                the exported JSON.
              </p>
              {graph && <NativeGraph key={graph.source} graph={graph} />}
              <details>
                <summary>
                  Native provenance and artifacts
                  <ChevronDown size={14} />
                </summary>
                <pre className="source-json">
                  {JSON.stringify(r.provenance, null, 2)}
                </pre>
              </details>
              <a
                className="artifact-link"
                target="_blank"
                href={
                  "/api/native/artifact?job=" + job.id + "&file=requests.csv"
                }
              >
                <ArrowDownToLine size={14} />
                Native request CSV
              </a>
              <a
                className="artifact-link"
                target="_blank"
                href={
                  "/api/native/artifact?job=" + job.id + "&file=simulator.log"
                }
              >
                <FileJson size={14} />
                Simulator log
              </a>
            </>
          )}
        </section>
      </div>
      <div className="method-note">
        <CircleHelp size={17} />
        <p>
          New hardware needs per-operation latency profiles across token count,
          context, batch size and parallelism, plus memory and interconnect
          characteristics. The native engine does not infer those profiles from
          peak TFLOPS alone.
        </p>
      </div>
    </div>
  );
}
function App() {
  const [catalog, setCatalog] = useState<Obj | null>(null);
  const [config, setConfig] = useState<Obj | null>(null);
  const [r, setR] = useState<Obj | null>(null);
  const [pinned, setPinned] = useState<Obj | null>(null);
  const [tab, setTab] = useState("overview");
  const [phase, setPhase] = useState("decode");
  const [editPhase, setEditPhase] = useState("prefill");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState<Obj | null>(null);
  const [mobileConfig, setMobileConfig] = useState(false);
  const [showMethods, setShowMethods] = useState(false);
  const [dirty, setDirty] = useState(false);
  async function refresh() {
    try {
      setHealth(await api("health"));
    } catch {}
  }
  useEffect(() => {
    api("catalog")
      .then(async (c) => {
        setCatalog(c);
        setConfig(c.default_config);
        setLoading(true);
        try {
          setR(await api("simulate", c.default_config));
        } catch (e: any) {
          setError(e.message);
        } finally {
          setLoading(false);
        }
      })
      .catch((e) => setError(e.message));
    refresh();
  }, []);
  function update(k: string, v: any) {
    setConfig((c: Obj | null) => ({ ...c, [k]: v }));
    setDirty(true);
  }
  function pool(k: string, v: any) {
    update(editPhase, { ...config![editPhase], [k]: v });
  }
  function hw(k: string, v: any) {
    update(editPhase, {
      ...config![editPhase],
      hardware_modified: true,
      hardware: { ...config![editPhase].hardware, [k]: v },
    });
  }
  function selectModel(id: string) {
    const model =
      id === "custom"
        ? {
            ...clone(
              config!.model_id === "custom"
                ? config!.custom_model
                : catalog!.models[config!.model_id],
            ),
            name: "Custom decoder",
          }
        : catalog!.models[id];
    const fit = (n: number, total: number) => {
      while (n > 1 && total % n) n--;
      return n;
    };
    setConfig((c: Obj | null) => ({
      ...c,
      model_id: id,
      ...(id === "custom" ? { custom_model: model } : {}),
      ...Object.fromEntries(
        ["prefill", "decode"].map((ph) => [
          ph,
          {
            ...c![ph],
            tp: fit(c![ph].tp, model.heads),
            pp: fit(c![ph].pp, model.layers),
            ep: fit(c![ph].ep, model.experts),
          },
        ]),
      ),
    }));
    setDirty(true);
  }
  async function run() {
    if (!config) return;
    setLoading(true);
    setError("");
    try {
      setR(await api("simulate", config));
      setDirty(false);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }
  async function importConfig(file: File) {
    try {
      if (file.size > 256000)
        throw Error("Configuration must be smaller than 256 KB.");
      const data = JSON.parse(await file.text());
      const validated = await api("validate", data.config || data);
      setConfig(validated.config);
      if (!validated.config.separated) setEditPhase("prefill");
      setDirty(true);
      setError("Imported. Run the scenario to validate and calculate results.");
    } catch (e: any) {
      setError(e.message);
    }
  }
  if (!catalog || !config)
    return (
      <div className="loading">
        <LoaderCircle className="spin" />
        {error || "Loading MemAnalysis…"}
      </div>
    );
  const h = config[editPhase].hardware;
  const selectedPool = config[editPhase];
  const selectedModel =
    config.model_id === "custom"
      ? config.custom_model
      : catalog.models[config.model_id];
  const metric = r?.metrics;
  const phaseButton = (
    <div className="segmented">
      <button
        className={phase === "prefill" ? "active" : ""}
        onClick={() => setPhase("prefill")}
      >
        Prefill
      </button>
      <button
        className={phase === "decode" ? "active" : ""}
        onClick={() => setPhase("decode")}
      >
        Decode
      </button>
    </div>
  );
  const hwGroups: [string, [string, string, string][]][] = [
    [
      "Compute throughput",
      [
        ["bf16_tensor_tflops", "BF16 tensor", "TFLOPS"],
        ["fp16_tensor_tflops", "FP16 tensor", "TFLOPS"],
        ["fp8_tensor_tflops", "FP8 tensor", "TFLOPS"],
        ["fp4_tensor_tflops", "FP4 tensor", "TFLOPS"],
        ["tf32_tensor_tflops", "TF32 tensor", "TFLOPS"],
        ["int8_tensor_tflops", "INT8 tensor", "TFLOPS"],
        ["fp32_tflops", "FP32 vector", "TFLOPS"],
        ["compute_efficiency", "Compute efficiency", "fraction"],
        ["clock_scale", "Compute clock scale", "×"],
        ["launch_overhead_us", "Extra launch overhead", "µs"],
      ],
    ],
    [
      "Memory + tile shape",
      [
        ["main_memory_GB", "Main memory", "GB"],
        ["dram_bandwidth_TBps", "Memory bandwidth", "TB/s"],
        ["sram_cache_MB", "KV SRAM budget", "MB"],
        ["memory_efficiency", "Memory efficiency", "fraction"],
        ["num_sms", "SM / CU count", ""],
        ["tensor_width", "Tensor tile width", ""],
      ],
    ],
    [
      "Network hierarchy",
      [
        ["intra_node_bw_GBps", "Intra-node bandwidth", "GB/s"],
        ["chip_cluster_size", "GPUs / chip domain", ""],
        ["same_rail_bw_GBps", "Same-rail bandwidth", "GB/s"],
        ["same_rail_gpus", "GPUs / same rail", ""],
        ["scale_out_bw_GBps", "Scale-out NIC", "GB/s"],
        ["default_GPUs_per_rack", "GPUs / rack", ""],
        ["cluster_default_racks", "Cluster racks", ""],
        ["network_latency_us", "Base fabric latency", "µs"],
        ["collective_allreduce_fraction", "All-reduce efficiency", "fraction"],
        ["collective_alltoall_fraction", "All-to-all efficiency", "fraction"],
        ["collective_alltoall_offset", "All-to-all offset", "µs"],
      ],
    ],
    [
      "Power + economics",
      [
        ["tdp_watts", "Device TDP", "W"],
        ["idle_watts", "Idle power", "W"],
        ["hourly_cost", "GPU hourly cost", "$/h"],
        ["pj_per_flop", "Tensor energy", "pJ/FLOP"],
        ["pj_per_vop", "Vector energy", "pJ/op"],
        ["pj_per_mem_byte", "HBM energy", "pJ/B"],
        ["pj_per_net_byte", "Network energy", "pJ/B"],
      ],
    ],
  ];
  const delta = (key: string) =>
    pinned && metric && pinned.metrics[key] !== 0
      ? `${metric[key] >= pinned.metrics[key] ? "+" : ""}${fmt((metric[key] / pinned.metrics[key] - 1) * 100)}%`
      : undefined;
  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">
            <Layers3 size={22} />
          </span>
          <strong>
            LLM <b>MemAnalysis</b>
          </strong>
          <span className="brand-divider" />
          <span className="brand-context">Architecture exploration</span>
        </a>
        <div className="top-right">
          <span className="local-pill">
            <i />
            Local workspace
          </span>
          <button
            className="icon-button"
            aria-label="Model methodology"
            onClick={() => setShowMethods(true)}
          >
            <CircleHelp size={19} />
          </button>
          <a
            className="upstream-link"
            href="https://github.com/casys-kaist/LLMServingSim"
            target="_blank"
            rel="noreferrer"
          >
            KAIST engine
            <ExternalLink size={13} />
          </a>
        </div>
      </header>
      <div className={"app-layout " + (tab === "native" ? "native-mode" : "")}>
        <aside className={"sidebar " + (mobileConfig ? "show" : "")}>
          <div className="sidebar-heading">
            <div>
              <SlidersHorizontal size={17} />
              <h2>Scenario inputs</h2>
            </div>
            <button
              className="icon-button"
              aria-label="Reset scenario"
              onClick={() => {
                setConfig(clone(catalog.default_config));
                setDirty(true);
              }}
            >
              <RotateCcw size={14} />
            </button>
          </div>
          <div className="sidebar-intro">
            Model, workload, hardware and topology.
          </div>
          <Section title="Model & workload" initial>
            <label className="field vertical">
              <span>Model architecture</span>
              <select
                aria-label="Model architecture"
                value={config.model_id}
                onChange={(e) => selectModel(e.target.value)}
              >
                <optgroup label="Dense decoders">
                  {Object.entries(catalog.models)
                    .filter(
                      ([id, m]: [string, any]) =>
                        id !== "custom" && m.experts === 1,
                    )
                    .map(([id, m]: [string, any]) => (
                      <option key={id} value={id}>
                        {m.name}
                      </option>
                    ))}
                </optgroup>
                <optgroup label="Mixture of experts">
                  {Object.entries(catalog.models)
                    .filter(
                      ([id, m]: [string, any]) =>
                        id !== "custom" && m.experts > 1,
                    )
                    .map(([id, m]: [string, any]) => (
                      <option key={id} value={id}>
                        {m.name}
                      </option>
                    ))}
                </optgroup>
                <option value="custom">Custom decoder architecture</option>
              </select>
            </label>
            <div className="preset-summary">
              <span>
                {selectedModel.layers} layers · {selectedModel.hidden} hidden
              </span>
              <span>
                {selectedModel.heads} attention / {selectedModel.kv_heads} KV
                heads
                {selectedModel.experts > 1
                  ? ` · ${selectedModel.top_k}/${selectedModel.experts} experts`
                  : " · dense"}
              </span>
              {config.model_id !== "custom" && (
                <button
                  className="text-button"
                  onClick={() => selectModel("custom")}
                >
                  <Settings2 size={13} /> Customize architecture
                </button>
              )}
            </div>
            <label className="field">
              <span>Weight precision</span>
              <select
                aria-label="Weight precision"
                value={config.precision}
                onChange={(e) => update("precision", e.target.value)}
              >
                {["bf16", "fp16", "fp8", "fp4", "tf32", "int8", "fp32"].map(
                  (x) => (
                    <option key={x}>{x}</option>
                  ),
                )}
              </select>
            </label>
            <label className="field vertical">
              <span>
                Engine efficiency policy <CircleHelp size={11} />
              </span>
              <select
                aria-label="Engine efficiency policy"
                value={config.inference_engine}
                onChange={(e) => update("inference_engine", e.target.value)}
              >
                <option value="vllm">vLLM-style · assumed efficiency</option>
                <option value="trtllm">
                  TensorRT-LLM-style · assumed efficiency
                </option>
                <option value="eager">Eager execution · conservative</option>
              </select>
            </label>
            {config.model_id === "custom" && (
              <details className="model-geometry" open>
                <summary>
                  Decoder architecture <ChevronDown size={14} />
                </summary>
                <label className="field vertical">
                  <span>Model name</span>
                  <input
                    aria-label="Model name"
                    value={config.custom_model.name}
                    maxLength={80}
                    onChange={(e) =>
                      update("custom_model", {
                        ...config.custom_model,
                        name: e.target.value,
                      })
                    }
                  />
                </label>
                {(
                  [
                    ["layers", "Transformer layers", 1, 256],
                    ["hidden", "Hidden width", 128, 65536],
                    ["intermediate", "FFN / expert width", 128, 262144],
                    ["heads", "Attention heads", 1, 512],
                    ["kv_heads", "KV heads", 1, 512],
                    ["head_dim", "Head dimension", 8, 512],
                    ["vocab", "Vocabulary size", 128, 300000],
                    ["experts", "Total experts", 1, 256],
                    ["top_k", "Active experts / token", 1, 16],
                    ["max_context", "Model context limit", 1, 1048576],
                  ] as [string, string, number, number][]
                ).map(([key, label, min, max]) => (
                  <Field
                    key={key}
                    label={label}
                    value={config.custom_model[key]}
                    min={min}
                    max={max}
                    onChange={(n) =>
                      update("custom_model", {
                        ...config.custom_model,
                        [key]: n,
                      })
                    }
                  />
                ))}
                {[
                  ["tie_embeddings", "Tied embeddings"],
                  ["qk_norm", "QK normalization"],
                ].map(([key, label]) => (
                  <label className="toggle-label" key={key}>
                    <span>{label}</span>
                    <input
                      type="checkbox"
                      aria-label={label}
                      checked={config.custom_model[key]}
                      onChange={(e) =>
                        update("custom_model", {
                          ...config.custom_model,
                          [key]: e.target.checked,
                        })
                      }
                    />
                  </label>
                ))}
                <p className="fine">
                  Uniform decoder blocks with RMSNorm, RoPE and SwiGLU. Set
                  experts to 1 for dense; KV heads equal attention heads for
                  MHA, or 1 for MQA. MLA, hybrid attention and vision encoders
                  need a different operation model.
                </p>
              </details>
            )}
            <label className="field">
              <span>KV cache precision</span>
              <select
                aria-label="KV cache precision"
                value={config.kv_precision}
                onChange={(e) => update("kv_precision", e.target.value)}
              >
                {["bf16", "fp16", "fp8"].map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
            </label>
            <Field
              label="Input length"
              value={config.input_tokens}
              unit="tokens"
              min={1}
              max={32768}
              step={128}
              onChange={(n) => update("input_tokens", n)}
            />
            <Field
              label="Output length"
              value={config.output_tokens}
              unit="tokens"
              min={1}
              max={256}
              onChange={(n) => update("output_tokens", n)}
            />
            <Field
              label="Requests"
              value={config.requests}
              min={1}
              max={128}
              onChange={(n) => update("requests", n)}
            />
            <Field
              label="Arrival rate"
              value={config.arrival_rate}
              unit="req/s"
              step={0.5}
              min={0.01}
              onChange={(n) => update("arrival_rate", n)}
            />
          </Section>
          <Section title="Serving topology" initial>
            <label className="toggle-label">
              <span>Separate prefill & decode</span>
              <input
                aria-label="Separate prefill and decode"
                type="checkbox"
                checked={config.separated}
                onChange={(e) => {
                  update("separated", e.target.checked);
                  if (!e.target.checked) setEditPhase("prefill");
                }}
              />
            </label>
            <div className="segmented edit-phase">
              <button
                className={editPhase === "prefill" ? "active" : ""}
                onClick={() => setEditPhase("prefill")}
              >
                Prefill pool
              </button>
              <button
                disabled={!config.separated}
                className={editPhase === "decode" ? "active" : ""}
                onClick={() => setEditPhase("decode")}
              >
                Decode pool
              </button>
            </div>
            <label className="field vertical">
              <span>Hardware preset</span>
              <select
                aria-label="Hardware preset"
                value={config[editPhase].hardware_id}
                onChange={(e) =>
                  update(editPhase, {
                    ...config[editPhase],
                    hardware_id: e.target.value,
                    hardware: clone(catalog.hardware[e.target.value]),
                    hardware_modified: false,
                  })
                }
              >
                {Object.entries(catalog.hardware).map(
                  ([id, x]: [string, any]) => (
                    <option key={id} value={id}>
                      {x.name}
                    </option>
                  ),
                )}
              </select>
            </label>
            <div className="preset-summary">
              <span>
                {fmt(h.main_memory_GB)} GB · {fmt(h.dram_bandwidth_TBps, 3)}{" "}
                TB/s
              </span>
              <span>
                {selectedPool.hardware_modified
                  ? "Edited scenario inputs"
                  : "Peak-spec scenario · uncalibrated"}
              </span>
              {h.source_url && (
                <a href={h.source_url} target="_blank" rel="noreferrer">
                  Preset source <ExternalLink size={12} />
                </a>
              )}
            </div>
            <div className="parallel-fields">
              {[
                ["tp", "TP"],
                ["pp", "PP"],
                ["ep", "EP"],
                ["replicas", "Replicas"],
              ].map(([key, label]) => (
                <Field
                  key={key}
                  label={label}
                  value={config[editPhase][key]}
                  min={1}
                  max={64}
                  onChange={(n) => pool(key, n)}
                />
              ))}
            </div>
            <div className="server-policy">
              <h3>
                {editPhase === "prefill" ? "Prefill" : "Decode"} server policy
              </h3>
              <label className="field vertical">
                <span>Server engine policy</span>
                <select
                  aria-label="Server engine policy"
                  value={selectedPool.engine}
                  onChange={(e) => pool("engine", e.target.value)}
                >
                  <option value="inherit">Use workload engine policy</option>
                  <option value="vllm">vLLM-style efficiency</option>
                  <option value="trtllm">TensorRT-LLM-style efficiency</option>
                  <option value="eager">Eager execution efficiency</option>
                </select>
              </label>
              <Field
                label={
                  editPhase === "prefill"
                    ? "Prefill microbatch"
                    : "Decode batch cap"
                }
                value={selectedPool.batch_size}
                min={1}
                max={32}
                onChange={(n) => pool("batch_size", n)}
                tip="Decode may split a prefill group. Groups from different prefill batches are not merged."
              />
              {editPhase === "prefill" && (
                <Field
                  label="Prefill chunk"
                  value={config.prefill_chunk}
                  min={1}
                  max={8192}
                  unit="tokens"
                  step={128}
                  onChange={(n) => update("prefill_chunk", n)}
                />
              )}
              <Field
                label="Active batches / replica"
                value={selectedPool.max_concurrent_batches}
                min={1}
                max={128}
                onChange={(n) => pool("max_concurrent_batches", n)}
                tip="Admission slots held for the whole phase, including all prefill chunks or decode tokens."
              />
              <Field
                label="Memory allocation limit"
                value={selectedPool.memory_utilization}
                min={0.1}
                max={1}
                step={0.05}
                unit="fraction"
                onChange={(n) => pool("memory_utilization", n)}
              />
              <Field
                label="Runtime reservation"
                value={selectedPool.runtime_reserved_GB}
                min={0}
                max={10000}
                step={0.5}
                unit="GB"
                onChange={(n) => pool("runtime_reserved_GB", n)}
              />
              <p className="fine">
                {fmt(
                  Math.max(
                    0,
                    h.main_memory_GB * selectedPool.memory_utilization -
                      selectedPool.runtime_reserved_GB,
                  ),
                )}{" "}
                GB usable per GPU.{" "}
                {editPhase === "prefill"
                  ? "Fixed prefill groups; round-robin replicas."
                  : "Prefill groups split to this cap; no cross-group merging."}{" "}
                Engine policies are assumed efficiencies.
              </p>
            </div>
            <Field
              label="Spine bandwidth"
              value={config.network.spine_bw_GBps}
              unit="GB/s"
              onChange={(n) =>
                update("network", { ...config.network, spine_bw_GBps: n })
              }
            />
            <Field
              label="Spine latency"
              value={config.network.spine_latency_us}
              unit="µs"
              onChange={(n) =>
                update("network", { ...config.network, spine_latency_us: n })
              }
            />
            {config.separated && (
              <Field
                label="KV transfer channels"
                value={config.network.handoff_channels}
                min={1}
                max={16}
                onChange={(n) =>
                  update("network", { ...config.network, handoff_channels: n })
                }
                tip="Independent parallel links, each with the spine bandwidth above."
              />
            )}
          </Section>
          <div className="hardware-editing">
            <Cpu size={13} />
            Editing {editPhase} hardware
          </div>
          {hwGroups.map(([title, fields]) => (
            <Section key={title} title={title}>
              {fields.map(([key, label, unit]) => (
                <Field
                  key={key}
                  label={label}
                  value={h[key]}
                  step={
                    unit === "fraction" || unit === "×"
                      ? 0.05
                      : unit === "TB/s"
                        ? 0.1
                        : 1
                  }
                  unit={unit}
                  onChange={(n) => hw(key, n)}
                />
              ))}
            </Section>
          ))}
          <div className="sidebar-footer">
            <button
              className="text-button"
              onClick={() => download("memanalysis-scenario.json", config)}
            >
              <Download size={13} />
              Save scenario
            </button>
            <label className="text-button upload-button">
              <Upload size={13} />
              Import
              <input
                type="file"
                accept="application/json,.json"
                aria-label="Import scenario"
                onChange={(e) => {
                  if (e.target.files?.[0]) importConfig(e.target.files[0]);
                  e.target.value = "";
                }}
              />
            </label>
          </div>
        </aside>
        <main>
          <div className="page-heading">
            <h1>
              {tab === "native" ? "Native simulation" : "Serving analysis"}
            </h1>
            <div className="run-actions">
              {tab === "native" ? (
                <button
                  className="secondary"
                  onClick={() => setTab("overview")}
                >
                  Back to behavioral scenario
                  <ArrowRight size={14} />
                </button>
              ) : (
                <>
                  <button
                    className="mobile-inputs secondary"
                    onClick={() => setMobileConfig(!mobileConfig)}
                  >
                    <SlidersHorizontal size={15} />
                    Inputs
                  </button>
                  <button
                    className="secondary pin-button"
                    disabled={!r}
                    onClick={() => setPinned(pinned ? null : clone(r))}
                  >
                    <Pin size={15} />
                    {pinned ? "Unpin baseline" : "Pin baseline"}
                  </button>
                  <button className="primary" onClick={run} disabled={loading}>
                    {loading ? (
                      <LoaderCircle size={16} className="spin" />
                    ) : (
                      <Play size={15} fill="currentColor" />
                    )}
                    {loading ? "Simulating…" : "Run scenario"}
                  </button>
                </>
              )}
            </div>
          </div>
          {tab !== "native" && (
            <div className="scenario-bar">
              <span className="scenario-name">
                {selectedModel.name}
                <ChevronRight size={13} />
                {config.precision.toUpperCase()}
                <ChevronRight size={13} />
                {config.separated ? "Prefill / decode split" : "Colocated"}
              </span>
              <span className={"scenario-status " + (dirty ? "dirty" : "")}>
                <i />
                {dirty
                  ? "Inputs changed · run to update"
                  : "Results match scenario"}
              </span>
              <button
                className="text-button"
                disabled={!r}
                onClick={() => download("memanalysis-result.json", r)}
              >
                <Download size={13} />
                Export run
              </button>
            </div>
          )}
          {error && (
            <div className="error-banner" role="alert">
              <CircleHelp size={16} />
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          <nav className="tabs" aria-label="Analysis views">
            {tabs.map(([id, title]) => (
              <button
                key={id}
                className={tab === id ? "active" : ""}
                onClick={() => setTab(id)}
              >
                {title}
                {id === "native" && <span className="native-dot" />}
              </button>
            ))}
          </nav>
          {tab === "native" ? (
            <Native health={health} onRefresh={refresh} />
          ) : !r ? (
            <div className="empty-native">
              <LoaderCircle className="spin" />
              Preparing first scenario…
            </div>
          ) : (
            <>
              <div className="result-caption">
                <span>
                  <FlaskConical size={13} />
                  MemAnalysis behavioral estimate · SimPy
                </span>
                <button onClick={() => setShowMethods(true)}>
                  Assumptions & provenance
                  <ArrowRight size={13} />
                </button>
              </div>
              {tab === "overview" && (
                <>
                  <div className="kpi-grid">
                    <Kpi
                      label="Generation throughput"
                      value={fmt(metric!.throughput_tps, 0)}
                      unit="tok/s"
                      note={`${fmt(metric!.tokens_per_gpu_s)} tok/s per GPU`}
                      icon={<Activity size={15} />}
                      delta={delta("throughput_tps")}
                    />
                    <Kpi
                      label="Time to first token"
                      value={fmt(metric!.ttft_mean_ms, 0)}
                      unit="ms"
                      note={`P95 ${fmt(metric!.ttft_p95_ms, 0)} ms`}
                      icon={<Clock3 size={15} />}
                      delta={delta("ttft_mean_ms")}
                    />
                    <Kpi
                      label="Time per output token"
                      value={fmt(metric!.tpot_mean_ms)}
                      unit="ms"
                      note="Mean inter-token interval"
                      icon={<Zap size={15} />}
                      delta={delta("tpot_mean_ms")}
                    />
                    <Kpi
                      label="Energy efficiency"
                      value={fmt(metric!.tokens_per_joule, 3)}
                      unit="tok/J"
                      note={`${fmt(metric!.energy_j / 1000, 2)} kJ estimated GPU energy`}
                      icon={<Sparkles size={15} />}
                      delta={delta("tokens_per_joule")}
                    />
                  </div>
                  <Pipeline r={r} />
                  <div className="overview-grid">
                    <MemoryChart r={r} />
                    <section className="card insight-card">
                      <div className="card-head">
                        <div>
                          <h3>Resource service demand</h3>
                          <p>
                            Accumulated resource work · compute and memory
                            overlap
                          </p>
                        </div>
                        <Activity size={16} />
                      </div>
                      {Object.entries(r.component_work_ms).map(
                        ([key, value]: [string, any]) => (
                          <div className="work-row" key={key}>
                            <div>
                              <span>
                                <i
                                  style={{
                                    background:
                                      colors[key as keyof typeof colors],
                                  }}
                                />
                                {key}
                              </span>
                              <b>{fmt(value)} ms</b>
                            </div>
                            <div className="work-track">
                              <i
                                style={{
                                  width: `${(value / Math.max(...(Object.values(r.component_work_ms) as number[]))) * 100}%`,
                                  background:
                                    colors[key as keyof typeof colors],
                                }}
                              />
                            </div>
                          </div>
                        ),
                      )}
                      <div className="insight-note">
                        <CircleHelp size={15} />
                        <p>
                          Decode KV reads served from ideal SRAM:{" "}
                          <strong>
                            {fmt(
                              r.phases.decode.detail.sram_hit_fraction * 100,
                            )}
                            %
                          </strong>
                          . Remaining KV reads spill to HBM.
                        </p>
                      </div>
                    </section>
                  </div>
                  <div className="economics-grid">
                    {[
                      [
                        "GPU pool",
                        `${metric!.gpu_count} GPUs`,
                        `${fmt(metric!.gpu_utilization * 100)}% resource occupancy`,
                      ],
                      [
                        "Serving cost",
                        `$${fmt(metric!.cost_per_million_tokens, 2)}`,
                        "per million output tokens · GPU rent only",
                      ],
                      [
                        "P95 request latency",
                        `${fmt(metric!.latency_p95_ms / 1000, 2)} s`,
                        `${r.requests.length} requests · ${fmt(metric!.elapsed_ms / 1000, 2)} s simulated`,
                      ],
                      [
                        "System traffic",
                        `${fmt(r.traffic.hbm_bytes / 1e9)} GB`,
                        "HBM traffic · all GPUs, all requests",
                      ],
                    ].map(([a, b, c]) => (
                      <div className="mini-card" key={a}>
                        <span>{a}</span>
                        <strong>{b}</strong>
                        <small>{c}</small>
                      </div>
                    ))}
                  </div>
                  <div className="method-note">
                    <CircleHelp size={17} />
                    <p>
                      Hardware presets and engine efficiency are scenario
                      assumptions. Use the native LLMServingSim tab for
                      profile-driven reference runs, then calibrate the
                      behavioral model against measurements.
                    </p>
                  </div>
                </>
              )}
              {tab === "roofline" && (
                <>
                  <div className="section-toolbar">
                    <div>
                      <h2>Performance roofline</h2>
                      <p>
                        See which resource limits each transformer operation.
                      </p>
                    </div>
                    {phaseButton}
                  </div>
                  <Roofline r={r} phase={phase} />
                  <div className="explain-grid">
                    <div className="card">
                      <Database size={21} />
                      <h3>HBM bandwidth</h3>
                      <p>
                        {fmt(r.config[phase].hardware.dram_bandwidth_TBps)} TB/s
                        peak ×{" "}
                        {fmt(r.config[phase].hardware.memory_efficiency * 100)}%
                        efficiency. KV SRAM budget reduces attention’s modeled
                        HBM reads.
                      </p>
                    </div>
                    <div className="card">
                      <Cpu size={21} />
                      <h3>Compute ceiling</h3>
                      <p>
                        {fmt(r.phases[phase].detail.peak_tflops)} dense TFLOPS
                        at {r.config.precision.toUpperCase()}. SM count, tile
                        width and launch overhead affect operation attainment.
                      </p>
                    </div>
                    <div className="card">
                      <Network size={21} />
                      <h3>Collective fabric</h3>
                      <p>
                        {r.phases[phase].detail.fabric.name}:{" "}
                        {fmt(r.phases[phase].detail.fabric.bw)} GB/s before
                        collective efficiency. Crossing a hardware domain
                        selects the slower tier.
                      </p>
                    </div>
                  </div>
                </>
              )}
              {tab === "layers" && (
                <>
                  <div className="section-toolbar">
                    <div>
                      <h2>Layer and chunk detail</h2>
                      <p>
                        Representative operation costs per layer, per GPU; full
                        run intervals below.
                      </p>
                    </div>
                    <button
                      className="secondary"
                      onClick={() => {
                        const keys = [
                          "phase",
                          "name",
                          "flops",
                          "hbm_bytes",
                          "compute_ms",
                          "memory_ms",
                          "network_ms",
                          "duration_ms",
                          "bound",
                        ];
                        download(
                          "layer-estimates.csv",
                          [
                            keys.join(","),
                            ...r.layers.map((o: Obj) =>
                              keys.map((k) => JSON.stringify(o[k])).join(","),
                            ),
                          ].join("\n"),
                          "text/csv",
                        );
                      }}
                    >
                      <Download size={14} />
                      CSV
                    </button>
                  </div>
                  <section className="card table-wrap">
                    <table>
                      <thead>
                        <tr>
                          {[
                            "Phase / operation",
                            "Compute",
                            "HBM",
                            "Network",
                            "Service time",
                            "Limit",
                          ].map((s) => (
                            <th key={s}>{s}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {r.layers.map((o: Obj) => (
                          <tr key={o.phase + o.id}>
                            <td>
                              <small>{o.phase.toUpperCase()}</small>
                              <strong>{o.name}</strong>
                            </td>
                            <td>{fmt(o.compute_ms * 1000, 2)} µs</td>
                            <td>{fmt(o.memory_ms * 1000, 2)} µs</td>
                            <td>{fmt(o.network_ms * 1000, 2)} µs</td>
                            <td>
                              <b>{fmt(o.duration_ms * 1000, 2)} µs</b>
                            </td>
                            <td>
                              <span
                                className={"bound " + o.bound.toLowerCase()}
                              >
                                {o.bound}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </section>
                  <BoxGraph r={r} />
                </>
              )}
              {tab === "boxes" && (
                <>
                  <div className="section-toolbar">
                    <div>
                      <h2>Execution timeline</h2>
                      <p>
                        Pipeline overlap, resource occupancy and request-level
                        distributions.
                      </p>
                    </div>
                  </div>
                  <BoxGraph r={r} />
                  <Distribution requests={r.requests} />
                </>
              )}
              {tab === "graph" && (
                <>
                  <div className="section-toolbar">
                    <div>
                      <h2>Computation graph</h2>
                      <p>
                        Click an operation to inspect its workload and limiting
                        resource.
                      </p>
                    </div>
                    {phaseButton}
                  </div>
                  <BehavioralGraph result={r} phase={phase} />
                </>
              )}
              {tab === "configuration" && (
                <>
                  <div className="section-toolbar">
                    <div>
                      <h2>Run configuration</h2>
                      <p>
                        Every estimate carries its inputs, formulas, units and
                        source revision.
                      </p>
                    </div>
                    <button
                      className="secondary"
                      onClick={() => download("scenario.json", r.config)}
                    >
                      <Download size={14} />
                      Configuration
                    </button>
                  </div>
                  <div className="config-output-grid">
                    <section className="card">
                      <h3>Result provenance</h3>
                      <dl className="provenance-list">
                        <div>
                          <dt>Timing engine</dt>
                          <dd>{r.engine}</dd>
                        </div>
                        <div>
                          <dt>Architecture source</dt>
                          <dd>{r.model.source}</dd>
                        </div>
                        <div>
                          <dt>Model config revision</dt>
                          <dd>
                            <code>{r.model.source_revision.slice(0, 12)}</code>
                          </dd>
                        </div>
                        {r.model.source_url && (
                          <div>
                            <dt>Model configuration</dt>
                            <dd>
                              <a
                                href={r.model.source_url}
                                target="_blank"
                                rel="noreferrer"
                              >
                                Source <ExternalLink size={12} />
                              </a>
                            </dd>
                          </div>
                        )}
                        <div>
                          <dt>LLMServingSim revision</dt>
                          <dd>
                            <code>
                              {r.provenance.upstream_revision.slice(0, 12)}
                            </code>
                          </dd>
                        </div>
                        <div>
                          <dt>Scenario fingerprint</dt>
                          <dd>
                            <code>
                              {r.provenance.config_sha256.slice(0, 16)}
                            </code>
                          </dd>
                        </div>
                        <div>
                          <dt>Data units</dt>
                          <dd>GB / TB decimal; time in milliseconds</dd>
                        </div>
                      </dl>
                      <h3>Scope and assumptions</h3>
                      <ul className="assumption-list">
                        {r.warnings.map((s: string) => (
                          <li key={s}>{s}</li>
                        ))}
                      </ul>
                    </section>
                    <section className="card">
                      <h3>Resolved run configuration</h3>
                      <pre className="source-json config-json">
                        {JSON.stringify(r.config, null, 2)}
                      </pre>
                    </section>
                  </div>
                </>
              )}
            </>
          )}
          <footer>
            <span>
              <Layers3 size={13} />
              LLM MemAnalysis
              <span>Behavioral modeling for architectural decisions</span>
            </span>
            <button onClick={() => setShowMethods(true)}>
              Model notes
              <ArrowRight size={12} />
            </button>
          </footer>
        </main>
      </div>
      {showMethods && (
        <div className="modal-backdrop" onClick={() => setShowMethods(false)}>
          <section
            className="method-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Model methodology"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="card-head">
              <div>
                <div className="label">SCOPE & PROVENANCE</div>
                <h2>Simulation methods and provenance</h2>
              </div>
              <button
                aria-label="Close methodology"
                className="icon-button"
                onClick={() => setShowMethods(false)}
              >
                <X size={20} />
              </button>
            </div>
            <h3>Behavioral scenario model</h3>
            <p>
              Our closed-form operation model uses max(compute, memory), adds
              network and launch costs, then runs FIFO microbatches through
              SimPy pipeline resources. It covers the six output families in the
              supplied specification. It is an independently implemented
              extension, not a reimplementation of KAIST’s measured clocks.
            </p>
            <h3>Native LLMServingSim 2.0</h3>
            <p>
              Runs the pristine pinned KAIST scheduler, measured operation
              profiles, Chakra computation DAG and ASTRA-Sim analytical backend.
              These are simulated serving results, not live LLM inference.
              Custom peak hardware inputs do not replace native profile tables.
            </p>
            <h3>Critical calibration data</h3>
            <p>
              Operation latency grids by batch, token/context length, dtype and
              TP; memory capacity and traffic; collective latency/bandwidth by
              topology; engine policy; measured idle/active power. Without those
              measurements, custom-hardware outputs remain strategy estimates.
            </p>
            <h3>Energy model</h3>
            <p>
              Default: idle power over the full horizon plus an assumed 85% of
              the TDP-minus-idle increment during resource occupancy. Optional
              pJ coefficients estimate dynamic energy from operation work and
              traffic, capped at the TDP budget. GPU energy only; networking
              equipment, CPU and cooling are excluded.
            </p>
            <a
              href="https://github.com/casys-kaist/LLMServingSim"
              target="_blank"
              rel="noreferrer"
            >
              Read KAIST’s source and profiling documentation{" "}
              <ExternalLink size={13} />
            </a>
            <div className="method-note">
              <CircleHelp size={16} />
              <p>
                MemAnalysis explores high-level behavior. MemScope inspects
                detailed recorded and modeled memory events. Neither turns an
                uncalibrated architecture assumption into a hardware
                measurement.
              </p>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
