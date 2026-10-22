import { ArrowRight, ChevronRight, Download, Pin } from "lucide-react";
import type { PointerEvent } from "react";
type Obj = Record<string, any>;
type Change = {
  path: string;
  label: string;
  before: string;
  after: string;
  unit: string;
};
const fields: Record<string, [string, string?]> = {
  model_id: ["Model"],
  precision: ["Weight precision"],
  kv_precision: ["KV cache precision"],
  timing_mode: ["Operator timing model"],
  inference_engine: ["Engine efficiency policy"],
  input_tokens: ["Input length", "tokens"],
  output_tokens: ["Output length", "tokens"],
  requests: ["Requests"],
  arrival_rate: ["Arrival rate", "req/s"],
  prefill_chunk: ["Prefill chunk", "tokens"],
  separated: ["Separate prefill and decode"],
  hardware_id: ["Hardware preset"],
  tp: ["Tensor parallelism"],
  pp: ["Pipeline parallelism"],
  ep: ["Expert parallelism"],
  replicas: ["Replicas"],
  batch_size: ["Microbatch", "requests"],
  max_concurrent_batches: ["Active batches / replica"],
  memory_utilization: ["Memory allocation limit"],
  runtime_reserved_GB: ["Runtime reservation", "GB"],
  engine: ["Server engine policy"],
  dram_bandwidth_TBps: ["HBM bandwidth", "TB/s"],
  main_memory_GB: ["Main memory", "GB"],
  clock_scale: ["Compute clock scale", "×"],
  compute_efficiency: ["Compute efficiency"],
  memory_efficiency: ["Memory efficiency"],
  sram_cache_MB: ["Reserved KV SRAM", "MB"],
  l2_cache_MiB: ["Physical L2 cache", "MiB"],
  num_sms: ["SM count"],
  tensor_width: ["Tensor tile width"],
  launch_overhead_us: ["Launch overhead", "µs"],
  idle_watts: ["Idle power", "W"],
  tdp_watts: ["TDP", "W"],
  hourly_cost: ["GPU hourly cost", "USD/h"],
  enabled: ["Prefix retention"],
  profile: ["SSD service model"],
  hbm_cache_GB: ["Retained HBM / GPU", "GB"],
  dram_cache_GB: ["Host DRAM cache", "GB"],
  ssd_cache_GB: ["SSD cache capacity", "GB"],
  pcie_GBps: ["Host–GPU aggregate link", "GB/s"],
  pcie_latency_us: ["Host–GPU latency", "µs"],
  ssd_transfer_mode: ["SSD transfer path"],
  gds_GBps: ["Direct SSD–GPU link", "GB/s"],
  gds_latency_us: ["GDS startup latency", "µs"],
  publication_mode: ["Prefix publication"],
  max_inflight_publications: ["In-flight prefix publications"],
  disk_min_uses: ["SSD admission minimum uses"],
  io_chunk_KiB: ["Storage I/O chunk", "KiB"],
  ssd_read_GBps: ["SSD read bandwidth", "GB/s"],
  ssd_write_GBps: ["SSD write bandwidth", "GB/s"],
  ssd_read_latency_us: ["SSD read latency", "µs"],
  ssd_write_latency_us: ["SSD write latency", "µs"],
  hbm_resident_GB: ["HBM resident budget / GPU", "GB"],
  staging_GB: ["GPU staging / GPU", "GB"],
  dram_total_GB: ["Total host DRAM capacity", "GB"],
  ssd_total_GB: ["Total SSD capacity", "GB"],
  spine_bw_GBps: ["Spine bandwidth", "GB/s"],
  spine_latency_us: ["Spine latency", "µs"],
  handoff_channels: ["KV transfer channels"],
  intra_node_bw_GBps: ["Intra-node bandwidth", "GB/s"],
  chip_cluster_size: ["GPUs / chip domain"],
  same_rail_bw_GBps: ["Same-rail bandwidth", "GB/s"],
  same_rail_gpus: ["GPUs / same rail"],
  scale_out_bw_GBps: ["Scale-out NIC", "GB/s"],
  default_GPUs_per_rack: ["GPUs / rack"],
  cluster_default_racks: ["Cluster racks"],
  network_latency_us: ["Base fabric latency", "µs"],
  collective_allreduce_fraction: ["All-reduce efficiency"],
  collective_alltoall_fraction: ["All-to-all efficiency"],
  collective_alltoall_offset: ["All-to-all offset", "µs"],
  pj_per_flop: ["Tensor energy", "pJ/FLOP"],
  pj_per_vop: ["Vector energy", "pJ/op"],
  pj_per_mem_byte: ["HBM energy", "pJ/B"],
  pj_per_net_byte: ["Network energy", "pJ/B"],
  hidden: ["Hidden width"],
  intermediate: ["Intermediate width"],
  layers: ["Transformer layers"],
  heads: ["Attention heads"],
  kv_heads: ["KV heads"],
  head_dim: ["Head dimension"],
  vocab: ["Vocabulary size"],
  experts: ["Experts"],
  top_k: ["Active experts / token"],
  max_context: ["Context limit", "tokens"],
  tie_embeddings: ["Tied embedding weights"],
  qk_norm: ["QK normalization"],
  workload: ["Request workload"],
};
const ignored = new Set([
  "hardware_modified",
  "source_sha256",
  "source_file_sha256",
  "source_revision",
  "source_id",
  "source_label",
  "source_url",
  "provenance",
]);
const humanize = (key: string) =>
  key.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
export function inputChanges(before: Obj, after: Obj, catalog: Obj): Change[] {
  const changes: Change[] = [];
  function walk(a: any, b: any, path: string[]) {
    const key = path.at(-1) || "";
    if (ignored.has(key) || (key === "batch_size" && path.length === 1)) return;
    if (
      path[0] === "custom_model" &&
      before.model_id !== "custom" &&
      after.model_id !== "custom"
    )
      return;
    if (key === "name" && path.includes("hardware")) return;
    if (JSON.stringify(a) === JSON.stringify(b)) return;
    if (
      (a && typeof a === "object" && !Array.isArray(a)) ||
      (b && typeof b === "object" && !Array.isArray(b))
    ) {
      for (const child of new Set([
        ...Object.keys(a || {}),
        ...Object.keys(b || {}),
      ]))
        walk(a?.[child], b?.[child], [...path, child]);
      return;
    }
    const scope = (
      {
        prefill: "Prefill",
        decode: "Decode",
        network: "Network",
        storage: "Retained prefixes",
        active_kv: "Active decode KV",
        custom_model: "Model",
      } as Obj
    )[path[0]];
    let [label, unit = ""] = fields[key] || [humanize(key)];
    if (key.endsWith("_tflops"))
      [label, unit] = [humanize(key.replace("_tflops", "")), "TFLOPS"];
    if (key === "placement")
      label = path[0] === "storage" ? "Cache layout" : "Placement";
    function format(value: any): string {
      if (value === undefined || value === null) return "Not set";
      if (Array.isArray(value)) return `${value.length} request records`;
      if (typeof value === "boolean") return value ? "Enabled" : "Disabled";
      if (key === "model_id")
        return catalog.models[value]?.name || String(value);
      if (key === "hardware_id")
        return catalog.hardware[value]?.name || String(value);
      if (key === "placement")
        return (
          (path[0] === "storage"
            ? catalog.storage_layouts
            : catalog.active_kv_placements)[value]?.label || String(value)
        );
      if (["precision", "kv_precision"].includes(key))
        return String(value).toUpperCase();
      if (key === "timing_mode")
        return value === "calibrated"
          ? "Fitted GPU functions"
          : "Original roofline";
      return String(value);
    }
    let afterValue = format(b);
    if (Array.isArray(a) && Array.isArray(b) && a.length === b.length)
      afterValue += " (updated)";
    changes.push({
      path: path.join("."),
      label: scope ? `${scope} · ${label}` : label,
      before: format(a),
      after: afterValue,
      unit,
    });
  }
  walk(before, after, []);
  return changes;
}
export default function RunComparison({
  pinned,
  current,
  draft,
  catalog,
  dirty,
  previewing,
  onPreview,
  onExport,
}: {
  pinned: Obj;
  current: Obj;
  draft: Obj;
  catalog: Obj;
  dirty: boolean;
  previewing: boolean;
  onPreview: (value: boolean) => void;
  onExport: (run: Obj, name: string) => void;
}) {
  const changes = inputChanges(pinned.config, draft, catalog);
  const sameRun =
    pinned.provenance.config_sha256 === current.provenance.config_sha256;
  const showNew = dirty || changes.length > 0 || !sameRun;
  const pinnedActive = previewing || sameRun;
  const name = (c: Obj) =>
    c.model_id === "custom"
      ? c.custom_model.name
      : catalog.models[c.model_id].name;
  const identity = (c: Obj) => (
    <span className="scenario-name">
      {name(c)}
      <ChevronRight size={13} />
      {c.precision.toUpperCase()}
      <ChevronRight size={13} />
      {c.separated ? "Prefill / decode split" : "Colocated"}
    </span>
  );
  function pointerDown(e: PointerEvent<HTMLButtonElement>) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    onPreview(true);
  }
  return (
    <div
      className="run-comparison"
      data-displayed-run={pinnedActive ? "pinned" : "current"}
    >
      <section
        className={`scenario-bar comparison-run ${pinnedActive ? "is-displayed" : ""}`}
        aria-label="Pinned run"
        aria-current={pinnedActive ? "true" : undefined}
      >
        <div className="comparison-summary-row">
          <span className="comparison-run-label">
            <Pin size={13} />
            Pinned run
          </span>
          {identity(pinned.config)}
          <span className="scenario-status">
            <i />
            {pinnedActive
              ? "Showing pinned results"
              : "Original results preserved"}
          </span>
          <button
            className="text-button"
            onClick={() => onExport(pinned, "servescope-pinned-result.json")}
          >
            <Download size={13} />
            Export pinned run
          </button>
        </div>
        <button
          className="pinned-preview-button"
          aria-label="Preview pinned results"
          aria-pressed={previewing}
          onPointerDown={pointerDown}
          onPointerUp={() => onPreview(false)}
          onPointerCancel={() => onPreview(false)}
          onLostPointerCapture={() => onPreview(false)}
          onKeyDown={(e) => {
            if (e.key === " " || e.key === "Enter") {
              e.preventDefault();
              onPreview(true);
            }
          }}
          onKeyUp={(e) => {
            if (e.key === " " || e.key === "Enter") {
              e.preventDefault();
              onPreview(false);
            }
          }}
          onBlur={() => onPreview(false)}
        >
          Hold <kbd>P</kbd> to preview original results
          <span> · or press and hold here</span>
        </button>
      </section>
      {showNew && (
        <section
          className={`scenario-bar comparison-run ${!pinnedActive ? "is-displayed" : ""}`}
          aria-label="New run"
          aria-current={!pinnedActive ? "true" : undefined}
        >
          <div className="comparison-summary-row">
            <span className="comparison-run-label">New run</span>
            {identity(draft)}
            <span className={`scenario-status ${dirty ? "dirty" : ""}`}>
              <i />
              {dirty
                ? "Inputs changed · run to update"
                : "Results match scenario"}
            </span>
            <button
              className="text-button"
              disabled={sameRun && dirty}
              onClick={() => onExport(current, "servescope-result.json")}
            >
              <Download size={13} />
              Export run
            </button>
          </div>
          {changes.length > 0 ? (
            <ul
              className="run-input-changes"
              aria-label="Changes from pinned inputs"
            >
              {changes.map((change) => (
                <li key={change.path} data-input-change={change.path}>
                  <span className="changed-input-label">{change.label}</span>
                  <span className="changed-input-values">
                    <span>{change.before}</span>
                    <ArrowRight size={13} aria-label="changed to" />
                    <strong>
                      {change.after}
                      {change.unit && ` ${change.unit}`}
                    </strong>
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="comparison-note">
              No input differences from the pinned setup.
            </p>
          )}
          <p className="comparison-note">
            {dirty
              ? sameRun
                ? "Run the edited setup to calculate new results. The original results remain displayed."
                : "Pending edits are not included in the displayed results. Run the scenario to update them."
              : !pinnedActive
                ? "Showing the completed new run. Hold P to compare with the pinned original."
                : "New results are preserved. Release P to return to them."}
          </p>
        </section>
      )}
      <span className="sr-only" role="status">
        {pinnedActive
          ? "Showing pinned run results"
          : "Showing new run results"}
      </span>
    </div>
  );
}
