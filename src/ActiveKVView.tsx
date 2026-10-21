import type { ComponentType } from "react";
import FormulaValue from "./FormulaValue";
import { calc, v, scaled } from "./calculations";
import {
  activeMetricCalculation,
  activeAllocationCalculation,
} from "./activeCalculations";
type Obj = Record<string, any>;
export const activeReport =
  "https://github.com/ryeii/LLM-ServeScope/blob/main/docs/active-kv-model.md";
export function ActiveKVControls({
  active: a,
  placements,
  Field,
  onChange,
}: {
  active: Obj;
  placements: Obj;
  Field: ComponentType<any>;
  onChange: (key: string, value: any) => void;
}) {
  const off = a.placement !== "hbm";
  return (
    <div className="storage-controls">
      <label className="field vertical">
        <span>Active decode KV placement</span>
        <select
          aria-label="Active decode KV placement"
          value={a.placement}
          onChange={(e) => onChange("placement", e.target.value)}
        >
          {Object.entries(placements).map(([id, p]: [string, any]) => (
            <option key={id} value={id}>
              {p.label}
            </option>
          ))}
        </select>
      </label>
      <p className="fine">
        {placements[a.placement].description} Placement changes preserve the
        workload, batch sizes and retained-prefix policy.
      </p>
      <p className="fine">
        Prefill working KV stays in HBM. Active decode backing is freed at
        request completion; retained prefixes have a separate lifetime.
      </p>
      {off && (
        <>
          <p className="model-scope-note">
            Serialized streaming estimate: each decode step stages remote KV,
            runs the HBM-reference GPU kernels, then writes appended KV.
            Streaming-kernel overhead and overlap have no measured validation.
          </p>
          {a.placement.startsWith("hbm-") && (
            <Field
              label="Active KV HBM budget / GPU"
              value={a.hbm_resident_GB}
              min={0}
              unit="GB"
              step={0.25}
              onChange={(n: number) => onChange("hbm_resident_GB", n)}
            />
          )}
          <Field
            label="Active GPU staging / GPU"
            value={a.staging_GB}
            min={0.000001}
            unit="GB"
            step={0.015625}
            onChange={(n: number) => onChange("staging_GB", n)}
          />
          <Field
            label="Total host DRAM capacity"
            value={a.dram_total_GB}
            min={0}
            unit="GB"
            step={16}
            onChange={(n: number) => onChange("dram_total_GB", n)}
          />
          <Field
            label="Total SSD capacity"
            value={a.ssd_total_GB}
            min={0}
            unit="GB"
            step={128}
            onChange={(n: number) => onChange("ssd_total_GB", n)}
          />
          <p className="fine">
            Device totals include retained-prefix reservations. HBM staging is
            reserved separately from the active resident budget. Configure
            shared bandwidths and service functions in KV I/O paths.
          </p>
          <a href={activeReport} target="_blank" rel="noreferrer">
            Equations, capacity accounting and validation scope ↗
          </a>
        </>
      )}
    </div>
  );
}
export function KVDeviceControls({
  storage: s,
  Field,
  onChange,
}: {
  storage: Obj;
  Field: ComponentType<any>;
  onChange: (key: string, value: any) => void;
}) {
  return (
    <div className="storage-controls">
      <p className="fine">
        Active decode KV and retained prefixes share these physical FIFO paths.
        Capacities and placement policies are configured separately.
      </p>
      <label className="field vertical">
        <span>SSD service model</span>
        <select
          aria-label="SSD service model"
          value={s.profile}
          onChange={(e) => onChange("profile", e.target.value)}
        >
          <option value="manual">
            Configured affine function · uncalibrated
          </option>
          <option value="local-mooncake-path">
            Fitted local Mooncake file path · QD1
          </option>
        </select>
      </label>
      <p className="fine">
        The fitted profile measures warm buffered file reads and fsync writes on
        a development Mac; it does not calibrate target-server SSD or streamed
        attention.
      </p>
      {[
        ["pcie_GBps", "Host–GPU aggregate link", "GB/s"],
        ["pcie_latency_us", "Host–GPU latency", "µs"],
        ["io_chunk_KiB", "Storage I/O chunk", "KiB"],
        ...(s.profile === "manual"
          ? [
              ["ssd_read_GBps", "SSD read bandwidth", "GB/s"],
              ["ssd_write_GBps", "SSD write bandwidth", "GB/s"],
              ["ssd_read_latency_us", "SSD read latency", "µs"],
              ["ssd_write_latency_us", "SSD write latency", "µs"],
            ]
          : []),
      ].map(([key, label, unit]) => (
        <Field
          key={key}
          label={label}
          value={s[key]}
          unit={unit}
          step={key.includes("GBps") ? 0.5 : 1}
          onChange={(n: number) => onChange(key, n)}
        />
      ))}
    </div>
  );
}
export default function ActiveKVView({ r }: { r: Obj }) {
  const a = r.active_kv;
  const fmt = (n: number, d = 2) =>
    Number(n || 0).toLocaleString("en-US", { maximumFractionDigits: d });
  if (!a)
    return (
      <div className="card active-kv-baseline">
        <h2>Active decode KV · HBM only</h2>
        <p className="storage-note">
          The active working set stays in GPU HBM. Select DRAM, SSD or an HBM
          spill layout under Active decode KV to model per-token backing
          traffic. Retained-prefix placement below is independent.
        </p>
      </div>
    );
  return (
    <div className="active-kv-view storage-view">
      <div className="calibration-status">
        <strong>Active decode KV · {a.label}</strong>
        <span>
          Independent of retained prefixes · shared physical I/O queues
        </span>
      </div>
      <div className="card">
        <h2>Active decode KV</h2>
        <p className="model-scope-note">{a.calibration}</p>
        <p className="storage-note">
          {a.model} Prefill working KV remains in HBM. All resource occupancy,
          memory and energy results include this configured policy.
        </p>
        <a href={activeReport} target="_blank" rel="noreferrer">
          Active-KV equations and limitations ↗
        </a>
      </div>
      <div className="economics-grid storage-kpis">
        {[
          ["attention_read_bytes", "Active decode KV reads", 1e-9, "GB"],
          ["append_write_bytes", "Active KV append writes", 1e-9, "GB"],
          ["ssd_service_ms", "Active SSD service", 1, "ms"],
          ["stage_io_stall_ms", "Decode-stage I/O stalls", 1, "ms"],
        ].map(([key, label, factor, unit]) => (
          <div className="card" key={String(key)}>
            <span>{label}</span>
            <strong>
              <FormulaValue
                calculation={scaled(
                  activeMetricCalculation(r, String(key)),
                  Number(factor),
                  String(unit),
                )}
                formulaKey={"active." + key}
              >
                {fmt((a.stats[String(key)] || 0) * Number(factor))}
              </FormulaValue>
              <small> {unit}</small>
            </strong>
          </div>
        ))}
      </div>
      <div className="card">
        <h3>Active residency and backing capacity</h3>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Resource</th>
                <th>Peak reserved</th>
                <th>Available budget</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Active HBM / decode GPU</td>
                <td>{fmt(a.peak_resident_bytes_per_gpu / 1e9)} GB</td>
                <td>{fmt(a.hbm_capacity_bytes_per_gpu / 1e9)} GB</td>
              </tr>
              <tr>
                <td>GPU staging / decode GPU</td>
                <td>{fmt(a.staging_GB_per_gpu)} GB</td>
                <td>{fmt(a.staging_GB_per_gpu)} GB</td>
              </tr>
              <tr>
                <td>{a.backing.toUpperCase()} active backing · aggregate</td>
                <td>{fmt(a.peak_remote_bytes / 1e9)} GB</td>
                <td>
                  {fmt(a.remote_capacity_bytes / 1e9)} GB after prefix/relay
                  reservations
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="storage-note">
          Requests reserve their maximum input + generated-context payload at
          admission. HBM residence is pinned for the request’s lifetime; the
          remainder is striped across its layers/tokens in the backing tier.
          Staging holds serial tiles, never a second full remote working set.
        </p>
      </div>
      <div className="card">
        <h3>Per-request active KV</h3>
        <div className="table-scroll active-request-table">
          <table>
            <thead>
              <tr>
                <th>Request</th>
                <th>Resident HBM / GPU</th>
                <th>Remote reserve · aggregate</th>
                <th>Remote fraction</th>
                <th>Admission wait</th>
              </tr>
            </thead>
            <tbody>
              {a.allocations.map((q: Obj) => (
                <tr key={q.request_id}>
                  <td>#{q.request_id}</td>
                  <td>{fmt(q.resident_bytes_per_gpu / 1e9)} GB</td>
                  <td>{fmt(q.remote_bytes / 1e9)} GB</td>
                  <td>
                    <FormulaValue
                      calculation={scaled(
                        activeAllocationCalculation(r, q),
                        100,
                        "%",
                      )}
                      formulaKey={
                        "active.request." + q.request_id + ".fraction"
                      }
                    >
                      {fmt(q.remote_fraction * 100)}%
                    </FormulaValue>
                  </td>
                  <td>{fmt(q.admission_ms - q.admission_ready_ms)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
