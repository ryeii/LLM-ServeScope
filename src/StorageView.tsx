import { useState, type ComponentType } from "react";
import FormulaValue from "./FormulaValue";
import { calc, v, numberText, scaled } from "./calculations";
import {
  cacheDecisionCalculation,
  storageMetricCalculation,
  storageReport,
  traceReuseReport,
  storageEventCalculation,
  storageRequestCalculation,
} from "./storageCalculations";
type Obj = Record<string, any>;
export function StorageControls({
  storage: s,
  Field,
  onChange,
  onLayout,
  layouts,
  onDemo,
  onDynamo,
  onWorkload,
  onError,
}: {
  storage: Obj;
  Field: ComponentType<any>;
  onChange: (k: string, v: any) => void;
  onLayout: (id: string) => void;
  layouts: Obj;
  onDemo: () => void;
  onDynamo: () => void;
  onWorkload: (rows: Obj[], source: Obj) => Promise<void>;
  onError: (message: string) => void;
}) {
  const [source, setSource] = useState("conversation"),
    [start, setStart] = useState(0),
    [count, setCount] = useState(4),
    [busy, setBusy] = useState(false);
  const layout = layouts[s.enabled ? s.placement || "inclusive" : "none"];
  const tiers: string[] = layout.tiers;
  const hasSSD = tiers.includes("ssd");
  async function parse(
    raw: string,
    id: string,
    label: string,
    revision?: string,
    expectedHash?: string,
  ) {
    const lines = raw
      .split(/\r?\n/)
      .map((text, index) => ({ text, index }))
      .filter((x) => x.text.trim());
    const selected = lines.slice(start, start + count);
    if (!selected.length) throw Error("Selected trace window is empty.");
    const rows = selected.map(({ text, index }) => ({
      ...JSON.parse(text),
      source_line: index + 1,
    }));
    const sha = Array.from(
      new Uint8Array(
        await crypto.subtle.digest("SHA-256", new TextEncoder().encode(raw)),
      ),
    )
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    if (expectedHash && sha !== expectedHash)
      throw Error(
        "Bundled Mooncake trace checksum does not match its pinned manifest.",
      );
    await onWorkload(rows, {
      source_id: id,
      source_label: label,
      source_file_sha256: sha,
      source_revision: revision || "uploaded",
      window_start: start,
      source_records: lines.length,
    });
  }
  async function load() {
    setBusy(true);
    try {
      const [response, manifestResponse] = await Promise.all([
        fetch(
          `${import.meta.env.BASE_URL}workloads/mooncake/${source}_trace.jsonl`,
        ),
        fetch(`${import.meta.env.BASE_URL}workloads/mooncake/manifest.json`),
      ]);
      if (!response.ok || !manifestResponse.ok)
        throw Error("Could not load bundled Mooncake trace.");
      const manifest = await manifestResponse.json();
      const expected = manifest.files.find(
        (f: Obj) => f.file === `${source}_trace.jsonl`,
      );
      if (!expected) throw Error("Workload absent from the pinned manifest.");
      await parse(
        await response.text(),
        source,
        `Mooncake ${source} · simulated window`,
        manifest.revision,
        expected.sha256,
      );
    } catch (e: any) {
      onError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="storage-controls">
      <label className="field vertical">
        <span>Retained prefix cache layout</span>
        <select
          aria-label="Retained prefix cache layout"
          value={s.enabled ? s.placement || "inclusive" : "disabled"}
          onChange={(e) => onLayout(e.target.value)}
        >
          <option value="disabled">Off · no retained prefixes</option>
          {Object.entries(layouts).map(([id, item]: [string, any]) => (
            <option key={id} value={id}>
              {item.label}
            </option>
          ))}
        </select>
      </label>
      <p className="fine">
        {s.enabled
          ? layout.description
          : "Choose a single tier or a directed spill layout to model prefix retention."}
      </p>
      <p className="fine">
        Arrows show primary placement → eviction destination. The Dynamo-style
        option writes through to lower tiers with a configurable disk filter.
        These are retained prefix caches. Selecting a layout preserves batch
        sizes, replicas and workload. Fixed uniform batches are supported with
        one replica per pool; variable-length replay requires batch size one.
      </p>
      <button className="secondary" onClick={onDynamo}>
        Apply Dynamo-style setup
      </button>
      <p className="fine">
        Sets active KV to HBM, tiered prefix retention, asynchronous publication
        and GDS. Preserves workload and GPU settings.
      </p>
      <button className="secondary" onClick={onDemo}>
        Load cache demo
      </button>
      <p className="fine">
        The synthetic cache demo uses configured host-staged timing assumptions.
        Published request traces can be loaded below.
      </p>
      {(s.enabled || s.workload?.length > 0) && (
        <>
          <label className="field vertical">
            <span>Prefix publication</span>
            <select
              aria-label="Prefix publication"
              value={s.publication_mode}
              onChange={(e) => onChange("publication_mode", e.target.value)}
            >
              <option value="async">Asynchronous · overlap with serving</option>
              <option value="blocking">Blocking · before decode handoff</option>
            </select>
          </label>
          <p className="fine">
            Asynchronous publication can overlap GPU work. Unfinished blocks
            cannot be reused; source HBM remains reserved until copying and
            handoff finish. Placement commits and each physical I/O path remain
            serialized bandwidth budgets.
          </p>
          {s.publication_mode === "async" && (
            <Field
              label="In-flight prefix publications"
              value={s.max_inflight_publications}
              min={1}
              max={128}
              onChange={(n: number) => onChange("max_inflight_publications", n)}
            />
          )}
          {s.placement === "dynamo" && (
            <>
              <Field
                label="SSD admission minimum uses"
                value={s.disk_min_uses}
                min={1}
                max={128}
                onChange={(n: number) => onChange("disk_min_uses", n)}
              />
              <p className="fine">
                Counts distinct requests referencing a prefix block. This
                simplified filter has no LFU aging or priority policy; it is not
                an exact Dynamo implementation.
              </p>
            </>
          )}
          {(
            [
              ["hbm_cache_GB", "Retained HBM / GPU"],
              ["dram_cache_GB", "Host DRAM cache"],
              ["ssd_cache_GB", "SSD cache capacity"],
            ] as string[][]
          )
            .filter(([key]) => tiers.includes(key.split("_")[0]))
            .map(([key, label]) => (
              <Field
                key={key}
                label={label}
                value={s[key]}
                unit="GB"
                min={0}
                step={0.125}
                onChange={(n: number) => onChange(key, n)}
              />
            ))}
          <div className="server-policy">
            <h3>Workload replay</h3>
            <p className="fine">
              {s.workload?.length
                ? `${s.source_label}: ${s.workload.length} requests. Original lengths and relative timestamps are retained.`
                : "Generated workload: four recurring prefix groups at the configured arrival rate."}
            </p>
            <Field
              label="Trace start record"
              value={start}
              min={0}
              max={1000000}
              onChange={setStart}
            />
            <Field
              label="Trace window requests"
              value={count}
              min={1}
              max={128}
              onChange={setCount}
            />
            <label className="field vertical">
              <span>Published Mooncake workload</span>
              <select
                aria-label="Published Mooncake workload"
                value={source}
                onChange={(e) => setSource(e.target.value)}
              >
                <option value="conversation">
                  Conversation · production sampled
                </option>
                <option value="toolagent">
                  Tool & agent · production sampled
                </option>
                <option value="synthetic">
                  Synthetic · upstream generated
                </option>
              </select>
            </label>
            <button className="secondary" disabled={busy} onClick={load}>
              {busy ? "Loading trace…" : "Load trace window"}
            </button>
            <label className="text-button upload-button">
              Import request JSONL
              <input
                type="file"
                aria-label="Import request JSONL"
                accept=".jsonl,.ndjson"
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  setBusy(true);
                  try {
                    if (file.size > 20e6)
                      throw Error("Request dump must be smaller than 20 MB.");
                    await parse(await file.text(), "uploaded", file.name);
                  } catch (error: any) {
                    onError(error.message);
                  } finally {
                    setBusy(false);
                    e.target.value = "";
                  }
                }}
              />
            </label>
            {s.workload?.length > 0 && (
              <button
                className="text-button"
                onClick={() => {
                  onWorkload([], {
                    source_id: "synthetic-demo",
                    source_label: "Generated repeated-prefix workload",
                    source_file_sha256: null,
                    source_revision: null,
                    window_start: null,
                    source_records: null,
                  });
                }}
              >
                Use generated requests
              </button>
            )}
            <p className="fine">
              Replay preserves configured batch and replica counts.
              Variable-length windows require batch size one. Replay creates
              simulated cache and serving outcomes. The published dumps remain
              unchanged. Anonymized agent traces contain no tool-call dependency
              graph. Interactive limit: 128 requests and 40,000 GPU work
              intervals; long output windows can be large.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
export default function StorageView({
  r,
  onDemo,
}: {
  r: Obj;
  onDemo: () => void;
}) {
  if (!r.storage)
    return (
      <div className="card storage-empty">
        <h2>KV storage tiers</h2>
        <p>
          Model retained prefix KV in GPU HBM, host DRAM and SSD. Ready cache
          hits reduce executed prefill query work; storage and transfer queues
          remain on the request path.
        </p>
        <button className="primary" onClick={onDemo}>
          Load cache demo
        </button>
        <p>
          Run the scenario after loading it. The demo is synthetic and uses
          explicit uncalibrated storage assumptions.
        </p>
        <a href={storageReport} target="_blank" rel="noreferrer">
          Read the published-trace coverage and storage assumptions ↗
        </a>
      </div>
    );
  const s = r.storage,
    stats = s.stats,
    format = (n: number, d = 2) =>
      Number(n || 0).toLocaleString("en-US", { maximumFractionDigits: d });
  return (
    <div className="storage-view">
      <div className="calibration-status">
        <strong>
          KV tiers · simulated{" "}
          {s.workload_source.classification === "generated-scenario"
            ? "generated workload"
            : "trace replay"}
        </strong>
        <span>
          {s.layout?.label || "Custom inclusive copies"} ·{" "}
          {s.workload_source.label} · {s.workload_source.requests} requests ·{" "}
          configured storage/transfer timing assumptions · cold start
          {" · "}
          {s.ssd_transfer_mode === "gds"
            ? "GDS direct path · configured"
            : "host-staged SSD path"}
          {" · "}
          {s.publication_mode} publication
        </span>
      </div>
      <div className="card prefix-scope">
        <div className="card-head">
          <h3>Retained prompt prefixes</h3>
          <button className="secondary" onClick={onDemo}>
            Load cache demo
          </button>
        </div>
        <p className="storage-note">
          This policy controls reuse between requests. Active decode KV has its
          own placement above. Prefill / decode batch caps:{" "}
          {r.config.prefill.batch_size} / {r.config.decode.batch_size}. Layout
          changes preserve them.
        </p>
      </div>
      <div className="economics-grid storage-kpis">
        {[
          ["hit_fraction", "Prefix block hits", stats.hit_fraction * 100, "%"],
          [
            "reused_tokens",
            "Prompt tokens reused",
            stats.reused_tokens || 0,
            "tokens",
          ],
          [
            "ssd_read_bytes",
            "Prefix SSD reads",
            (stats.ssd_read_bytes || 0) / 1e9,
            "GB",
          ],
          [
            "ssd_write_bytes",
            "Prefix SSD writes",
            (stats.ssd_write_bytes || 0) / 1e9,
            "GB",
          ],
          ["ssd_queue_ms", "Prefix SSD waiting", stats.ssd_queue_ms || 0, "ms"],
        ].map(([key, label, value, unit]) => (
          <div className="card" key={String(key)}>
            <span>{label}</span>
            <strong>
              <FormulaValue
                calculation={scaled(
                  storageMetricCalculation(r, String(key)),
                  key === "hit_fraction"
                    ? 100
                    : String(key).endsWith("_bytes")
                      ? 1e-9
                      : 1,
                  String(unit),
                )}
                formulaKey={"storage." + key}
              >
                {format(Number(value))}
              </FormulaValue>
              <small> {unit}</small>
            </strong>
          </div>
        ))}
      </div>
      <div className="card">
        <h3>Capacity and residency</h3>
        <p className="storage-note">
          {s.policy}
          <br />
          HBM cache capacity is reserved per prefill GPU. DRAM and SSD hold the
          aggregate prefill shard payload. Active request KV is pinned
          separately. Active decode backing and its per-token traffic are
          reported above; these counters cover retained-prefix operations only.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Tier</th>
                <th>Placement role</th>
                <th>Cache capacity</th>
                <th>Peak retained</th>
                <th>Prefix blocks served</th>
                <th>Evictions</th>
                <th>Pinned-tier bypass / drops</th>
              </tr>
            </thead>
            <tbody>
              {["hbm", "dram", "ssd"].map((t) => (
                <tr key={t}>
                  <td>{t.toUpperCase()}</td>
                  <td>
                    {!s.layout?.tiers.includes(t)
                      ? "Unused"
                      : s.layout.kind === "inclusive"
                        ? "Retained copy"
                        : s.layout.tiers.indexOf(t) === 0
                          ? "Primary"
                          : `Spill ${s.layout.tiers.indexOf(t)}`}
                  </td>
                  <td>{format(s.capacity_blocks[t], 0)} blocks</td>
                  <td>{format(s.peak_blocks[t], 0)} blocks</td>
                  <td>{format(stats[t + "_hit_blocks"], 0)}</td>
                  <td>{format(s.evictions[t], 0)}</td>
                  <td>{format(s.admission_drops[t], 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="storage-note">
          512 tokens/block · {format(s.block_bytes / 1e6)} MB per aggregate
          cache object · {format(s.per_gpu_block_bytes / 1e6)} MB per GPU
          retained copy. Read the execution box graph for I/O and GPU queue
          lanes.
        </p>
      </div>
      {s.layout?.kind === "spill" && (
        <div className="card">
          <h3>Eviction spill transfers</h3>
          <p className="storage-note">
            {s.stats.spill_blocks || 0} completed block moves ·{" "}
            {s.stats.discarded_blocks || 0} blocks discarded after exhausting
            placement capacity. One retained copy per block; hits stay in their
            current tier. A victim is unavailable while its spill copy is in
            flight.
          </p>
          <div className="table-scroll storage-spill-table">
            <table>
              <thead>
                <tr>
                  <th>Completed ms</th>
                  <th>From</th>
                  <th>To</th>
                  <th>Triggered by request</th>
                </tr>
              </thead>
              <tbody>
                {(s.spill_transitions || [])
                  .slice(-100)
                  .map((move: Obj, i: number) => (
                    <tr key={i}>
                      <td>{format(move.time_ms)}</td>
                      <td>{move.source.toUpperCase()}</td>
                      <td>{move.destination.toUpperCase()}</td>
                      <td>#{move.request_id}</td>
                    </tr>
                  ))}
                {!s.spill_transitions?.length && (
                  <tr>
                    <td colSpan={4}>
                      No spills in this run. Reduce the primary cache capacity
                      to exercise the spill tier.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="storage-note">
            Last 100 moves shown; the export retains all moves. Transfer and SSD
            service intervals appear in the box graph and request latency
            explanations.
          </p>
        </div>
      )}
      <div className="card">
        <h3>Request paths and modeled compute</h3>
        <p className="storage-note">
          Only a ready contiguous prefix can be reused. The final prompt token
          is computed for logits. Blocking publication finishes before handoff;
          asynchronous publication can overlap serving. The first token follows
          the required request-path barriers and decode-memory admission.
        </p>
        <div className="table-scroll storage-request-table">
          <table>
            <thead>
              <tr>
                <th>Request</th>
                <th>Arrival ms</th>
                <th>Input / output</th>
                <th>Reused tokens</th>
                <th>Computed tokens</th>
                <th>Hit tiers</th>
                <th>Restore ms</th>
                <th>Publish ms</th>
                <th>TTFT ms</th>
              </tr>
            </thead>
            <tbody>
              {r.requests.map((q: Obj) => (
                <tr key={q.id}>
                  <td>#{q.id}</td>
                  <td>{format(q.arrival_ms)}</td>
                  <td>
                    {q.input_tokens} / {q.output_tokens}
                  </td>
                  <td>
                    <FormulaValue
                      calculation={cacheDecisionCalculation(r, q)}
                      formulaKey={"storage.request." + q.id + ".reuse"}
                    >
                      {format(q.cache.reused_tokens, 0)}
                    </FormulaValue>
                  </td>
                  <td>
                    <FormulaValue
                      calculation={calc(
                        "Computed prompt tokens",
                        "Computed = input_tokens − reused_tokens",
                        [
                          v(
                            "input_tokens",
                            "Prompt length",
                            q.input_tokens,
                            "tokens",
                          ),
                          v(
                            "reused_tokens",
                            "Restored prefix",
                            q.cache.reused_tokens,
                            "tokens",
                            () => cacheDecisionCalculation(r, q),
                          ),
                        ],
                        q.cache.computed_tokens,
                        "tokens",
                      )}
                    >
                      {format(q.cache.computed_tokens, 0)}
                    </FormulaValue>
                  </td>
                  <td>
                    {Object.entries(q.cache.hit_tiers)
                      .map(([t, n]) => `${t}: ${n}`)
                      .join(", ") || "miss"}
                  </td>
                  <td>
                    {format(q.cache.cache_ready_ms - q.cache.memory_ready_ms)}
                  </td>
                  <td>
                    {format(
                      q.cache.publication_end_ms - q.cache.gpu_prefill_end_ms,
                    )}
                  </td>
                  <td>
                    <FormulaValue
                      calculation={storageRequestCalculation(r, q, "ttft_ms")}
                      formulaKey={"storage.request." + q.id + ".ttft"}
                    >
                      {format(q.ttft_ms)}
                    </FormulaValue>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="card">
        <h3>Trace evidence and timing assumptions</h3>
        <p className="storage-note">
          Published Mooncake request traces supply arrival timestamps,
          input/output lengths and prefix identities. They do not contain
          measured SSD reads, writes, host–GPU transfers or GDS timings. Storage
          service uses the configured bandwidth and latency values; no device
          timing error bound is available from these traces.
        </p>
        <p className="storage-note">
          Runtime hits come from the cache state. Fitted hit curves are separate
          forecasts trained on trace-replayed LRU references; they never
          manufacture cache hits.
        </p>
        {s.hit_forecast && (
          <p className="storage-note">
            {r.config.storage.source_id} policy-curve forecast:{" "}
            <FormulaValue
              calculation={{
                ...calc(
                  "Fitted policy hit forecast",
                  s.hit_forecast.fit.family === "capped-linear"
                    ? "H = H_max × a × min(1,x/s)"
                    : s.hit_forecast.fit.family === "weibull"
                      ? "H = H_max × a × [1−exp(−(x/s)^p)]"
                      : "H = H_max × a × [1−exp(−x/s)]",
                  [
                    v(
                      "x",
                      "Capacity / unique blocks",
                      s.hit_forecast.effective_capacity_blocks /
                        Math.max(1, s.hit_forecast.working_set_blocks),
                    ),
                    v(
                      "H_max",
                      "Workload compulsory-miss upper bound",
                      s.hit_forecast.max_hit_fraction,
                    ),
                    v(
                      "a",
                      "Fitted asymptote multiplier",
                      s.hit_forecast.fit.coefficients[0],
                    ),
                    v(
                      "s",
                      "Fitted capacity scale",
                      s.hit_forecast.fit.coefficients[1],
                    ),
                    ...(s.hit_forecast.fit.coefficients.length > 2
                      ? [
                          v(
                            "p",
                            "Fitted exponent",
                            s.hit_forecast.fit.coefficients[2],
                          ),
                        ]
                      : []),
                  ],
                  s.hit_forecast.prediction,
                  "fraction",
                ),
                summary: s.hit_forecast.scope,
                links: [
                  {
                    label: "Held-region validation and data coverage",
                    url: traceReuseReport,
                  },
                ],
              }}
            >
              {format(s.hit_forecast.prediction * 100)}%
            </FormulaValue>{" "}
            · held-out MAE {format(s.hit_forecast.fit.validation.mae_pp)}{" "}
            percentage points.{" "}
            {s.hit_forecast.outside_request_range
              ? "This short replay window is outside the fitted request-count range. "
              : ""}
            The forecast assumes instantaneous publication; actual I/O queues
            can reduce reuse.
          </p>
        )}
        <a href={storageReport} target="_blank" rel="noreferrer">
          Published request traces, simulation policies and timing assumptions
          ↗
        </a>
        <p className="storage-note">
          Published request traces:{" "}
          <a
            href="https://github.com/kvcache-ai/Mooncake"
            target="_blank"
            rel="noreferrer"
          >
            Mooncake contributors
          </a>
          . Trace-policy replay, tier modeling and ServeScope integration:{" "}
          <a href="https://github.com/ryeii" target="_blank" rel="noreferrer">
            ryeii (Ryan Z An)
          </a>
          .
        </p>
      </div>
      <div className="card">
        <h3>Scope of this run</h3>
        <ul>
          {s.limitations.map((text: string) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
