import FormulaValue from "./FormulaValue";
import { calc, v } from "./calculations";
type Obj = Record<string, any>;
const report =
  "https://github.com/ryeii/LLM-ServeScope/blob/main/docs/prefix-reuse/README.md";
const percent = (n: number) => (100 * n).toFixed(2) + "%";

export function PrefixReuseControls({
  source,
  evidence,
  onChange,
}: {
  source: string;
  evidence: Obj;
  onChange: (source: string) => void;
}) {
  const record = evidence.workloads[source];
  return (
    <div className="prefix-reuse-controls">
      <label className="field vertical">
        <span>Prefix reuse assumption</span>
        <select
          aria-label="Prefix reuse assumption"
          value={source}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="none">No reuse · cold baseline</option>
          {Object.entries(evidence.workloads).map(([id, raw]) => {
            const row = raw as Obj;
            return (
              <option key={id} value={id}>
                Mooncake {row.label} · {percent(row.fraction)} potential
              </option>
            );
          })}
        </select>
      </label>
      <p className="fine">
        {record
          ? `${percent(record.fraction)} of input tokens have reusable earlier-arrival prefixes in the published trace, under unlimited history. We assume that fraction is already ready in HBM for each scenario request. This is potential, not measured cache hits.`
          : "Complete prompts are computed. All KV stays in HBM. Optional Mooncake estimates describe workload reuse potential, not a cache policy or measured serving speedup."}
      </p>
      <a href={report} target="_blank" rel="noreferrer">
        Trace analysis and assumption scope ↗
      </a>
    </div>
  );
}

export function PrefixReuseSummary({ r }: { r: Obj }) {
  const reuse = r.prefix_reuse,
    source = reuse?.evidence;
  if (!reuse || reuse.source === "none") return null;
  return (
    <section
      className="card prefix-reuse-summary"
      aria-label="Prefix reuse assumption in this run"
    >
      <div className="card-head">
        <div>
          <h3>Trace-informed prefix reuse</h3>
          <p>Mooncake {source.label} · ideal HBM availability assumption</p>
        </div>
      </div>
      <p>
        <FormulaValue
          formulaKey="prefix.reused_tokens"
          calculation={calc(
            "Assumed reusable prompt tokens",
            "reused = min(L_in − 1, floor(L_in × f_trace))",
            [
              v("L_in", "Full prompt length", r.config.input_tokens, "tokens"),
              v(
                "f_trace",
                "Token-weighted earlier-arrival reuse potential",
                reuse.fraction,
                "",
                () => ({
                  ...calc(
                    "Published trace reuse potential",
                    "f_trace = Σ reusable_input_tokens / Σ input_tokens",
                    [
                      v(
                        "reusable_input_tokens",
                        "Matching earlier-arrival prefix tokens",
                        source.reusable_tokens,
                        "tokens",
                      ),
                      v(
                        "input_tokens",
                        "All published input tokens",
                        source.input_tokens,
                        "tokens",
                      ),
                    ],
                    reuse.fraction,
                    "",
                    "Cold history, complete 512-token leading blocks, no reuse within tied arrivals, unlimited retention and no readiness delay. This is not measured hardware hits.",
                  ),
                  source: `Mooncake ${source.label} · SHA-256 ${source.source_sha256}`,
                  links: [
                    { label: "Reproducible trace analysis", url: report },
                  ],
                }),
              ),
            ],
            reuse.reused_tokens_per_request,
            "tokens",
            "This scalar fraction is applied uniformly to the scenario; original trace arrivals and lengths are not replayed. Cached KV is assumed already ready in HBM. Whole-system speedup has no measured validation.",
          )}
        >
          <strong>
            {reuse.reused_tokens_per_request.toLocaleString()} reused tokens
          </strong>
        </FormulaValue>{" "}
        per request; {reuse.computed_tokens_per_request.toLocaleString()} new
        query tokens. Attention retains the full{" "}
        {r.config.input_tokens.toLocaleString()}-token context. KV capacity,
        decode work and handoff still use the full context.
      </p>
      <p className="fine">
        Unlimited-history potential does not establish feasibility in a finite
        cache. No persistent cache pool, eviction, readiness or transfer policy
        is simulated. GPU timing uses the fitted functions at the resulting
        shapes; their errors do not validate the resulting serving speedup.
      </p>
      <a href={report} target="_blank" rel="noreferrer">
        Source counts, methodology and limits ↗
      </a>
    </section>
  );
}
