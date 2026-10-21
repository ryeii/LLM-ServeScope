"""Reproduce independent active/prefix placement comparisons; these are predictions."""
import copy, json
from pathlib import Path
from backend.model import DEFAULT, simulate
from backend.active_kv import PLACEMENTS
ROOT = Path(__file__).resolve().parents[2]
def save(path, value):
    (ROOT/path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def main():
    rows=[]
    for prefix in ['disabled', 'ssd-only']:
        for active in PLACEMENTS:
            c=copy.deepcopy(DEFAULT)
            c['active_kv'].update(placement=active)
            c['storage'].update(enabled=prefix!='disabled', placement='ssd-only')
            r=simulate(c)
            rows.append(dict(active=active, prefix=prefix, config=r['config'], metrics=r['metrics'], active_stats=r.get('active_kv',{}).get('stats',{}), prefix_stats=r.get('storage',{}).get('stats',{})))
    save('examples/active-kv-comparison.json',dict(kind='behavioral-predictions-not-hardware-measurements', results=rows))
    c=copy.deepcopy(DEFAULT)
    c.update(model_id='llama8',requests=8,input_tokens=1024,output_tokens=8,arrival_rate=8)
    for phase in ['prefill','decode']:c[phase].update(tp=1,pp=1,ep=1,batch_size=2)
    c['active_kv'].update(placement='ssd',hbm_resident_GB=.05)
    c['storage'].update(enabled=True,placement='ssd-only',hbm_cache_GB=0,dram_cache_GB=0,ssd_cache_GB=4,profile='manual')
    r=simulate(c)
    save('examples/active-kv-scenario.json',c)
    save('examples/active-kv-result.json',r)
    table=['| Active decode placement | Retained prefixes | Generation tok/s | Mean TPOT ms | Remote decode KV reads GB |', '| --- | --- | ---: | ---: | ---: |']
    for row in rows:
        table.append(f"| {row['active']} | {row['prefix']} | {row['metrics']['throughput_tps']:.4f} | {row['metrics']['tpot_mean_ms']:.4f} | {row['active_stats'].get('attention_read_bytes',0)/1e9:.4f} |")
    (ROOT/'docs/active-kv-comparison.md').write_text('# Independent KV placement comparison\n\nThese are model predictions, **not measured serving benchmarks or validation**.\nAll runs hold the original default model, hardware, workload, batch caps and manual I/O settings fixed. The default hybrid resident budget is 1 GB per decode GPU; pure backing modes retain no active KV in HBM beyond staging.\n\n'+'\n'.join(table)+'\n\nSee [model assumptions](active-kv-model.md), [resolved configurations and numbers](../examples/active-kv-comparison.json), and [generator](../integration/storage/active_comparison.py).\n\n```bash\n.venv/bin/python -B -m integration.storage.active_comparison\n```\n')
if __name__=='__main__':main()
