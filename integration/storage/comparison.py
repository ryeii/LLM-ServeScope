"""Reproduce layout comparisons while holding workload and serving configuration fixed."""
import copy,json
from pathlib import Path
from backend.model import DEFAULT,simulate
from backend.storage import CACHE_LAYOUTS
ROOT=Path(__file__).resolve().parents[2]

def compact(result):
    return dict(metrics=result['metrics'],config=result['config'],reused_tokens=result.get('storage',{}).get('stats',{}).get('reused_tokens',0),storage_service_ms=sum(e['duration_ms'] for e in result['timeline'] if e.get('kind')=='storage_io'))

def main():
    base=copy.deepcopy(DEFAULT)
    base['storage'].update(enabled=False,placement='inclusive',profile='manual',
        ssd_transfer_mode='host-staged',publication_mode='blocking')
    baseline=simulate(base);after={}
    for layout in CACHE_LAYOUTS:
        c=copy.deepcopy(baseline['config']);c['storage'].update(enabled=True,placement=layout)
        after[layout]=compact(simulate(c))
    assert after['none']['metrics']==baseline['metrics']
    single=copy.deepcopy(baseline['config'])
    for phase in ['prefill','decode']:single[phase]['batch_size']=1
    data=dict(kind='behavioral-simulator-regression-not-hardware-measurements',baseline=compact(baseline),batch_one_without_retention=compact(simulate(single)),layouts=after)
    (ROOT/'examples/prefix-layout-comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    text='''# Prefix-layout comparisons and the batching regression

These are **ServeScope simulation results**, not GPU/SSD measurements. The original UI
silently forced prefill/decode batch caps from four to one when a layout was selected.
That made the comparison invalid: the disabled default and the tier runs had different
batching. Selecting a layout now preserves model, workload, batch caps and replicas.

Uniform-length requests use the same fixed microbatch formation, GPU operation shapes,
decode splitting and resource queues. Prefix mode reserves active KV for a full group
atomically. A fixed rectangular prefill batch uses the minimum restored prefix among its
members; a higher-hit member recomputes excess tokens. Variable-length request replay
still requires batch size one. Unsupported replica counts are rejected explicitly.
There is no automatic reduction of batching or replica counts. Imported records
keep their original arrivals and token lengths when retention is turned off;
retention-off replay allocates no retained cache capacity.

## Reproduction

Llama 3.1 70B, BF16, separate H100 TP4 prefill/decode pools, 16 requests,
2048 input tokens, 64 output tokens, 8 requests/s. Original/default batch caps: 4/4.
The configured SSD profile is manual: read 3 GB/s + 100 us per I/O chunk; write
2 GB/s + 200 us per chunk; chunk size 1 MiB. HBM prefix capacity 0.125 GB/GPU,
host DRAM 0.25 GB aggregate, SSD 4 GB aggregate. Device timings are configured
assumptions. Publication and spill costs block before handoff in this comparison.

| Configuration | Prefill / decode caps | Generation throughput (tok/s) | Reused prompt tokens |
| --- | --- | ---: | ---: |
'''
    for name,r,batch in [('Original disabled baseline',data['baseline'],'4 / 4'),('Batch change alone, no prefix retention',data['batch_one_without_retention'],'1 / 1')]:
        text+=f"| {name} | {batch} | {r['metrics']['throughput_tps']:.6f} | {r['reused_tokens']} |\n"
    for layout,r in after.items():text+=f"| {CACHE_LAYOUTS[layout]['label']} | 4 / 4 | {r['metrics']['throughput_tps']:.6f} | {r['reused_tokens']} |\n"
    text+='''
“No retained prefix cache” matches **every original default performance metric and
per-request token timestamp exactly**. Its optional active-memory ledger reports
observed allocations, while disabled mode retains the original conservative resident
KV accounting. Under memory pressure, explicit group admission can introduce waits;
this regression proves equality for the documented setup, not every possible policy.

## What SSD selection means

This controlled comparison holds **active attention KV in GPU HBM**. The layout controls
retained prompt prefixes used between requests. A hit restores ready prefix blocks
before prefill; new retained blocks and evicted blocks incur publication/spill work.
Decode does not stream its working KV from SSD at each output token. Thus SSD prefix
retention is not a simulation of active attention-KV offload.

Active decode placement is now a separate control, with explicit per-step reads,
append writes, staging capacity and shared I/O queues. See the
[active-KV model](active-kv-model.md) for equations and its unvalidated serialized
streaming assumptions. No active-offload accuracy claim is inferred from these
prefix-cache experiments or from the published request traces.

A prefix layout can have a modest effect when decoding dominates, few ready prefixes
are reused, capacity is small or arrival rate limits the finite run. Prefix reuse also
saves prefill work while reads/publication add cost, so throughput need not be ordered
solely by the name of the retention tier.

[Resolved configurations and results](../examples/prefix-layout-comparison.json)
· [Reproduction script](../integration/storage/comparison.py)

```bash
.venv/bin/python -B -m integration.storage.comparison
.venv/bin/python -B -m unittest tests.test_storage
```
'''
    (ROOT/'docs/prefix-layout-comparison.md').write_text(text)

if __name__=='__main__':main()
