"""Reproducible configured GDS/staged predictions, not hardware measurements."""
import copy
import json
from pathlib import Path
from backend.model import DEFAULT, simulate

ROOT = Path(__file__).resolve().parents[2]


def save(name, value):
    (ROOT/'examples'/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    save('dynamo-default-result.json', simulate(DEFAULT))
    c = copy.deepcopy(DEFAULT)
    c.update(model_id='llama8', requests=8, input_tokens=1024,
             output_tokens=8, arrival_rate=2)
    for phase in ('prefill', 'decode'):
        c[phase].update(tp=1, pp=1, ep=1, replicas=1, batch_size=1)
    c['storage'].update(placement='ssd-only', publication_mode='blocking',
                        hbm_cache_GB=0, dram_cache_GB=0, disk_min_uses=1)
    save('gds-scenario.json', c)
    rows = []
    for mode in ('gds', 'host-staged'):
        c['storage']['ssd_transfer_mode'] = mode
        r = simulate(c)
        rows.append(dict(path=mode, config=r['config'], metrics=r['metrics'],
                         storage_stats=r['storage']['stats']))
    save('gds-comparison.json', dict(kind='configured-model-predictions-not-measurements', results=rows))
    # Repository examples use explicit configured timing assumptions only.
    for name in ('storage-tier-result.json', 'active-kv-result.json'):
        old = json.loads((ROOT/'examples'/name).read_text())
        old['config']['storage']['profile'] = 'manual'
        new = simulate(old['config'])
        save(name, new)
    print(json.dumps({row['path']: row['metrics'] for row in rows}, indent=2))


if __name__ == '__main__':
    main()
