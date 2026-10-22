"""Audit published prefix reuse potential; no device or cache-policy model."""
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / 'third_party/mooncake-traces'


def prefix_keys(row):
    parent, keys = '', []
    for value in row['hash_ids'][:row['input_length'] // 512]:
        parent = hashlib.sha256((parent + ':' + str(value)).encode()).hexdigest()
        keys.append(parent)
    return keys


def analyze(rows):
    """Cold, unlimited earlier-timestamp history; no reuse within tied arrivals."""
    if not rows or any(a['timestamp'] > b['timestamp'] for a, b in zip(rows, rows[1:])):
        raise ValueError('A nonempty timestamp-ordered request trace is required.')
    seen = set()
    total = reused = hit_requests = 0
    windows = [dict(requests=0, input_tokens=0, reusable_tokens=0) for _ in range(6)]
    origin, end = rows[0]['timestamp'], rows[-1]['timestamp']
    for stamp, batch in itertools.groupby(rows, key=lambda row: row['timestamp']):
        additions = set()
        for row in batch:
            keys, matched = prefix_keys(row), 0
            for key in keys:
                if key not in seen:
                    break
                matched += 1
            tokens = min(matched * 512, row['input_length'] - 1)
            total += row['input_length']
            reused += tokens
            hit_requests += tokens > 0
            window = min(5, int(6 * (stamp - origin) / max(1, end - origin)))
            windows[window]['requests'] += 1
            windows[window]['input_tokens'] += row['input_length']
            windows[window]['reusable_tokens'] += tokens
            additions.update(keys)
        seen.update(additions)
    for window in windows:
        window['fraction'] = window['reusable_tokens'] / max(1, window['input_tokens'])
    return dict(requests=len(rows), input_tokens=total, reusable_tokens=reused,
                fraction=reused/total, requests_with_reuse=hit_requests,
                unique_prefix_blocks=len(seen), temporal_windows=windows)


def main():
    manifest = json.loads((TRACE/'manifest.json').read_text())
    workloads = {}
    for item in manifest['files']:
        raw = (TRACE/item['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Published trace checksum mismatch: '+item['file'])
        result = analyze([json.loads(line) for line in raw.splitlines()])
        assert result['requests'] == item['requests']
        name = item['file'].removesuffix('_trace.jsonl')
        workloads[name] = dict(label=item['label'], origin=item['origin'],
            source_file=item['file'], source_sha256=item['sha256'], **result)
    data = dict(schema_version='servescope.prefix-reuse.v1',
        definition='Input-token-weighted earlier-arrival prefix reuse potential; cold start, unlimited history, no same-timestamp reuse, complete 512-token blocks, at least one recomputed token.',
        scope='Workload evidence only. Prefix availability in HBM is an explicit ideal assumption; no eviction, transfer or end-to-end latency validation.',
        source_url=manifest['repository']+'/tree/'+manifest['revision']+'/FAST25-release',
        source_revision=manifest['revision'],
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),workloads=workloads)
    data['payload_sha256'] = hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (ROOT/'backend/prefix-reuse.json').write_text(json.dumps(data,indent=2)+'\n')
    print('\n'.join(f"{v['label']}: {v['requests']:,} requests; {v['reusable_tokens']:,}/{v['input_tokens']:,} reusable/input tokens = {v['fraction']*100:.4f}%" for v in workloads.values()))


if __name__ == '__main__':
    main()
