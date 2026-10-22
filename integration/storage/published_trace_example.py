"""Reproduce a serving scenario using an unchanged published request window."""
import copy
import hashlib
import json
from pathlib import Path
from backend.model import DEFAULT, simulate

ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = ROOT/'third_party/mooncake-traces'
    manifest = json.loads((folder/'manifest.json').read_text())
    item = next(f for f in manifest['files'] if f['file']=='conversation_trace.jsonl')
    raw = (folder/item['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:
        raise ValueError('Published request trace checksum mismatch.')
    rows = [dict(json.loads(line),source_line=i+1) for i,line in enumerate(raw.splitlines()[:4])]
    config = copy.deepcopy(DEFAULT)
    for phase in ('prefill','decode'):
        config[phase]['batch_size']=1
    config['storage'].update(profile='manual',workload=rows,source_id='conversation',
        source_label='Published Mooncake conversation · first four requests · simulated replay',
        source_file_sha256=item['sha256'],source_revision=manifest['revision'])
    result = simulate(config)
    summary = dict(kind='simulated-published-request-replay-not-device-measurements',
        source=result['storage']['workload_source'],metrics=result['metrics'],
        workload_totals=result['workload_totals'],timing_evidence=result['storage']['timing_evidence'],
        requests=[dict(id=q['id'],arrival_ms=q['arrival_ms'],input_tokens=q['input_tokens'],
            output_tokens=q['output_tokens'],computed_tokens=q['cache']['computed_tokens'],
            reused_tokens=q['cache']['reused_tokens']) for q in result['requests']])
    for name,data in [('mooncake-conversation-scenario.json',config),('mooncake-conversation-result-summary.json',summary)]:
        (ROOT/'examples'/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
