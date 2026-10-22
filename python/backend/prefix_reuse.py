"""Trace-derived workload potential; no simulated cache policy or device timing."""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def evidence():
    path = Path(__file__).with_name('prefix-reuse.json')
    data = json.loads(path.read_text())
    payload = dict(data)
    expected = payload.pop('payload_sha256')
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual != expected:
        raise ValueError('Prefix-reuse evidence checksum mismatch.')
    return data


def assumption(config):
    source = config.get('prefix_reuse_source','none')
    record = evidence()['workloads'].get(source)
    fraction = record['fraction'] if record else 0
    reused = min(config['input_tokens']-1, math.floor(config['input_tokens']*fraction))
    return dict(source=source, fraction=fraction, reused_tokens_per_request=reused,
        computed_tokens_per_request=config['input_tokens']-reused,
        effective_fraction=reused/config['input_tokens'],
        evidence=record, payload_sha256=evidence()['payload_sha256'],
        scope='Trace-derived reuse potential applied uniformly. Cached prefixes are assumed already ready in HBM; no readiness, eviction or cache-policy simulation. This is not an observed hardware hit rate or validated end-to-end serving result.')
