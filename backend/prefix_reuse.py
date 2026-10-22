"""Prefix-reuse forecasts fitted only to published request-trace replay.

These curves approximate an explicit LRU policy. They contain no device timing,
storage benchmark observations or measured hardware cache hits.
"""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def hit_fit_dataset():
    path = Path(__file__).with_name('prefix-hit-fit.json')
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    payload = dict(data)
    expected = payload.pop('payload_sha256')
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual != expected:
        raise ValueError('Prefix-reuse fit checksum mismatch.')
    return data


def hit_prediction(fit, capacity_blocks, unique_blocks, max_hit_fraction=1):
    a, scale = fit['coefficients'][:2]
    x = capacity_blocks / max(1, unique_blocks)
    if fit['family'] == 'capped-linear':
        return max_hit_fraction * a * min(1, x / scale)
    power = fit['coefficients'][2] if fit['family'] == 'weibull' else 1
    return max_hit_fraction * a * (-math.expm1(-min(700, (x / scale) ** power)))
