"""Analyze only the published Mooncake request traces; no device benchmark.

Replay is an instantaneous-publication LRU reference, not measured cache hits.
Temporal holdouts keep overlapping windows and capacities in the same fold.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT/'third_party/mooncake-traces'
OUT = ROOT/'docs/trace-reuse'
FAMILIES = {'capped-linear': 2, 'exponential': 2, 'weibull': 3}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def prefix_keys(row):
    parent = ''
    keys = []
    for h in row['hash_ids'][:row['input_length']//512]:
        parent = hashlib.sha256((parent+':'+str(h)).encode()).hexdigest()
        keys.append(parent)
    return keys


def replay():
    manifest = json.loads((TRACE/'manifest.json').read_text())
    points = []
    for item in manifest['files']:
        source = TRACE/item['file']
        if digest(source) != item['sha256']:
            raise ValueError('Published trace checksum mismatch: '+item['file'])
        name = item['file'].removesuffix('_trace.jsonl')
        requests = [json.loads(line) for line in source.read_bytes().splitlines()]
        for window, region in enumerate(np.array_split(np.array(requests, dtype=object), 6)):
            for length in sorted({8, 32, 128, 512, len(region)}):
                groups = [prefix_keys(r) for r in region[:length]]
                unique = len({key for group in groups for key in group}) or 1
                total = sum(map(len, groups))
                potential = max(0, (total-unique)/max(1, total))
                for capacity in [0,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,65536]:
                    cache = collections.OrderedDict()
                    hits = 0
                    for keys in groups:
                        for key in keys:
                            if key not in cache:
                                break
                            hits += 1
                        for key in keys:
                            if not capacity:
                                continue
                            cache[key] = None
                            cache.move_to_end(key)
                            if len(cache) > capacity:
                                cache.popitem(last=False)
                    points.append(dict(workload=name,window=window,capacity_blocks=capacity,
                        working_set_blocks=unique,max_hit_fraction=potential,requests=len(groups),
                        eligible_blocks=total,hit_blocks=hits,hit_fraction=hits/max(1,total)))
        print('Analyzed published trace', name, len(requests), 'requests', flush=True)
    dump(OUT/'hit-observations.json', dict(kind='published-request-trace-policy-replay',
        policy='Cold start, request-order LRU, instantaneous publication, chained prefix identities, 512-token blocks. Six temporal regions; all nested windows/capacities share their region holdout. No device timings or measured hardware cache hits.',
        source_manifest=manifest, observations=points))


def predict(rows, family, coefficients):
    x = np.array([r['capacity_blocks']/r['working_set_blocks'] for r in rows])
    a, scale = coefficients[:2]
    bound = np.array([r['max_hit_fraction'] for r in rows])
    if family == 'capped-linear':
        return bound*a*np.minimum(1,x/scale)
    power = coefficients[2] if family == 'weibull' else 1
    return bound*a*(-np.expm1(-np.minimum(700,(x/scale)**power)))


def fit(rows, family):
    target = np.array([r['hit_fraction'] for r in rows])
    result = least_squares(lambda coefficients: predict(rows,family,coefficients)-target,
        [.8,.1]+([1.] if family=='weibull' else []),
        bounds=([0,1e-6]+([.1] if family=='weibull' else []),[1,100]+([8] if family=='weibull' else [])),max_nfev=2000)
    return result.x.tolist()


def error(rows, predictions):
    residual = abs(predictions-np.array([r['hit_fraction'] for r in rows]))*100
    return dict(mae_pp=float(residual.mean()),rmse_pp=float(np.sqrt((residual**2).mean())),p95_pp=float(np.percentile(residual,95)))


def cross_validate(rows, family):
    predictions = np.zeros(len(rows))
    for region in sorted({r['window'] for r in rows}):
        training = [r for r in rows if r['window'] != region]
        indexes = [i for i,r in enumerate(rows) if r['window'] == region]
        predictions[indexes] = predict([rows[i] for i in indexes],family,fit(training,family))
    return predictions


def select(rows):
    scores = {family:error(rows,cross_validate(rows,family)) for family in FAMILIES}
    best = min(value['mae_pp'] for value in scores.values())
    family = min([f for f in FAMILIES if scores[f]['mae_pp'] <= best+.5],key=lambda f:(FAMILIES[f],scores[f]['mae_pp']))
    return family, scores


def train():
    data = json.loads((OUT/'hit-observations.json').read_text())
    artifact = dict(schema_version='servescope.prefix-hit-fit.v1',hit={},
        scope='Published request-trace LRU replay only; no storage or transfer timing and no measured hardware cache hits.',
        sources={'hit_sha256':digest(OUT/'hit-observations.json'),'trace_manifest_sha256':digest(TRACE/'manifest.json')},
        training_script_sha256=digest(Path(__file__)),runtime_sha256=digest(ROOT/'backend/prefix_reuse.py'))
    validation, predictions = {}, {}
    for name in ['conversation','toolagent','synthetic']:
        rows = [r for r in data['observations'] if r['workload'] == name]
        family, candidates = select(rows)
        nested = np.zeros(len(rows))
        models = {}
        for region in range(6):
            training = [r for r in rows if r['window'] != region]
            selected, _ = select(training)
            coefficients = fit(training, selected)
            indexes = [i for i,r in enumerate(rows) if r['window'] == region]
            nested[indexes] = predict([rows[i] for i in indexes],selected,coefficients)
            models[str(region)] = dict(family=selected,coefficients=coefficients,training_regions=sorted({r['window'] for r in training}))
        artifact['hit'][name] = dict(family=family,coefficients=fit(rows,family),observations=len(rows),
            validation=error(rows,nested),min_requests=min(r['requests'] for r in rows),max_requests=max(r['requests'] for r in rows))
        validation[name] = dict(candidates=candidates,selected=family,nested=error(rows,nested),outer_models=models)
        predictions[name] = [dict(row,predicted_hit=float(value)) for row,value in zip(rows,nested)]
    artifact['payload_sha256'] = hashlib.sha256(json.dumps(artifact,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dump(ROOT/'backend/prefix-hit-fit.json',artifact)
    dump(OUT/'validation.json',validation)
    dump(OUT/'held-out-predictions.json',predictions)
    print(json.dumps({name:value['validation'] for name,value in artifact['hit'].items()},indent=2))


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    predictions = json.loads((OUT/'held-out-predictions.json').read_text())
    fig, axes = plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for column,(name,rows) in enumerate(predictions.items()):
        for line,length in enumerate([128,None]):
            ax = axes[line,column]
            for region in range(6):
                maximum = max(r['requests'] for r in rows if r['window']==region)
                part = [r for r in rows if r['window']==region and r['requests']==(length or maximum)]
                x = [r['capacity_blocks']/r['working_set_blocks'] for r in part]
                ax.plot(x,[100*r['hit_fraction'] for r in part],color='#527ab2',alpha=.4)
                ax.plot(x,[100*r['predicted_hit'] for r in part],color='#1a8a6c',alpha=.7,linestyle='--')
            ax.set(xscale='symlog',ylim=(0,100),xlabel='Capacity / unique prefix blocks',ylabel='Prefix-block hit (%)',title=name+(' · 128 requests' if length else ' · full region'))
    fig.suptitle('Published request traces: explicit LRU replay (blue), held-out predictions (green)\nPolicy approximation only; no storage or transfer timing observations')
    for extension in ['png','svg']:
        target = OUT/f'hit-fit.{extension}'
        fig.savefig(target,dpi=170)
        if extension == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['replay','train','figures','all'])
    action = parser.parse_args().action
    for name,function in [('replay',replay),('train',train),('figures',figures)]:
        if action in (name,'all'):
            function()
