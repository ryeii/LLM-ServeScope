"""Reproducible storage-path measurements and trace-replay hit-curve study.
Offline dependencies: NumPy/SciPy/Matplotlib. Runtime coefficients need stdlib only.
"""
from pathlib import Path
import argparse, collections, contextlib, hashlib, io, json, math, os, platform, random, statistics, sys, tempfile, time
import numpy as np
from scipy.optimize import nnls, least_squares
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/storage-2026-10-20'
TRACE=ROOT/'third_party/mooncake-traces'

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def measure(storage_parent=None):
    # Exact archived Mooncake DiskHashTable, with a bounded temporary file.
    sys.path.insert(0,str(ROOT/'third_party/mooncake-storage/source/benchmarks/storage_benchmark_v1'))
    from storage.disk import DiskHashTable
    sizes=[16384,32768,65536,131072,262144,524288,1048576,2097152]
    order=[(trial,size) for trial in range(4) for size in sizes]
    random.Random(20261020).shuffle(order)
    observations=[]
    with tempfile.TemporaryDirectory(prefix='servescope-storage-',dir=storage_parent or ('/private/tmp' if Path('/private/tmp').is_dir() else tempfile.gettempdir())) as folder:
        for trial,size in order:
            with contextlib.redirect_stdout(io.StringIO()):
                store=DiskHashTable(folder+f'/{trial}-{size}',size,max_pages=2,file_mode='single',fsync_mode='always')
                for _ in range(2):store.write(0);store.read(0)
                for rep in range(12):
                    for op in ['write','read']:
                        latency=getattr(store,op)(rep%2)
                        if latency is None:raise RuntimeError('Mooncake storage operation failed')
                        observations.append(dict(operation=op,bytes=size,trial=trial,repeat=rep,latency_ms=latency,bandwidth_MBps=size/latency/1000))
                store.close()
    data=dict(kind='local-measured-storage-path',date='2026-10-20',system=platform.system(),machine=platform.machine(),os_release=platform.release(),python=platform.python_version(),source_revision='7e4fc11ee0fe9553a82a408009ea2b499c76842e',source_sha256=digest(ROOT/'third_party/mooncake-storage/source/benchmarks/storage_benchmark_v1/storage/disk.py'),protocol='Archived Mooncake DiskHashTable; single file, 2 slots, fsync=always; serial QD1; warm buffered reads following writes; 2 warmup pairs excluded; 4 shuffled trials x 8 sizes x 12 pairs. No O_DIRECT, cold-cache control, SSD identification or power-loss durability verification. fsync is the OS call, not a device persistence certification.',observations=observations)
    dump(OUT/'io-observations.json',data)
    print('Measured',len(observations),'I/O observations',flush=True)

def replay_curves():
    capacities=[0,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,65536]
    points=[]
    for workload in ['conversation','toolagent','synthetic']:
        raw=(TRACE/f'{workload}_trace.jsonl').read_bytes();requests=[json.loads(x) for x in raw.splitlines()]
        # Keep source order and count cold-start outcomes for all requests.
        for window,region in enumerate(np.array_split(np.array(requests,dtype=object),6)):
            # Nested shorter windows stay in the same temporal hold-out group.
            for length in sorted(set([8,32,128,512,len(region)])):
                part=region[:length]
                groups=[r['hash_ids'][:r['input_length']//512] for r in part]
                unique=len(set(x for g in groups for x in g)) or 1
                total=sum(map(len,groups));potential=max(0,(total-unique)/max(1,total))
                for capacity in capacities:
                    cache=collections.OrderedDict();hit=0
                    for keys in groups:
                        matched=0
                        for k in keys:
                            if k not in cache:break
                            matched+=1
                        hit+=matched
                        for k in keys:
                            if not capacity:continue
                            cache[k]=None;cache.move_to_end(k)
                            if len(cache)>capacity:cache.popitem(last=False)
                    points.append(dict(workload=workload,window=window,capacity_blocks=capacity,working_set_blocks=unique,max_hit_fraction=potential,requests=len(groups),warmup_requests=0,eligible_blocks=total,hit_blocks=hit,hit_fraction=hit/max(1,total)))
        print('Replayed',workload,len(requests),'requests',flush=True)
    dump(OUT/'hit-observations.json',dict(kind='trace-replayed-policy-reference',policy='Cold-start, request-order, instantaneous-publication, fully associative LRU; longest contiguous complete 512-token prefix; 6 non-overlapping temporal regions per trace; cold-start counters include every request; 8/32/128/512/full-region windows share their parent temporal fold. No storage timings, serving scheduler or physical hit telemetry.',source_manifest=json.loads((TRACE/'manifest.json').read_text()),observations=points))

IO_FAMILIES={'affine':2,'sqrt-affine':3,'two-slope':3}
def design(family,s):
    x=np.array(s,dtype=float)/1e6
    if family=='affine':return np.array([np.ones_like(x),x]).T
    if family=='sqrt-affine':return np.array([np.ones_like(x),x,np.sqrt(x)]).T
    return np.array([np.ones_like(x),x,np.maximum(0,x-.25)]).T

def io_fit(rows,family):
    X=design(family,[r['bytes'] for r in rows]);y=np.array([r['latency_ms'] for r in rows]);w=1/np.maximum(y,1e-9)
    return nnls(X*w[:,None],y*w)[0].tolist()
def io_predict(rows,family,coef):return np.maximum(1e-9,design(family,[r['bytes'] for r in rows])@coef)
def io_error(rows,p):
    y=np.array([r['latency_ms'] for r in rows]);size=np.array([r['bytes'] for r in rows]);bw=np.array([r['bandwidth_MBps'] for r in rows]);e=np.abs(p/y-1)*100;be=np.abs((size/p/1000)/bw-1)*100
    return dict(latency_mape_pct=float(e.mean()),latency_p95_pct=float(np.percentile(e,95)),bandwidth_mape_pct=float(be.mean()),bandwidth_p95_pct=float(np.percentile(be,95)),joint_mape_pct=float((e.mean()+be.mean())/2))
def io_cv(rows,family):
    predictions=np.zeros(len(rows))
    for size in sorted({r['bytes'] for r in rows}):
        train=[r for r in rows if r['bytes']!=size];indices=[i for i,r in enumerate(rows) if r['bytes']==size]
        predictions[indices]=io_predict([rows[i] for i in indices],family,io_fit(train,family))
    return predictions

def hit_predict(rows,family,coef):
    x=np.array([r['capacity_blocks']/r['working_set_blocks'] for r in rows]);a,s=coef[:2];bound=np.array([r.get('max_hit_fraction',1) for r in rows])
    if family=='capped-linear':return bound*a*np.minimum(1,x/s)
    p=coef[2] if family=='weibull' else 1
    return bound*a*(-np.expm1(-np.minimum(700,(x/s)**p)))
def hit_fit(rows,family):
    y=np.array([r['hit_fraction'] for r in rows]);start=[.8,.1]+([1.] if family=='weibull' else [])
    result=least_squares(lambda coef:hit_predict(rows,family,coef)-y,start,bounds=([0,1e-6]+([.1] if family=='weibull' else []),[1,100]+([8] if family=='weibull' else [])),max_nfev=2000)
    return result.x.tolist()
def hit_error(rows,p):
    e=np.abs(p-np.array([r['hit_fraction'] for r in rows]))*100
    return dict(mae_pp=float(e.mean()),rmse_pp=float(np.sqrt((e**2).mean())),p95_pp=float(np.percentile(e,95)))
def hit_cv(rows,family):
    p=np.zeros(len(rows))
    for window in sorted({r['window'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['window']==window];train=[r for r in rows if r['window']!=window]
        p[indices]=hit_predict([rows[i] for i in indices],family,hit_fit(train,family))
    return p

def train():
    io_data=json.loads((OUT/'io-observations.json').read_text());hit_data=json.loads((OUT/'hit-observations.json').read_text())
    artifact=dict(schema_version='servescope.storage-fit.v1',io={},hit={},sources={'io_sha256':digest(OUT/'io-observations.json'),'hit_sha256':digest(OUT/'hit-observations.json')},scope='Local warm buffered macOS storage path, QD1 fsync(always) writes; hit functions approximate policy replay, not observed production cache hits.',training_script_sha256=digest(Path(__file__)),runtime_sha256=digest(ROOT/'backend/storage.py'))
    evaluations={'io':{},'hit':{}};predictions={'io':{},'hit':{}}
    # Nested grouped selection gives every size/window a prediction whose target
    # group was excluded from both model selection and coefficient fitting.
    for op in ['read','write']:
        rows=[r for r in io_data['observations'] if r['operation']==op];candidates={}
        for f in IO_FAMILIES:
            p=io_cv(rows,f);candidates[f]=io_error(rows,p)
        best=min(x['joint_mape_pct'] for x in candidates.values());chosen=min([f for f in IO_FAMILIES if candidates[f]['joint_mape_pct']<=best+1],key=lambda f:(IO_FAMILIES[f],candidates[f]['joint_mape_pct']))
        nested=np.zeros(len(rows));selections={};outer_models={}
        for group in sorted({r['bytes'] for r in rows}):
            train=[r for r in rows if r['bytes']!=group];scores={f:io_error(train,io_cv(train,f))['joint_mape_pct'] for f in IO_FAMILIES};low=min(scores.values());f=min([f for f in scores if scores[f]<=low+1],key=lambda f:(IO_FAMILIES[f],scores[f]));indices=[i for i,r in enumerate(rows) if r['bytes']==group];coef=io_fit(train,f);nested[indices]=io_predict([rows[i] for i in indices],f,coef);selections[str(group)]=f;outer_models[str(group)]=dict(family=f,coefficients=coef,training_sizes=sorted({r['bytes'] for r in train}))
        artifact['io'][op]=dict(family=chosen,coefficients=io_fit(rows,chosen),min_bytes=min(r['bytes'] for r in rows),max_bytes=max(r['bytes'] for r in rows),observations=len(rows),validation=io_error(rows,nested))
        evaluations['io'][op]=dict(candidates=candidates,selected=chosen,nested=io_error(rows,nested),outer_selections=selections,outer_models=outer_models);predictions['io'][op]=[dict(**r,predicted_ms=float(p)) for r,p in zip(rows,nested)]
    for workload in ['conversation','toolagent','synthetic']:
        rows=[r for r in hit_data['observations'] if r['workload']==workload];families={'capped-linear':2,'exponential':2,'weibull':3};candidates={f:hit_error(rows,hit_cv(rows,f)) for f in families};best=min(x['mae_pp'] for x in candidates.values());chosen=min([f for f in families if candidates[f]['mae_pp']<=best+.5],key=lambda f:(families[f],candidates[f]['mae_pp']))
        nested=np.zeros(len(rows));selections={};outer_models={}
        for group in range(6):
            train=[r for r in rows if r['window']!=group];scores={f:hit_error(train,hit_cv(train,f))['mae_pp'] for f in families};low=min(scores.values());f=min([f for f in families if scores[f]<=low+.5],key=lambda f:(families[f],scores[f]));indices=[i for i,r in enumerate(rows) if r['window']==group];coef=hit_fit(train,f);nested[indices]=hit_predict([rows[i] for i in indices],f,coef);selections[str(group)]=f;outer_models[str(group)]=dict(family=f,coefficients=coef,training_regions=sorted({r['window'] for r in train}))
        artifact['hit'][workload]=dict(family=chosen,coefficients=hit_fit(rows,chosen),observations=len(rows),validation=hit_error(rows,nested),min_requests=min(r['requests'] for r in rows),max_requests=max(r['requests'] for r in rows))
        evaluations['hit'][workload]=dict(candidates=candidates,selected=chosen,nested=hit_error(rows,nested),outer_selections=selections,outer_models=outer_models);predictions['hit'][workload]=[dict(**r,predicted_hit=float(p)) for r,p in zip(rows,nested)]
    artifact['payload_sha256']=hashlib.sha256(json.dumps(artifact,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dump(ROOT/'backend/storage-fit.json',artifact);dump(OUT/'validation.json',evaluations);dump(OUT/'held-out-predictions.json',predictions)
    print(json.dumps(evaluations,indent=2),flush=True)

def figures():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=json.loads((OUT/'held-out-predictions.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for j,op in enumerate(['read','write']):
        rows=p['io'][op];sizes=sorted({r['bytes'] for r in rows})
        for ax,metric,unit in [(axes[0,j],'latency','ms'),(axes[1,j],'bandwidth','MB/s')]:
            observed=[r['latency_ms'] if metric=='latency' else r['bandwidth_MBps'] for r in rows];predicted=[r['predicted_ms'] if metric=='latency' else r['bytes']/r['predicted_ms']/1000 for r in rows]
            ax.scatter([r['bytes']/1024 for r in rows],observed,s=7,alpha=.25,label='Measured (local path)',color='#527ab2')
            ax.plot([s/1024 for s in sizes],[statistics.mean(v for r,v in zip(rows,predicted) if r['bytes']==s) for s in sizes],marker='o',color='#1a8a6c',label='Nested held-out prediction');ax.set_xscale('log',base=2);
            if metric=='latency':ax.set_yscale('log')
            ax.set_xlabel('I/O size (KiB)');ax.set_ylabel(unit);ax.set_title(op.title()+' '+metric);ax.legend(fontsize=8)
    fig.suptitle('Local Mooncake I/O path · QD1 · warm buffered reads · fsync writes\nBandwidth is bytes/time, derived consistently from the same fitted latency function',fontsize=12);fig.tight_layout();fig.savefig(OUT/'io-fit.png',dpi=180);fig.savefig(OUT/'io-fit.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for j,(workload,rows) in enumerate(p['hit'].items()):
        longest=max(r['requests'] for r in rows)
        for i,length in enumerate([128,longest]):
            ax=axes[i,j]
            for w in range(6):
                part=[r for r in rows if r['window']==w and r['requests']==(128 if i==0 else max(t['requests'] for t in rows if t['window']==w))]
                ax.plot([r['capacity_blocks']/r['working_set_blocks'] for r in part],[100*r['hit_fraction'] for r in part],color='#527ab2',alpha=.35)
                ax.plot([r['capacity_blocks']/r['working_set_blocks'] for r in part],[100*r['predicted_hit'] for r in part],color='#1a8a6c',alpha=.65,linestyle='--')
            ax.set_xscale('symlog',linthresh=.01);ax.set_ylim(0,100);ax.set_xlabel('Capacity / unique blocks in window');ax.set_ylabel('Contiguous-prefix block hit (%)');ax.set_title(workload+(' · 128 requests' if i==0 else ' · full temporal region'))
    fig.suptitle('Cold-start LRU replay references (blue) vs nested held-out hit functions (green dashed)\nSelected window lengths illustrated; all 1,260 curve observations contribute to reported errors',fontsize=11);fig.tight_layout();fig.savefig(OUT/'hit-fit.png',dpi=180);fig.savefig(OUT/'hit-fit.svg');plt.close(fig)
    # Show every candidate, including abandoned ones, on its own diagram.
    data=json.loads((OUT/'io-observations.json').read_text())['observations']
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,op in enumerate(['read','write']):
        rows=[r for r in data if r['operation']==op];sizes=sorted({r['bytes'] for r in rows})
        for j,f in enumerate(IO_FAMILIES):
            predicted=io_cv(rows,f);score=io_error(rows,predicted);ax=axes[i,j]
            ax.scatter([r['bytes']/1024 for r in rows],[r['latency_ms'] for r in rows],s=5,alpha=.25,color='#527ab2')
            ax.plot([n/1024 for n in sizes],[statistics.mean(t for r,t in zip(rows,predicted) if r['bytes']==n) for n in sizes],color='#1a8a6c',marker='o');ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('KiB');ax.set_ylabel('ms');ax.set_title(f'{op} · {f}\nLatency MAPE {score["latency_mape_pct"]:.2f}% / BW {score["bandwidth_mape_pct"]:.2f}%')
    fig.suptitle('I/O function-space search · fixed-family grouped CV, not nested selection scores',fontsize=12);fig.tight_layout();fig.savefig(OUT/'io-candidates.png',dpi=180);fig.savefig(OUT/'io-candidates.svg');plt.close(fig)
    data=json.loads((OUT/'hit-observations.json').read_text())['observations'];fig,axes=plt.subplots(3,3,figsize=(13,10))
    for i,workload in enumerate(['conversation','toolagent','synthetic']):
        rows=[r for r in data if r['workload']==workload]
        for j,f in enumerate(['capped-linear','exponential','weibull']):
            predicted=hit_cv(rows,f);score=hit_error(rows,predicted);ax=axes[i,j]
            for w in range(6):
                length=max(r['requests'] for r in rows if r['window']==w);indices=[n for n,r in enumerate(rows) if r['window']==w and r['requests']==length];part=[rows[n] for n in indices]
                ax.plot([r['capacity_blocks']/r['working_set_blocks'] for r in part],[r['hit_fraction']*100 for r in part],color='#527ab2',alpha=.3)
                ax.plot([r['capacity_blocks']/r['working_set_blocks'] for r in part],[predicted[n]*100 for n in indices],color='#1a8a6c',alpha=.6,linestyle='--')
            ax.set_xscale('symlog',linthresh=.01);ax.set_ylim(0,100);ax.set_xlabel('Capacity / unique blocks');ax.set_ylabel('Hit %');ax.set_title(f'{workload} · {f}\nHeld-window MAE {score["mae_pp"]:.2f} pp')
    fig.suptitle('Hit function-space search · full-region curves illustrated; all window lengths scored',fontsize=12);fig.tight_layout();fig.savefig(OUT/'hit-candidates.png',dpi=180);fig.savefig(OUT/'hit-candidates.svg');plt.close(fig)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['measure','replay','train','figures','all']);a.add_argument('--storage-dir',help='Existing parent directory on the storage filesystem to measure (temporary files are removed)');args=a.parse_args()
    for name,fn in [('measure',measure),('replay',replay_curves),('train',train),('figures',figures)]:
        if args.action in [name,'all']:
            if name=='measure':fn(args.storage_dir)
            else:fn()
