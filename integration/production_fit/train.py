"""Nested grouped CV on every imported compatible profile; no timing lookup at runtime.

Run with one BLAS thread: .venv-sim/bin/python -u -B -m integration.production_fit.train
Caches contain training-only inner selection and are invalidated by source hashes.
"""
from pathlib import Path
import gzip,hashlib,json,platform,time
import numpy as np
import scipy
from integration.fit_study.data import ROOT,kaist_rows,vidur_rows,hashfile
from integration.fit_study.functions import FAMILIES,SCALES,fit_fixed,predict,metrics

OUT=ROOT/'docs/accuracy-2026-10-19'
CACHE=ROOT/'.cache/production-fit'
SEED='servescope-production-grouped-v1'

def split_key(r):
    # Embedding does not shard hidden rows. Keep equal work across vocab/TP/IDs together.
    return json.dumps(dict(op='embedding',n=r['n'],hidden=r['geometry'][0]),sort_keys=True) if r['op']=='embedding' else r['key']

def fold_ids(rows):
    hashes=[hashlib.sha256((SEED+split_key(r)).encode()).hexdigest() for r in rows]
    return np.array([int(h[:8],16)%5 for h in hashes]),np.array([int(h[8:16],16)%3 for h in hashes])

def configs(op):
    result=[]
    for family in FAMILIES:
        if family=='F8' and op!='embedding':continue
        for scale in SCALES if family in ('F5','F6','F7') else (1.,):result.append((family,scale))
    if op=='embedding':result.append(('F9',1.))
    return result

def affine(x,y):
    from itertools import combinations
    b=np.column_stack((np.ones(len(x)),x[:,6]))/y[:,None]
    normalizers=np.maximum(np.linalg.norm(b,axis=0),1e-30);a=b/normalizers
    best=(float('inf'),np.zeros(2))
    for size in (1,2):
        for columns in combinations(range(2),size):
            z=np.linalg.lstsq(a[:,columns],np.ones(len(y)),rcond=None)[0]
            if np.any(z < -1e-10):continue
            coef=np.zeros(2);coef[list(columns)]=np.maximum(0,z)/normalizers[list(columns)]
            loss=float(np.mean((b@coef-1)**2))
            if loss<best[0]:best=(loss,coef)
    return best[1].tolist()

def fit_config(config,x,y,n):
    family,scale=config
    if family!='F9':return fit_fixed(family,x,y,scale)
    small=n<=16
    if min(np.sum(small),np.sum(~small))<2:raise ValueError('Insufficient embedding dispatch coverage')
    return dict(family='F9',threshold_tokens=16,threshold_fitted=False,
                branches={name:affine(x[mask],y[mask]) for name,mask in [('small_index',small),('gather',~small)]})

def predict_config(model,x,n):
    if model['family']!='F9':return predict(model,x)
    small=n<=model['threshold_tokens'];p=np.empty(len(x))
    for name,mask in [('small_index',small),('gather',~small)]:
        a,b=model['branches'][name];p[mask]=a+b*x[mask,6]
    return np.maximum(1,p)

def parameter_count(family):return 4 if family=='F9' else FAMILIES[family]['parameters']

def choose(op,x,y,n,inner):
    candidates=[]
    for config in configs(op):
        losses=[];valid=True
        for f in range(3):
            mask=inner!=f
            if not np.any(~mask) or not np.any(mask):continue
            model=fit_config(config,x[mask],y[mask],n[mask])
            if model.get('optimizer_success') is False:valid=False;break
            losses.extend(np.abs(predict_config(model,x[~mask],n[~mask])/y[~mask]-1).tolist())
        if valid and losses:candidates.append(dict(family=config[0],scale=config[1],inner_mape_pct=float(np.mean(losses)*100),parameters=parameter_count(config[0])))
    best=min(c['inner_mape_pct'] for c in candidates)
    eligible=[c for c in candidates if c['inner_mape_pct']<=best+1.]
    chosen=min(eligible,key=lambda c:(c['parameters'],c['inner_mape_pct'],c['family'],c['scale']))
    model=fit_config((chosen['family'],chosen['scale']),x,y,n)
    if model.get('optimizer_success') is False:raise RuntimeError('Final selected optimizer did not converge')
    return dict(model=model,choice=chosen,candidates=candidates,selection_rule='Fewest parameters within 1 percentage point of minimum inner-CV MAPE; ties use lower MAPE.')

def cached_fit(cohort,op,scope,x,y,n,inner,signature):
    path=CACHE/cohort/op.replace(':','_')/(scope+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(x.tobytes()+y.tobytes()+n.tobytes()+inner.tobytes()+json.dumps(signature,sort_keys=True).encode()).hexdigest()
    if path.exists():
        r=json.loads(path.read_text())
        if r['signature']==digest:return r['result']
    start=time.time();r=choose(op,x,y,n,inner)
    path.write_text(json.dumps(dict(signature=digest,result=r),indent=2)+'\n')
    print(cohort,op,scope,r['model']['family'],f"inner={r['choice']['inner_mape_pct']:.2f}% {time.time()-start:.1f}s",flush=True)
    return r

def load_rows():
    k=kaist_rows();v,info=vidur_rows()
    return k+v,info

def main():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    signature=dict(train_sha256=hashfile(Path(__file__)),functions_sha256=hashfile(ROOT/'integration/fit_study/functions.py'),
                   adapter_sha256=hashfile(ROOT/'integration/fit_study/data.py'),kaist_sha256=hashfile(ROOT/'third_party/kaist-dense-profile-grids.json.gz'),
                   vidur_manifest_sha256=hashfile(ROOT/'third_party/vidur-profiles/manifest.json'),seed=SEED)
    rows,info=load_rows();print('Loaded',len(rows),'operator observations',flush=True)
    oof=np.empty(len(rows));baseline=np.empty(len(rows));folds,inner=fold_ids(rows);exports={};results={}
    index=np.arange(len(rows))
    for source in ['kaist','vidur']:
        exports[source]={};results[source]={}
        for op in sorted({r['op'] for r in rows if r['source']==source}):
            ix=np.array([i for i,r in enumerate(rows) if r['source']==source and r['op']==op])
            x=np.array([rows[i]['x'] for i in ix]);y=np.array([rows[i]['ns'] for i in ix]);n=np.array([rows[i]['n'] for i in ix]);part_folds=folds[ix];part_inner=inner[ix];audits=[]
            for f in range(5):
                mask=part_folds!=f
                selected=cached_fit(source,op,'outer-'+str(f),x[mask],y[mask],n[mask],part_inner[mask],signature)
                p=predict_config(selected['model'],x[~mask],n[~mask]);oof[ix[~mask]]=p
                base=fit_config(('F0',1.),x[mask],y[mask],n[mask]);baseline[ix[~mask]]=predict_config(base,x[~mask],n[~mask])
                audits.append(dict(fold=f,training_count=int(np.sum(mask)),tested_count=int(np.sum(~mask)),selection=selected,metrics=metrics(y[~mask],p)))
            full=cached_fit(source,op,'full',x,y,n,part_inner,signature);exports[source][op]=full['model']
            results[source][op]=dict(count=len(ix),full_selection=full,outer_folds=audits,cv=metrics(y,oof[ix]),baseline_cv=metrics(y,baseline[ix]))
    # Columnar archive includes every point; figures may aggregate into bins but never discard targets.
    meta_keys=['source','gpu','model','tp','op','n','geometry','shape','query_tokens']
    archive=OUT/'cross-validated-predictions.jsonl.gz'
    with gzip.GzipFile(filename=str(archive),mode='wb',mtime=0) as out:
        for i,r in enumerate(rows):
            record={key:r[key] for key in meta_keys};record.update(fold=int(folds[i]),inner_fold=int(inner[i]),group_key=split_key(r),features=r['x'],measured_ns=r['ns'],predicted_ns=float(oof[i]),baseline_ns=float(baseline[i]))
            out.write((json.dumps(record,separators=(',',':'))+'\n').encode())
    y=np.array([r['ns'] for r in rows]);source=np.array([r['source'] for r in rows]);ops=np.array([r['op'] for r in rows]);gpus=np.array([r['gpu'] for r in rows])
    report=dict(protocol=dict(outer_folds=5,inner_folds=3,grouping='Canonical executed operator geometry across GPU/model/TP; embedding groups token count and hidden width.',selection='Family and transition scale selected on inner folds only. Outer fold never fits or selects coefficients.',cohort_policy='BF16 KAIST and FP16 Vidur kept separate. No per-GPU/model coefficient selection.',candidates='F0–F8; embedding additionally F9 with fixed 16-token switch.',interpretation='Exploratory nested cross-validation within imported dense profile conditions, not a new-GPU or end-to-end serving error guarantee.'),
      count=len(rows),overall=metrics(y,oof,ops),baseline=metrics(y,baseline,ops),
      by_source={s:metrics(y[source==s],oof[source==s],ops[source==s]) for s in sorted(set(source))},
      by_gpu={g:metrics(y[gpus==g],oof[gpus==g],ops[gpus==g]) for g in sorted(set(gpus))},
      by_operator={op:metrics(y[ops==op],oof[ops==op]) for op in sorted(set(ops))},
      groups=results,models=exports,vidur_adapter_info=info,signature=signature,
      prediction_archive_sha256=hashfile(archive),environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    (OUT/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Completed',report['overall'],flush=True)
if __name__=='__main__':main()
