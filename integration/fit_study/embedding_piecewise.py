"""F9: two nonnegative affine embedding functions; source-defined switch at 16.

Standalone follow-up: preserves all F0–F8 study results and deployed baseline.
Run: .venv-sim/bin/python -B -m integration.fit_study.embedding_piecewise
"""
from pathlib import Path
import csv,hashlib,io,itertools,json,math,platform,statistics
import numpy as np
from .data import ROOT,VIDUR_HARDWARE,kaist_rows,make_row,read_archived,hashfile
from .functions import fit as fit_original,predict as predict_original,metrics

OUT=ROOT/'docs/fit-study-2026-10-19/embedding-piecewise'
THRESHOLD=16

def regime(n):
    if not math.isfinite(n) or n<1 or n!=int(n):
        raise ValueError('Embedding token count must be a positive integer.')
    return 'small_index' if n<=THRESHOLD else 'gather'

def query_fold(row):
    # Group equal branch-work inputs across GPU/model/TP and vocabulary shards.
    key=dict(n=row['n'],hidden=row['geometry'][0])
    return int(hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:8],16)%5

def fit_affine(rows):
    x=np.array([[1.,r['x'][6]] for r in rows]);y=np.array([r['ns'] for r in rows])
    design=x/y[:,None];normalizers=np.maximum(np.linalg.norm(design,axis=0),1e-30)
    scaled=design/normalizers;best=None
    for size in (1,2):
        for subset in itertools.combinations(range(2),size):
            value=np.linalg.lstsq(scaled[:,subset],np.ones(len(rows)),rcond=None)[0]
            if np.any(value < -1e-10):continue
            coef=np.zeros(2);coef[list(subset)]=np.maximum(0,value)/normalizers[list(subset)]
            loss=float(np.mean((design@coef-1)**2))
            if best is None or loss<best[0]:best=(loss,coef)
    coef=best[1]
    return dict(intercept_ns=float(coef[0]),memory_scale=float(coef[1]),training_count=len(rows))

def fit_piecewise(rows):
    branches={name:[r for r in rows if regime(r['n'])==name] for name in ['small_index','gather']}
    if any(len(part)<2 for part in branches.values()):raise ValueError('Both dispatch regimes require at least two training observations.')
    return dict(family='F9',threshold_tokens=THRESHOLD,threshold_fitted=False,
      formula='max(1 ns, L_branch + b_branch * corrected_nominal_memory_ns)',
      parameter_count=4,branches={name:fit_affine(part) for name,part in branches.items()})

def predict_piecewise(model,n,memory_ns):
    if model['threshold_tokens']!=THRESHOLD:raise ValueError('F9 uses the fixed source-defined threshold of 16.')
    if not math.isfinite(memory_ns) or memory_ns<0:raise ValueError('Memory time must be finite and nonnegative.')
    branch=model['branches'][regime(n)]
    return max(1.,branch['intercept_ns']+branch['memory_scale']*memory_ns)

def predictions(model,rows):return np.array([predict_piecewise(model,r['n'],r['x'][6]) for r in rows])

def score(rows,pred):
    return dict(overall=metrics([r['ns'] for r in rows],pred),
      by_regime={name:metrics([r['ns'] for r in rows if regime(r['n'])==name],
                             [p for r,p in zip(rows,pred) if regime(r['n'])==name]) for name in ['small_index','gather']})

def run_fit(rows):
    full=fit_piecewise(rows);oof=np.empty(len(rows));old=np.empty(len(rows));folds=[]
    for fold in range(5):
        train=[r for r in rows if query_fold(r)!=fold];indices=[i for i,r in enumerate(rows) if query_fold(r)==fold];test=[rows[i] for i in indices]
        model=fit_piecewise(train);oof[indices]=predictions(model,test)
        baseline=fit_original('F8',train);old[indices]=predict_original(baseline,np.array([r['x'] for r in test]))
        folds.append(dict(fold=fold,model=model,F8_model=baseline,training_count=len(train),tested_count=len(test)))
    baseline=fit_original('F8',rows)
    return dict(count=len(rows),model=full,training=score(rows,predictions(full,rows)),
      oof=score(rows,oof),F8_training=score(rows,predict_original(baseline,np.array([r['x'] for r in rows]))),
      F8_oof=score(rows,old),folds=folds),oof,old

def audit(train,test):
    full=fit_piecewise(train);baseline=fit_original('F8',train)
    return dict(training_count=len(train),tested_count=len(test),model=full,
      F9=score(test,predictions(full,test)),F8=score(test,predict_original(baseline,np.array([r['x'] for r in test]))))

def vidur_embeddings():
    manifest=json.loads((ROOT/'third_party/vidur-profiles/manifest.json').read_text());rows=[]
    for entry in manifest['files']:
        path=entry['path']
        if not path.endswith('mlp.csv'):continue
        parts=path.split('/');gpu=parts[3];model='/'.join(parts[4:-1])
        for r in csv.DictReader(io.StringIO(read_archived(path))):
            h=int(r['n_embd']);q=int(r['n_head']);tp=int(r['num_tensor_parallel_workers'])
            m=dict(hidden=h,intermediate=int(r['n_expanded_embd']),heads=q,kv_heads=int(r['n_kv_head']),head_dim=h//q,vocab=int(r['vocab_size']),experts=1,top_k=1)
            rows.append(make_row('vidur',gpu,model,tp,m,VIDUR_HARDWARE[gpu],'embedding',int(r['num_tokens']),float(r['time_stats.emb.mean'])*1e6))
    grouped={}
    for r in rows:grouped.setdefault((r['gpu'],r['model'],r['tp'],r['n']),[]).append(r)
    result=[]
    for part in grouped.values():
        r=dict(part[0]);r['ns']=statistics.median(x['ns'] for x in part);r['repeat_count']=len(part);result.append(r)
    return sorted(result,key=lambda r:(r['gpu'],r['model'],r['tp'],r['n']))

def plot_local_and_portable(rows,local,local_result,portable_result,oof):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'servescope-embedding-F9'})
    fig,axes=plt.subplots(1,2,figsize=(16,6.4),gridspec_kw={'width_ratios':[1.3,1]})
    ax=axes[0];n=np.array([r['n'] for r in local]);y=np.array([r['ns']/1000 for r in local]);small=n<=16
    ax.scatter(n[small],y[small],color='#152d43',s=24,label='Profile: ≤16 tokens',zorder=4)
    ax.scatter(n[~small],y[~small],color='#246d65',marker='s',s=16,label='Profile: >16 tokens',zorder=4)
    # Lines are independent. No interpolated bridge across the dispatch jump.
    step=local[0]['x'][6]/local[0]['n'];colors={'small_index':'#bc601a','gather':'#1466a1'}
    for name,grid in [('small_index',np.linspace(1,16,120)),('gather',np.geomspace(17,max(n),200))]:
        c=local_result['model']['branches'][name];p=c['intercept_ns']+c['memory_scale']*step*grid
        ax.plot(grid,p/1000,color=colors[name],lw=2,label='F9: '+('small-index function' if name=='small_index' else 'gather function'))
    old=fit_original('F8',local);oldp=predict_original(old,np.array([r['x'] for r in local]))
    ax.plot(n,oldp/1000,color='#7d8389',lw=1.1,ls=':',label='F8: one smooth function')
    ax.axvline(16,color='#526273',lw=1,ls='--');ax.set(xscale='log',yscale='log',xlabel='Tokens per embedding call',ylabel='Latency (µs)',title='Two fitted functions on the originally plotted configuration')
    ax.grid(alpha=.13);ax.legend(fontsize=8.5,loc='upper left')
    formula=[]
    for name in ['small_index','gather']:
        c=local_result['model']['branches'][name];cv=local_result['oof']['by_regime'][name]['mape_pct']
        formula.append(f"{'n ≤ 16' if name=='small_index' else 'n > 16'}: T = {c['intercept_ns']/1000:.4f} + {c['memory_scale']*step/1000:.6f} n µs  |  shape CV {cv:.2f}%")
    ax.text(.5,-.22,'\n'.join(formula),transform=ax.transAxes,ha='center',va='top',fontsize=10,color='#152d43')
    ax=axes[1];target=np.array([r['ns']/1000 for r in rows]);mask=np.array([r['n']<=16 for r in rows])
    for b,color,label in [(True,'#bc601a','Small-index regime'),(False,'#1466a1','Gather regime')]:
        ax.scatter(target[mask==b],oof[mask==b]/1000,color=color,s=12,alpha=.5,label=label,rasterized=True)
    lo=min(target)*.8;hi=max(target)*1.2;ax.plot([lo,hi],[lo,hi],color='#526273',lw=1);ax.fill_between([lo,hi],[lo*.9,hi*.9],[lo*1.1,hi*1.1],color='#a9bac7',alpha=.15)
    ax.set(xscale='log',yscale='log',xlabel='Profiled latency (µs)',ylabel='Held-out prediction (µs)',title='Shared hardware-normalized functions: all KAIST bundles');ax.grid(alpha=.13);ax.legend(fontsize=9)
    ax.text(.5,-.22,f"Same four coefficients on all GPUs/models/TP\nShape CV: F8 {portable_result['F8_oof']['overall']['mape_pct']:.2f}% → F9 {portable_result['oof']['overall']['mape_pct']:.2f}%",transform=ax.transAxes,ha='center',va='top',fontsize=10,color='#152d43')
    fig.suptitle('Embedding F9: independent functions with a fixed switch after 16 tokens',fontsize=17,fontweight='bold',color='#152d43')
    fig.text(.5,.91,'Left: RTX PRO 6000 · Llama 8B · TP1 local diagnostic. Right: 760 observations; no GPU/model-ID corrections.',ha='center',fontsize=11,color='#526273')
    fig.subplots_adjust(top=.80,bottom=.28,wspace=.28)
    for ext in ['png','svg']:
        path=OUT/('two-function-fit.'+ext);kwargs={'metadata':{'Date':None}} if ext=='svg' else {}
        fig.savefig(path,dpi=180,facecolor='white',**kwargs)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)

def write_report(result):
    k=result['kaist'];local=result['plotted_local'];v=result['vidur'];rows=[]
    for name,r in [('Plotted configuration only',local),('KAIST, all five bundles',k),('Vidur, A100/A40 development',v)]:
        for branch in ['small_index','gather']:
            a=r['training']['by_regime'][branch];b=r['oof']['by_regime'][branch]
            rows.append(f"| {name} | {branch} | {a['count']} | {a['mape_pct']:.2f}% | {b['mape_pct']:.2f}% | {b['p95_ape_pct']:.2f}% |")
    transfers=[]
    for source,data in [('KAIST',result['kaist_audits']),('Vidur',result['vidur_audits'])]:
        for name,a in data.items():
            transfers.append(f"| {source}: {name} | {a['tested_count']} | {a['F8']['overall']['mape_pct']:.2f}% → {a['F9']['overall']['mape_pct']:.2f}% | {a['F8']['overall']['latency_weighted_ape_pct']:.2f}% → {a['F9']['overall']['latency_weighted_ape_pct']:.2f}% |")
    equations=[];step=result['local_memory_ns_per_token']
    for name,condition in [('small_index','n ≤ 16'),('gather','n > 16')]:
        c=local['model']['branches'][name]
        equations.append(f"{condition}: T_us = {c['intercept_ns']/1000:.6f} + {c['memory_scale']*step/1000:.9f} * n")
    text=f'''# F9: two embedding functions with a fixed dispatch switch

Follow-up to the [original nine-function study](../README.md) and
[embedding-dispatch diagnosis](../embedding-dispatch.md).

**The two regimes do fit substantially better separately.** For the originally
plotted configuration, small-index and gather training errors are
**{local['training']['by_regime']['small_index']['mape_pct']:.2f}%** and
**{local['training']['by_regime']['gather']['mape_pct']:.2f}%** respectively.
Five-fold shape-validation errors are **{local['oof']['by_regime']['small_index']['mape_pct']:.2f}%**
and **{local['oof']['by_regime']['gather']['mape_pct']:.2f}%**.

![Two independently fitted embedding functions](two-function-fit.png)

[Full-resolution SVG](two-function-fit.svg).

## Functions and the switch

```
M_ns = n * (2 * dtype_bytes * hidden + 8) / (DRAM_bandwidth_TBps * 1000)
T(n, hardware) = max(1 ns, L_small + b_small * M_ns)  if n ≤ 16
                max(1 ns, L_gather + b_gather * M_ns) if n > 16
```

Both intercepts and both slopes are nonnegative. Four coefficients are fitted
by nonnegative relative least squares. There is no compute slope for embedding,
which has zero FLOPs in this abstraction. The boundary is fixed at 16 from the
observed discontinuity and published software dispatch, not optimized against
validation targets. The functions have independent intercepts: enforcing continuity
would erase the observed latency drop. Integer counts 17–19 are unmeasured in the
KAIST grid and follow the source-defined gather branch, rather than invented data.

For **the plotted configuration only**, the fitted equations reduce to:

```
{chr(10).join(equations)}
```

These local coefficients illustrate how well each line can be fitted with fixed
hardware/tensor width. They are diagnostic fits, not per-GPU runtime lookup entries.
The separately exported portable fit uses one set of four coefficients shared
across every KAIST GPU/model/TP bundle and operates on nominal memory time `M`.
No device or model ID selects numerical coefficients. Increasing bandwidth lowers
the predicted cost within each branch. Latency is allowed to jump at dispatch.

## Fit and validation

| Corpus | Regime | Points | Training MAPE | Shape-CV MAPE | Shape-CV P95 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

Overall KAIST shape-CV error: **F8 {k['F8_oof']['overall']['mape_pct']:.2f}% →
F9 {k['oof']['overall']['mape_pct']:.2f}%**. These are embedding-only scores.
Both are retrained under the same stricter split: equal `(token count, hidden width)`
queries stay in one fold across GPU, model, TP and vocabulary shard. Thus repeated
work cannot appear in both training and validation. F8 scores may differ slightly
from the original study, whose folds also included vocabulary-shard geometry.

F9 was proposed after examining these curves, so this is exploratory follow-up
validation. It is not a pristine confirmatory test of an untouched hypothesis.

| Transfer audit | Test points | F8 mean → F9 mean | F8 weighted → F9 weighted |
| --- | --- | --- | --- |
{chr(10).join(transfers)}

Vidur FP16 coefficients are fitted separately from KAIST BF16. Its H100 and
CodeLlama profiles are excluded from development fitting. These tests assess the
function form within that separate corpus; they do not validate KAIST coefficients
on H100. Full per-regime results, coefficients and input/source hashes are in
[results.json](results.json).

The local curves are cleaner than the portable fit. The switch addresses a real
missing software regime; it does not resolve cross-GPU intercept/rate differences,
unknown cache behavior, engine versions or timing methodology. It is a simple,
source-informed improvement rather than a universal GPU accuracy claim.

## Reproduce and use

```sh
.venv-sim/bin/python -B -m integration.fit_study.embedding_piecewise
.venv-sim/bin/python -B -m unittest integration.fit_study.test_embedding_piecewise -v
```

The working predictor is `predict_piecewise(model, token_count, memory_ns)` in
[embedding_piecewise.py](../../../integration/fit_study/embedding_piecewise.py).
It evaluates exactly one of the two fitted functions using the fixed switch.
The exported models include branch coefficients and five-fold models. The original
F0–F8 results and application baseline are preserved; this follow-up uses F9 for
its reported predictions and figures, without claiming a production deployment.

The switch is specific to the documented eager PyTorch embedding/index-select
path and its indexing conditions. Other software versions, fused implementations,
padding/CUDA-graph policies and quantization may dispatch differently. Source
references and the distinction between source evidence and actual kernel traces
are retained in the [dispatch diagnosis](../embedding-dispatch.md).

Credit: KAIST CASYS Lab / LLMServingSim 2.0 and Microsoft Vidur contributors for
published profiles; [ryeii (Ryan Z An)](https://github.com/ryeii) for ServeScope
and this experimental function implementation.
'''
    (OUT/'README.md').write_text(text)

def main():
    OUT.mkdir(exist_ok=True)
    rows=sorted([r for r in kaist_rows() if r['op']=='embedding'],key=lambda r:(r['gpu'],r['model'],r['tp'],r['n']))
    local=[r for r in rows if r['gpu']=='rtx6000' and r['model']=='llama8' and r['tp']==1]
    k,oof,old=run_fit(rows);l,_,_=run_fit(local)
    v=vidur_embeddings();dev=[r for r in v if r['gpu']!='h100' and not r['model'].startswith('codellama/')]
    vr,_,_=run_fit(dev)
    ka={f'GPU_{g}':audit([r for r in rows if r['gpu']!=g],[r for r in rows if r['gpu']==g]) for g in sorted(set(r['gpu'] for r in rows))}
    ka['model_qwen32']=audit([r for r in rows if r['model']!='qwen32'],[r for r in rows if r['model']=='qwen32'])
    va=dict(H100=audit(dev,[r for r in v if r['gpu']=='h100' and not r['model'].startswith('codellama/')]),
            CodeLlama_seen_GPUs=audit(dev,[r for r in v if r['gpu']!='h100' and r['model'].startswith('codellama/')]),
            H100_CodeLlama=audit(dev,[r for r in v if r['gpu']=='h100' and r['model'].startswith('codellama/')]))
    records=[dict(gpu=r['gpu'],model=r['model'],tp=r['tp'],n=r['n'],hidden=r['geometry'][0],fold=query_fold(r),regime=regime(r['n']),measured_ns=r['ns'],predicted_ns=float(p),F8_predicted_ns=float(b)) for r,p,b in zip(rows,oof,old)]
    result=dict(family='F9',status='tested_research_predictor',threshold_tokens=16,threshold_fitted=False,
      kaist=k,plotted_local=l,local_memory_ns_per_token=local[0]['x'][6]/local[0]['n'],vidur=vr,
      kaist_audits=ka,vidur_audits=va,kaist_oof_predictions=records,
      provenance=dict(implementation_sha256=hashfile(Path(__file__)),adapter_sha256=hashfile(Path(__file__).with_name('data.py')),
        kaist_grid_sha256=hashfile(ROOT/'third_party/kaist-dense-profile-grids.json.gz'),
        vidur_manifest_sha256=hashfile(ROOT/'third_party/vidur-profiles/manifest.json'),
        python=platform.python_version(),numpy=np.__version__))
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot_local_and_portable(rows,local,l,k,oof);write_report(result)
    print(json.dumps(dict(local=l['oof'],kaist=k['oof'],F8_kaist=k['F8_oof'],vidur=vr['oof']),indent=2))
if __name__=='__main__':main()
