"""Render all-point residual figures, operator breakdowns and an auditable accuracy report."""
import collections,csv,gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from integration.fit_study.data import ROOT,hashfile
from .train import OUT

NAMES={'act_fn':'Activation / SwiGLU','attention:decode':'Decode attention','attention:prefill':'Prefill attention',
 'down_proj':'MLP down projection','embedding':'Embedding','final_layernorm':'Final normalization',
 'gate_up_proj':'MLP gate + up projection','layernorm':'Layer normalization','lm_head':'Language-model output head',
 'o_proj':'Attention output projection','qk_norm':'QK normalization','qkv_proj':'QKV projection','rotary_emb':'Rotary embedding','sampler':'Sampling'}
COLORS={'kaist':'#b55c20','vidur':'#176a91'}
LABELS={'kaist':'KAIST · BF16','vidur':'Vidur · FP16'}


def read_predictions(path):
    data={key:[] for key in ['source','gpu','op','measured_ns','predicted_ns']}
    with gzip.open(path,'rt') as stream:
        for line in stream:
            r=json.loads(line)
            for key in data:data[key].append(r[key])
    return {key:np.array(value) for key,value in data.items()}


def scores(data):
    y=data['measured_ns'];p=data['predicted_ns'];e=np.abs(p/y-1)*100
    return dict(count=len(y),mape_pct=float(np.mean(e)),median_ape_pct=float(np.median(e)),p95_ape_pct=float(np.percentile(e,95)),within_10_pct=float(np.mean(e<=10)*100),within_20_pct=float(np.mean(e<=20)*100))


def save(fig,name):
    for ext in ['png','svg']:
        p=OUT/(name+'.'+ext);fig.savefig(p,dpi=170,facecolor='white',metadata={'Date':None} if ext=='svg' else {})
        if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)


def residual_figure(data,name,title,subtitle):
    s=scores(data);fig,axes=plt.subplots(1,3,figsize=(17,6.5),gridspec_kw={'width_ratios':[1,1.15,1]})
    y=data['measured_ns']/1000;p=data['predicted_ns']/1000;error=(p/y-1)*100;abs_error=np.abs(error)
    lo=min(np.min(y),np.min(p))*.8;hi=max(np.max(y),np.max(p))*1.2
    axes[0].fill_between([lo,hi],[lo*.8,hi*.8],[lo*1.2,hi*1.2],color='#aabccb',alpha=.2,label='±20% band')
    axes[0].plot([lo,hi],[lo,hi],color='#3c5265',lw=1)
    axes[1].axhspan(-20,20,color='#aabccb',alpha=.2);axes[1].axhline(0,color='#3c5265',lw=1)
    for source in ['kaist','vidur']:
        mask=data['source']==source
        if not np.any(mask):continue
        color=COLORS[source];label=LABELS[source]
        axes[0].scatter(y[mask],p[mask],s=3,alpha=.15,color=color,rasterized=True,label=label)
        axes[1].scatter(y[mask],error[mask],s=3,alpha=.15,color=color,rasterized=True)
        sorted_error=np.sort(abs_error[mask]);axes[2].plot(sorted_error,np.arange(1,len(sorted_error)+1)/len(sorted_error)*100,color=color,lw=1.8,label=label)
    axes[0].set(xscale='log',yscale='log',xlabel='Profiled operator time (µs)',ylabel='Cross-validated prediction (µs)',title='Predicted versus profiled')
    axes[0].legend(fontsize=8,loc='upper left',markerscale=3)
    axes[1].set(xscale='log',yscale='symlog',xlabel='Profiled operator time (µs)',ylabel='Signed relative residual (%)',title='Residuals · zero is exact')
    axes[1].set_yscale('symlog',linthresh=20,linscale=.8)
    # All absolute errors enter the ECDF; no clipping of the tail.
    axes[2].set(xscale='symlog',xlabel='Absolute relative error (%)',ylabel='Observations at or below error (%)',title='Error distribution',ylim=(0,101))
    axes[2].set_xscale('symlog',linthresh=1)
    axes[2].axhline(95,color='#647486',lw=.8,ls='--');axes[2].axvline(20,color='#647486',lw=.8,ls=':');axes[2].legend(fontsize=8,loc='lower right')
    for ax in axes:ax.grid(alpha=.16)
    fig.suptitle(title,fontsize=19,fontweight='bold',color='#183247',y=.97)
    fig.text(.5,.885,subtitle,ha='center',fontsize=11,color='#506477')
    fig.text(.5,.085,f"N = {s['count']:,}    |    Mean absolute error {s['mape_pct']:.2f}%    |    Median {s['median_ape_pct']:.2f}%    |    P95 {s['p95_ape_pct']:.2f}%    |    Within ±20%: {s['within_20_pct']:.1f}%",ha='center',fontsize=12,color='#183247')
    fig.text(.5,.025,'Every eligible observation is plotted. Relative error = (prediction − profile) / profile. GPU timing only; serving throughput and energy require separate validation.',ha='center',fontsize=9,color='#506477')
    fig.subplots_adjust(top=.78,bottom=.23,left=.065,right=.985,wspace=.32)
    save(fig,name)
    return s


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def excluded_profiles():
    used={('RTXPRO6000','meta-llama/Llama-3.1-8B','bf16',1),('RTXPRO6000','meta-llama/Llama-3.1-8B','bf16',2),('RTX4090','meta-llama/Llama-3.1-8B','bf16',1),('RTXPRO6000','Qwen/Qwen3-32B','bf16',1),('RTXPRO6000','Qwen/Qwen3-32B','bf16',2)}
    base=ROOT/'.vendor/LLMServingSim/profiler/perf';excluded=[]
    if not base.exists():
        inventory=OUT/'coverage-exclusions.json'
        if not inventory.exists():raise FileNotFoundError('Excluded-profile inventory requires the pinned upstream checkout on first generation.')
        return json.loads(inventory.read_text())
    for path in sorted(base.rglob('dense.csv')):
        relative=path.parent.relative_to(base);parts=relative.parts;key=(parts[0],'/'.join(parts[1:-2]),parts[-2],int(parts[-1][2:]))
        if key in used:continue
        files=[]
        for source in sorted(path.parent.glob('*.csv')):
            with source.open() as stream:count=sum(1 for _ in csv.DictReader(stream))
            files.append(dict(path=str(source.relative_to(ROOT)),rows=count,sha256=hashfile(source)))
        excluded.append(dict(bundle=str(relative),reason='Specialized MoE/MLA/hybrid/MTP operators lack a compatible runtime cost contract; not scored or counted as accurate.',files=files))
    return excluded


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'servescope-production-accuracy-v1'})
    report=json.loads((OUT/'validation.json').read_text());transfer=json.loads((OUT/'gpu-transfer-validation.json').read_text());data=read_predictions(OUT/'cross-validated-predictions.jsonl.gz')
    overall=residual_figure(data,'main-residuals','How accurate is ServeScope’s GPU timing model?','Grouped five-fold outer CV · function/scale selection inside three inner folds · all imported compatible dense profiles')
    op_rows=[];embeds=[]
    for op,metric in report['by_operator'].items():
        mask=data['op']==op;part={key:value[mask] for key,value in data.items()};slug=op.replace(':','-')
        residual_figure(part,'operator-'+slug,NAMES[op]+' · cross-validated GPU timing','Same nested protocol and residual definitions as the main figure · separate historical precision/software cohorts')
        k=report['models']['kaist'].get(op,{}).get('family','—');v=report['models']['vidur'].get(op,{}).get('family','—')
        op_rows.append([NAMES[op],f"{metric['count']:,}",f"{metric['mape_pct']:.2f}%",f"{metric['p95_ape_pct']:.2f}%",k,v])
        embeds.append(f"### {NAMES[op]}\n\n![{NAMES[op]} residuals](operator-{slug}.png)\n\n[Vector figure](operator-{slug}.svg).")
    # Equal visual space per GPU avoids letting Vidur's much larger grids hide KAIST.
    fig,axes=plt.subplots(2,3,figsize=(15,9));axes=axes.ravel()
    for ax,(gpu,metric) in zip(axes,report['by_gpu'].items()):
        mask=data['gpu']==gpu;y=data['measured_ns'][mask]/1000;p=data['predicted_ns'][mask]/1000
        ax.scatter(y,p,s=3,alpha=.16,color=COLORS[data['source'][mask][0]],rasterized=True)
        lo=min(y.min(),p.min());hi=max(y.max(),p.max());ax.plot([lo,hi],[lo,hi],color='#546b7c',lw=1)
        ax.set(xscale='log',yscale='log',xlabel='Profiled time (µs)',ylabel='CV prediction (µs)',title=f"{gpu} · N={metric['count']:,}\nMAPE {metric['mape_pct']:.2f}% · P95 {metric['p95_ape_pct']:.2f}%");ax.grid(alpha=.14)
    axes[-1].axis('off');axes[-1].text(.05,.7,'Known-GPU shape validation\n\nEach query is excluded across GPUs/models/TP.\nGPU families remain represented in training.\n\nNew-GPU transfer is reported separately.',fontsize=12,va='top')
    fig.suptitle('Cross-validated residuals by GPU family',fontsize=19,fontweight='bold',color='#183247');fig.tight_layout(rect=(0,0,1,.95));save(fig,'by-gpu')
    td=read_predictions(OUT/'gpu-transfer-predictions.jsonl.gz');ts=residual_figure(td,'gpu-transfer-residuals','How well does ServeScope transfer to an unseen GPU?','Leave-one-GPU-out validation · target GPU excluded from fitting and inner function/scale selection')
    gpu_rows=[[g,f"{m['count']:,}",f"{m['mape_pct']:.2f}%",f"{m['p95_ape_pct']:.2f}%",f"{m['latency_weighted_ape_pct']:.2f}%"] for g,m in report['by_gpu'].items()]
    transfer_rows=[[name,f"{a['tested_count']:,}",f"{a['metrics']['mape_pct']:.2f}%",f"{a['metrics']['p95_ape_pct']:.2f}%",', '.join(x['operator'] for x in a['unsupported']) or 'None'] for name,a in transfer['audits'].items()]
    cohort_rows=[[s.upper(),f"{m['count']:,}",f"{m['mape_pct']:.2f}%",f"{m['p95_ape_pct']:.2f}%"] for s,m in report['by_source'].items()]
    excluded=excluded_profiles();(OUT/'coverage-exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
    exclusions_table=table(['Locally available excluded bundle','CSV rows available; not comparable operator count'],[[e['bundle'],sum(f['rows'] for f in e['files'])] for e in excluded])
    report_text=f'''# How accurate is LLM-ServeScope?

GPU operator timing evaluation · 19 October 2026 · [ryeii (Ryan Z An)](https://github.com/ryeii)

The production fitting procedure has **{overall['mape_pct']:.2f}% mean absolute relative
operator-time error**, **{overall['median_ape_pct']:.2f}% median error**, and
**{overall['p95_ape_pct']:.2f}% P95 error** across **{overall['count']:,} imported compatible
observations**. **{overall['within_20_pct']:.1f}%** of predictions fall within ±20%.
These are grouped nested cross-validation results, not training residuals.
They describe GPU operator service time under recorded profile conditions.
They are **not an error guarantee for serving throughput, request latency, energy,
new architectures, or a customer workload distribution**.

![Main cross-validated residual graph](main-residuals.png)

[Vector figure](main-residuals.svg). Left: prediction versus profile; center: signed
relative residual; right: cumulative absolute error. Every eligible observation
enters the plots and metrics; no tail clipping, random point subsampling or outlier
removal is used. Positive signed residual means overprediction. Rasterization of
scatter marks reduces vector-file size without discarding observations.

## What data and model are being tested?

- **KAIST / BF16:** all 9,411 imported pure-phase operator observations from
  RTX PRO 6000 and RTX 4090; Llama 3.1 8B and Qwen3 32B; TP1/2 where published.
- **Vidur / FP16:** all 590,040 distinct imported observations from A100, A40 and
  H100; Llama 2 7B/70B and CodeLlama 34B; TP1/2/4/8. This expands the previous
  development-only Vidur study to include its previously reserved GPU/model groups.
- Original Vidur observations: 602,856 before aggregation of repeated exact
  coordinates within the same model/GPU/TP. Published timing means are converted
  from ms to ns. KAIST pure-phase extraction preserves upstream ns and TP-stable rules.
- Historical precision/software cohorts are **fitted separately**, never pooled.
  Production selects Vidur coefficients for FP16, KAIST coefficients for BF16.
  Other precisions extrapolate KAIST functions. Operators missing from Vidur use
  an explicitly labeled KAIST extrapolation. No GPU or model ID selects coefficients.
- Existing F0–F8 families are evaluated; embedding also considers F9's two affine
  functions with a fixed ≤16/>16 token switch. The fixed switch is source-informed,
  not an optimized validation threshold. Runtime shapes, rates and physical L2
  capacity evaluate the selected formula. If an F7 input lacks physical L2 capacity,
  runtime uses the exported additive fit and records the fallback reason.

### Cross-validation and final training

1. Five **outer folds** group canonical executed operator geometry across GPU,
   model and TP. Embedding uses `(token count, hidden width)` to prevent vocabulary
   shards or TP copies of equal work crossing folds.
2. For each outer fold, **three inner folds use only its training observations**
   to choose the family and any transition scale. We choose the fewest parameters
   within one percentage point of minimum inner-CV MAPE, then lower MAPE for ties.
3. The chosen function is refitted on that outer training set and predicts the
   untouched outer fold. Every eligible observation receives one out-of-fold prediction.
4. After evaluation, family/scale selection runs on the full cohort's inner folds,
   and the selected functions are refitted on **all observations in their cohort**
   for deployment. Those full-data coefficients do not generate the CV graph.

This evaluates the **selection and fitting procedure**, since selected families
can differ among folds. It does not imply that the exact final full-data coefficient
vector was tested on data excluded from that final fit. The function space was
informed by earlier exploration of these profiles: this is exploratory nested CV,
not a preregistered or newly collected confirmatory dataset.

The re-fitted additive F0 comparator scores **{report['baseline']['mape_pct']:.2f}%**
mean error on the same folds, versus **{overall['mape_pct']:.2f}%** for selection.
That comparator also gets a separate FP16 fit; it is not the former KAIST-only
production model evaluated indiscriminately on another profiler.

## Accuracy by corpus and GPU

{table(['Corpus','Observations','MAPE','P95 absolute relative error'],cohort_rows)}

{table(['GPU family','Observations','MAPE','P95','Latency-weighted error'],gpu_rows)}

![Cross-validated predictions by GPU](by-gpu.png)

The pooled score is dominated by Vidur's denser attention grids. Equal-operator
macro MAPE is **{report['overall']['macro_operator_mape_pct']:.2f}%**. Per-GPU and
per-operator tables must accompany the pooled number. Latency-weighted error is
sum of absolute ns errors divided by sum of profiled ns, not serving-workload weighting.

## Unseen-GPU accuracy is a different test

![Leave-one-GPU-out residual graph](gpu-transfer-residuals.png)

{table(['Excluded GPU/cohort','Scored observations','MAPE','P95','Unsupported groups'],transfer_rows)}

The target GPU is absent from coefficient fitting **and** function/scale selection.
QK normalization cannot be learned when RTX PRO 6000 is withheld: RTX 4090 only
contains Llama, which lacks this operation. Those 304 targets are explicitly
unsupported rather than scored as a successful prediction. Thus this transfer
figure has **{ts['count']:,}** points, versus 599,451 in shape CV.
KAIST also confounds hardware with software version. This audit is not evidence
that peak specifications alone predict new GPU architectures within a fixed error.

## Main residual graphs for every subcomponent

{table(['Subcomponent','Observations','MAPE','P95','Deployed KAIST family','Deployed Vidur family'],op_rows)}

A dash means no measured operator group in that corpus. The deployed runtime
retains a documented extrapolation/fallback where a scenario requests such a group.
Families: F0/F1 additive; F2 max roofline; F3 smooth overlap; F4 tile-wave;
F5/F6 bounded shape transition; F7 cache-capacity transition; F8 corrected embedding
traffic; F9 independently fitted embedding dispatch branches.

{chr(10).join(embeds)}

## Serving integration recheck (separate evidence)

The final selected KAIST functions were also compared with nine existing native
KAIST/ASTRA profile-backed serving scenarios: Llama 3.1 8B BF16, TP1/2 on RTX PRO
6000 and TP1 on RTX 4090; batches 1/4 and inputs 128/512. Mean differences are:

| Metric | Previous deployed F0 | Selected production model |
| --- | --- | --- |
| Output throughput | 4.10% | 4.83% |
| TTFT | 14.83% | 11.60% |
| TPOT | 4.65% | 3.87% |

Throughput agreement worsens slightly while TTFT/TPOT improve. No coefficients
were adjusted to improve these nine serving targets. This is a simulator-to-simulator
integration check on known profiled identities, not new end-to-end GPU measurements
or an unseen-GPU test. The native artifact hashes were reverified. Read the
[recheck metrics and source hashes](serving-recheck-comparison.json),
[scenario CSV](serving-recheck-comparison.csv), and
[operator CSV](serving-recheck-operators.csv). Differences also include scheduling,
communications and endpoint semantics; these scores cannot be inferred from the
main operator residual plot.

## Coverage exclusions and limits

“All data” in the main graph means **all imported compatible dense-decoder
observations**, not every CSV row present in an upstream repository. The following
locally available specialized bundles remain outside the runtime cost contract:

{exclusions_table}

These MoE/MLA/hybrid/MTP tables have incompatible operator structures and were
neither silently mapped onto dense kernels nor counted as accurate. Exact CSV
paths, row counts and hashes are in [coverage-exclusions.json](coverage-exclusions.json).
Mixed-phase and heterogeneous-history KAIST attention also remain outside the
pure-phase imported corpus. Published hardware specifications and model configs
are inputs, not additional timing measurements.

No new GPU inference was run. No profiled energy, power-counter, network-collective,
cache-miss or memory-controller measurements were used to calibrate those outputs.
SimPy serving queues and analytic energy/network estimates require separate checks.
Historical environment metadata is incomplete, especially for Vidur. The FP16
profile-reference model presets preserve Vidur's padded 32768 vocabulary; these
are labeled as profile geometry, not a claim of a new checkpoint measurement.

## Reproduce and inspect

```sh
# Python 3.10 and the pinned study dependencies; CPU only.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
  .venv-sim/bin/python -u -B -m integration.production_fit.train
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
  .venv-sim/bin/python -u -B -m integration.production_fit.transfer
.venv-sim/bin/python -B -m integration.production_fit.export
.venv-sim/bin/python -B -m integration.production_fit.report
.venv-sim/bin/python -B -m unittest integration.production_fit.test_production_fit -v
```

[All nested validation models and selections](validation.json),
[every out-of-fold prediction](cross-validated-predictions.jsonl.gz),
[GPU-transfer models and selections](gpu-transfer-validation.json),
[GPU-transfer predictions](gpu-transfer-predictions.jsonl.gz),
[production coefficient artifact](../../backend/gpu-fit.json), and
[implementation](../../integration/production_fit). Caches are ignored and checked
against input-array and source hashes. Graphs can be recreated from archived
predictions without refitting. Runtime uses standard-library Python only.

Sources: [KAIST pinned profiles](https://github.com/casys-kaist/LLMServingSim/tree/b09c7aaab4b072d73136641841cc2688616c49eb/profiler/perf),
[Vidur pinned profiles](https://github.com/microsoft/vidur/tree/abae7f63aa857300f5cdc6f5e0d27860cd24721b/data/profiling/compute),
[Vidur archive manifest](../../third_party/vidur-profiles/manifest.json).
Profile authors receive credit under retained MIT licenses. ServeScope's function
selection, implementation, UI and evaluation: [ryeii (Ryan Z An)](https://github.com/ryeii).
'''
    (OUT/'README.md').write_text(report_text)
    (OUT/'figure-summary.json').write_text(json.dumps(dict(overall=overall,transfer=ts,figure_count=17,point_subsampling=False),indent=2)+'\n')
    print('Rendered main graph, 14 operator graphs, GPU panels, and transfer graph.',flush=True)
if __name__=='__main__':main()
