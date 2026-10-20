"""Render all-point residual figures, operator breakdowns and an auditable accuracy report."""
import collections,csv,gzip,json,re
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


def prediction_marks(ax,y,p,source):
    """All observations contribute; dense clouds become log-count hexagonal bins."""
    lo=min(np.min(y),np.min(p))*.8;hi=max(np.max(y),np.max(p))*1.2
    ax.fill_between([lo,hi],[lo*.8,hi*.8],[lo*1.2,hi*1.2],color='#aabccb',alpha=.18,zorder=0)
    ax.plot([lo,hi],[lo,hi],color='#526778',lw=1,ls='--',zorder=1)
    if len(y)>5000:
        artist=ax.hexbin(y,p,xscale='log',yscale='log',gridsize=65,mincnt=1,
                         bins='log',cmap='Oranges' if source=='kaist' else 'Blues',
                         linewidths=0,rasterized=True,zorder=2)
        colorbar=ax.figure.colorbar(artist,ax=ax,pad=.025,fraction=.045)
        colorbar.set_label('Observations per bin',fontsize=9)
    else:
        ax.scatter(y,p,s=9,alpha=.42,color=COLORS[source],edgecolors='none',rasterized=True,zorder=2)
    ax.set(xscale='log',yscale='log')


def residual_figure(data,name,title,subtitle):
    sources=set(data['source'])
    if len(sources)!=1:raise ValueError('Each figure must contain exactly one profiling corpus.')
    source=next(iter(sources));s=scores(data)
    fig,axes=plt.subplots(1,2,figsize=(12.8,6.1),gridspec_kw={'width_ratios':[1.12,1]})
    y=data['measured_ns']/1000;p=data['predicted_ns']/1000;abs_error=np.abs(p/y-1)*100
    prediction_marks(axes[0],y,p,source)
    axes[0].set(xlabel='Profiled operator time (µs)',ylabel='Cross-validated prediction (µs)',title='Predicted versus profiled · ±20% band')
    sorted_error=np.sort(abs_error)
    axes[1].plot(sorted_error,np.arange(1,len(sorted_error)+1)/len(sorted_error)*100,color=COLORS[source],lw=2)
    axes[1].set(xlabel='Absolute relative error (%)',ylabel='Observations at or below error (%)',title='Error distribution',ylim=(0,101))
    axes[1].set_xscale('symlog',linthresh=1)
    axes[1].axhline(95,color='#647486',lw=.8,ls='--');axes[1].axvline(20,color='#647486',lw=.8,ls=':')
    for ax in axes:ax.grid(alpha=.14)
    fig.suptitle(title,fontsize=18,fontweight='bold',color='#183247',y=.97)
    fig.text(.5,.885,subtitle,ha='center',fontsize=10,color='#506477')
    fig.text(.5,.105,f"N = {s['count']:,}    |    Mean error {s['mape_pct']:.2f}%    |    Median {s['median_ape_pct']:.2f}%    |    P95 {s['p95_ape_pct']:.2f}%",ha='center',fontsize=11,color='#183247')
    fig.text(.5,.06,f"Within ±20%: {s['within_20_pct']:.1f}%    ·    All observations included; dense clouds use count bins, without subsampling.",ha='center',fontsize=9,color='#506477')
    fig.text(.5,.022,'GPU operator timing only; serving throughput and energy require separate validation.',ha='center',fontsize=9,color='#506477')
    fig.subplots_adjust(top=.78,bottom=.25,left=.085,right=.975,wspace=.35)
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
    overall=scores(data);figures=[];main_embeds=[];gpu_embeds=[];transfer_embeds=[];embeds=[];cohort_metrics={}
    td=read_predictions(OUT/'gpu-transfer-predictions.jsonl.gz');ts=scores(td)
    for source in ['kaist','vidur']:
        mask=data['source']==source;part={key:value[mask] for key,value in data.items()}
        cohort_metrics[source]=residual_figure(part,source+'-main',LABELS[source]+' · GPU timing accuracy',
            'Grouped five-fold outer CV · function and scale selection inside three training-only inner folds')
        figures.append(source+'-main')
        main_embeds.append(f"### {LABELS[source]}\n\n![{LABELS[source]} cross-validated accuracy]({source}-main.png)\n\n[Vector figure]({source}-main.svg).")
        op_rows=[];operator_embeds=[]
        for op in sorted(report['groups'][source]):
            metric=report['groups'][source][op]['cv'];slug=op.replace(':','-');name=source+'-operator-'+slug
            selected=(data['source']==source)&(data['op']==op)
            operator_data={key:value[selected] for key,value in data.items()}
            residual_figure(operator_data,name,LABELS[source]+' · '+NAMES[op],
                'Grouped nested cross-validation · identical validation protocol to the corpus overview')
            figures.append(name);family=report['models'][source][op]['family']
            op_rows.append([NAMES[op],f"{metric['count']:,}",f"{metric['mape_pct']:.2f}%",f"{metric['p95_ape_pct']:.2f}%",family])
            operator_embeds.append(f"#### {NAMES[op]}\n\n![{LABELS[source]} {NAMES[op]}]({name}.png)\n\n[Vector figure]({name}.svg).")
        embeds.append('### '+LABELS[source]+'\n\n'+table(['Subcomponent','Observations','MAPE','P95','Deployed family'],op_rows)+'\n\n'+'\n\n'.join(operator_embeds))
        gpu_metrics={gpu:metric for gpu,metric in report['by_gpu'].items() if np.any(part['gpu']==gpu)}
        fig,axes=plt.subplots(1,len(gpu_metrics),figsize=(6*len(gpu_metrics),5.4),squeeze=False)
        for ax,(gpu,metric) in zip(axes[0],gpu_metrics.items()):
            selected=part['gpu']==gpu;y=part['measured_ns'][selected]/1000;p=part['predicted_ns'][selected]/1000
            prediction_marks(ax,y,p,source)
            ax.set(xlabel='Profiled time (µs)',ylabel='CV prediction (µs)',title=f"{gpu} · N={metric['count']:,}\nMAPE {metric['mape_pct']:.2f}% · P95 {metric['p95_ape_pct']:.2f}%");ax.grid(alpha=.14)
        fig.suptitle(LABELS[source]+' · known-GPU shape validation',fontsize=17,fontweight='bold',color='#183247')
        fig.tight_layout(rect=(0,.025,1,.9));save(fig,source+'-by-gpu');figures.append(source+'-by-gpu')
        gpu_embeds.append(f"![{LABELS[source]} by GPU]({source}-by-gpu.png)")
        selected=td['source']==source;transfer_data={key:value[selected] for key,value in td.items()}
        residual_figure(transfer_data,source+'-gpu-transfer',LABELS[source]+' · unseen-GPU transfer',
            'Leave-one-GPU-out validation · target GPU excluded from fitting and function/scale selection')
        figures.append(source+'-gpu-transfer')
        transfer_embeds.append(f"![{LABELS[source]} leave-one-GPU-out accuracy]({source}-gpu-transfer.png)\n\n[Vector figure]({source}-gpu-transfer.svg).")
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

## Separate corpus overviews

{chr(10).join(main_embeds)}

Each figure has two panels: **predicted versus profiled time** and the
**cumulative absolute-error distribution**. KAIST and Vidur are never overlaid.
Dense prediction clouds use hexagonal bins colored by observation count; smaller
plots show individual points. Every eligible observation contributes to the bins,
error distributions and metrics; there is no point subsampling, outlier removal,
or tail clipping. The signed-residual middle panel has been removed.
The numerical validation results and production coefficients are unchanged.

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

{chr(10).join(gpu_embeds)}

The pooled score is dominated by Vidur's denser attention grids. Equal-operator
macro MAPE is **{report['overall']['macro_operator_mape_pct']:.2f}%**. Per-GPU and
per-operator tables must accompany the pooled number. Latency-weighted error is
sum of absolute ns errors divided by sum of profiled ns, not serving-workload weighting.

## Unseen-GPU accuracy is a different test

{chr(10).join(transfer_embeds)}

{table(['Excluded GPU/cohort','Scored observations','MAPE','P95','Unsupported groups'],transfer_rows)}

The target GPU is absent from coefficient fitting **and** function/scale selection.
QK normalization cannot be learned when RTX PRO 6000 is withheld: RTX 4090 only
contains Llama, which lacks this operation. Those 304 targets are explicitly
unsupported rather than scored as a successful prediction. Thus this transfer
figure has **{ts['count']:,}** points, versus 599,451 in shape CV.
KAIST also confounds hardware with software version. This audit is not evidence
that peak specifications alone predict new GPU architectures within a fixed error.

## Subcomponent figures, separated by profiling corpus

KAIST has 14 measured operator groups; Vidur has 10. A missing group has no figure
in that corpus. The deployed runtime retains a documented extrapolation/fallback
where a scenario requests an operator missing from its precision-specific corpus.
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

“All data” across the two overview figures means **all imported compatible dense-decoder
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
    # GitHub relative image links redirect to raw.githubusercontent.com. Serve the
    # same committed files through Pages, as for the project logo/screenshots.
    report_text=re.sub(r'\]\(((?:kaist|vidur)-[a-z0-9_-]+\.(?:png|svg))\)',
        lambda match: ']('+'https://ryeii.github.io/LLM-ServeScope/docs/accuracy-2026-10-19/'+match[1]+'?v='+hashfile(OUT/match[1])[:12]+')',report_text)
    (OUT/'README.md').write_text(report_text)
    (OUT/'figure-summary.json').write_text(json.dumps(dict(overall=overall,transfer=ts,by_source=cohort_metrics,
        figure_count=len(figures),figures=figures,panels_per_accuracy_figure=2,signed_residual_panel=False,
        dense_mark_style='Log-count hexagonal bins above 5000 observations; scatter otherwise.',point_subsampling=False),indent=2)+'\n')
    # Retire the mixed-corpus figures so stale files cannot be mistaken for current plots.
    for stem in ['main-residuals','gpu-transfer-residuals','by-gpu']+[p.stem for p in OUT.glob('operator-*.png')]:
        for ext in ['png','svg']:
            path=OUT/(stem+'.'+ext)
            if path.exists():path.unlink()
    print(f'Rendered {len(figures)} corpus-separated figures; signed-residual panels removed.',flush=True)
if __name__=='__main__':main()
