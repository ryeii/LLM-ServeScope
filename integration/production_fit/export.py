"""Export coefficient-only full-data models after nested validation."""
import gzip,json,hashlib
from pathlib import Path
from collections import defaultdict
from backend.catalog import HARDWARE,MODELS
from integration.fit_study.data import ROOT,hashfile
from integration.fit_study.functions import fit_fixed
from .train import OUT,load_rows
import numpy as np


def main():
    report=json.loads((OUT/'validation.json').read_text());rows,_=load_rows()
    groups=defaultdict(list)
    for r in rows:groups[(r['source'],r['gpu'],r['model'],r['tp'])].append(r)
    mapping={'meta-llama/Llama-2-7b-hf':'vidur_llama7','meta-llama/Llama-2-70b-hf':'vidur_llama70','codellama/CodeLlama-34b-Instruct-hf':'vidur_codellama34'}
    domains=[]
    for (source,gpu,model,tp),part in sorted(groups.items()):
        mid=mapping.get(model,model);m=MODELS[mid];h=HARDWARE[gpu]
        bounds={}
        for op in sorted({r['op'] for r in part}):
            subset=[r for r in part if r['op']==op]
            bounds[op]={key:[min(r['shape'][key] for r in subset),max(r['shape'][key] for r in subset)] for key in ['pc','kp','nd','kv']} if op.startswith('attention:') else {'n':[min(r['n'] for r in subset),max(r['n'] for r in subset)]}
        software=({'vllm_version':'0.19.0' if gpu=='rtx4090' else '0.28.0','cuda_version':'12.8' if gpu=='rtx4090' else '13.0'} if source=='kaist' else {'engine':'Vidur/Sarathi-vLLM-derived profiler','versions':'Historical build/clock manifest unavailable'})
        domains.append(dict(id=f'{source}:{gpu}:{mid}:tp{tp}',cohort=source,precision='bf16' if source=='kaist' else 'fp16',hardware_id=gpu,model_id=mid,tp=tp,hardware=h,model=m,software=software,bounds=bounds,observation_count=len(part)))
    cohorts={}
    for source,models in report['models'].items():
        fallback={}
        for op in models:
            part=[r for r in rows if r['source']==source and r['op']==op]
            fallback[op]=fit_fixed('F0',np.array([r['x'] for r in part]),np.array([r['ns'] for r in part]))
        cohorts[source]=dict(precision='bf16' if source=='kaist' else 'fp16',models=models,fallback_models=fallback,training_count=sum(r['source']==source for r in rows))
    data=dict(schema_version='servescope.gpu-fit.v2',upstream_revision='b09c7aaab4b072d73136641841cc2688616c49eb',
        vidur_revision='abae7f63aa857300f5cdc6f5e0d27860cd24721b',cohorts=cohorts,domains=domains,training_observations=len(rows),
        fit_method='Fewest parameters within 1 percentage point of best inner three-fold MAPE; five grouped outer folds evaluate the complete selection procedure. Deployment refits each selected function on every observation in its precision/software cohort.',
        accuracy_report='https://github.com/ryeii/LLM-ServeScope/blob/main/docs/accuracy-2026-10-19/README.md',
        validation={key:report[key] for key in ['count','protocol','overall','baseline','by_source','by_gpu','by_operator']},
        sources=report['signature'],prediction_archive_sha256=report['prediction_archive_sha256'],
        runtime_sha256=hashfile(ROOT/'backend/calibration.py'),gpu_functions_sha256=hashfile(ROOT/'backend/gpu_functions.py'),catalog_sha256=hashfile(ROOT/'backend/catalog.py'),exporter_sha256=hashfile(Path(__file__)))
    transfer_path=OUT/'gpu-transfer-validation.json'
    if transfer_path.exists():
        transfer=json.loads(transfer_path.read_text())
        data['validation']['gpu_transfer']={key:{k:v for k,v in audit.items() if k!='by_operator'} for key,audit in transfer['audits'].items()}
    data['payload_sha256']=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    (ROOT/'backend/gpu-fit.json').write_text(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    print('Exported',len(domains),'domains;',len(rows),'training observations; coefficient-only artifact')
if __name__=='__main__':main()
