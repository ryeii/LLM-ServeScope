"""Offline exhaustive small-family search. Run from repository root as a module.
All training data and predictions remain outside the application runtime.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,os,platform,sys,time
import numpy as np
import scipy
from .data import ROOT,kaist_rows,vidur_rows,arrays,hashfile
from .functions import FAMILIES,fit,predict,metrics
OUT=ROOT/'docs/fit-study-2026-10-19'

def grouped(rows):
    return {op:[r for r in rows if r['op']==op] for op in sorted(set(r['op'] for r in rows))}

def score(rows,preds): return metrics([r['ns'] for r in rows],preds,[r['op'] for r in rows])

def cv_candidate(family,rows,folder):
    path=folder/(family+'.json.gz')
    if path.exists(): return json.load(gzip.open(path,'rt'))
    start=time.time(); predictions=[None]*len(rows); by_op={};models={}
    index={id(r):i for i,r in enumerate(rows)}
    for op,subset in grouped(rows).items():
        full=fit(family,subset);models[op]=full
        train_prediction=predict(full,arrays(subset)[0]);folds=[]
        for fold in range(5):
            train=[r for r in subset if r['fold']!=fold];test=[r for r in subset if r['fold']==fold]
            model=fit(family,train);p=predict(model,arrays(test)[0])
            for r,v in zip(test,p): predictions[index[id(r)]]=float(v)
            folds.append(dict(fold=fold,metrics=score(test,p),model=model))
        by_op[op]=dict(training=score(subset,train_prediction),folds=folds,
                       oof=score(subset,[predictions[index[id(r)]] for r in subset]))
        print(f'{folder.name} {family} {op}: OOF {by_op[op]["oof"]["mape_pct"]:.2f}%',flush=True)
    full_predictions=[float(predict(models[r['op']],np.array([r['x']]))[0]) for r in rows]
    result=dict(family=family,description=FAMILIES[family],training=score(rows,full_predictions),
      oof=score(rows,predictions),by_operator=by_op,models=models,oof_predictions=predictions,
      elapsed_seconds=time.time()-start)
    path.write_bytes(gzip.compress((json.dumps(result,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
    return result

def select_candidates(results):
    selected={};ops=results['F0']['by_operator']
    for op in ops:
        ranked=sorted(results,key=lambda f:results[f]['by_operator'][op]['oof']['mape_pct'])
        best=ranked[0];best_data=results[best]['by_operator'][op]
        eligible=[]; details={}
        for f,r in results.items():
            d=r['by_operator'][op];delta=d['oof']['mape_pct']-best_data['oof']['mape_pct']
            differences=np.array([a['metrics']['mape_pct']-b['metrics']['mape_pct'] for a,b in zip(d['folds'],best_data['folds'])])
            se=float(np.std(differences,ddof=1)/np.sqrt(len(differences)))
            tolerance=max(1.,se)
            details[f]=dict(delta_mape_pp=delta,paired_fold_se_pp=se,tolerance_pp=tolerance)
            if delta<=tolerance:eligible.append(f)
        chosen=min(eligible,key=lambda f:(FAMILIES[f]['parameters'],int(f[1:])))
        selected[op]=dict(family=chosen,best_accuracy_family=best,eligible=eligible,comparison=details,
                          model=results[chosen]['models'][op])
    return selected

def combined_selection_oof(rows,results,selected):
    return [results[selected[r['op']]['family']]['oof_predictions'][i] for i,r in enumerate(rows)]

def evaluate_split(train,test,family_by_operator):
    models={op:fit(family_by_operator[op],part) for op,part in grouped(train).items() if op in family_by_operator}
    supported=[r for r in test if r['op'] in models]
    pred=[float(predict(models[r['op']],np.array([r['x']]))[0]) for r in supported]
    return dict(training_count=len(train),tested_count=len(supported),unsupported=sorted(set(r['op'] for r in test)-set(models)),
      metrics=score(supported,pred),by_operator={op:score(part,[float(predict(models[op],np.array([r['x']]))[0]) for r in part]) for op,part in grouped(supported).items()},models=models),pred,supported

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--families',nargs='*',default=list(FAMILIES));parser.add_argument('--corpora',nargs='*',default=['kaist','vidur']);parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    k=kaist_rows();v,vi=vidur_rows()
    development_v=[r for r in v if r['gpu']!='h100' and not r['model'].startswith('codellama/')]
    for source,rows in [('kaist',k),('vidur',development_v)]:
        if source not in args.corpora: continue
        folder=OUT/source;folder.mkdir(exist_ok=True)
        fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        signature=dict(rows_sha256=fingerprint,functions_sha256=hashfile(Path(__file__).with_name('functions.py')),data_adapter_sha256=hashfile(Path(__file__).with_name('data.py')),runner_sha256=hashfile(Path(__file__)),protocol_sha256=hashfile(OUT/'protocol.json'))
        marker=folder/'signature.json'
        if marker.exists():assert json.load(open(marker))==signature,'Cached results invalid: archive/remove this corpus cache before rerun'
        else:marker.write_text(json.dumps(signature,indent=2)+'\n')
        results={f:cv_candidate(f,rows,folder) for f in args.families}
        if set(results)!=set(FAMILIES):continue
        selection=select_candidates(results)
        # Selection uses these CV results, so this composite OOF score is
        # selection-biased; hardware/model audits are reported separately.
        compact={f:{key:value for key,value in result.items() if key!='oof_predictions'} for f,result in results.items()}
        report=dict(source=source,count=len(rows),operators=sorted(set(r['op'] for r in rows)),signature=signature,
          candidates=compact,selection=selection,selection_oof=score(rows,combined_selection_oof(rows,results,selection)),
          selection_oof_warning='Family selection reuses the outer OOF scores; selected-composite OOF is descriptive and optimistic, not unbiased post-selection test accuracy.',
          environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
          vidur_adapter_info=vi if source=='vidur' else None)
        (folder/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(source,'SELECTION',json.dumps({o:s['family'] for o,s in selection.items()}),report['selection_oof'],flush=True)
        if args.audit:
            chosen={op:s['family'] for op,s in selection.items()};base={op:'F0' for op in chosen};audits={}
            if source=='kaist':
                splits={f'gpu_{gpu}':([r for r in rows if r['gpu']!=gpu],[r for r in rows if r['gpu']==gpu]) for gpu in sorted(set(r['gpu'] for r in rows))}
                splits['model_qwen32']=([r for r in rows if r['model']!='qwen32'],[r for r in rows if r['model']=='qwen32'])
            else:
                splits={
                  'gpu_h100_llama':(rows,[r for r in v if r['gpu']=='h100' and not r['model'].startswith('codellama/')]),
                  'model_codellama_seen_gpus':(rows,[r for r in v if r['gpu']!='h100' and r['model'].startswith('codellama/')]),
                  'gpu_and_model_h100_codellama':(rows,[r for r in v if r['gpu']=='h100' and r['model'].startswith('codellama/')])}
            for split,(train,test) in splits.items():
                audits[split]={}
                split_chosen=chosen
                if source=='kaist':
                    # Retrain family selection too: test-GPU/model measurements
                    # cannot choose which formula to use in a transfer audit.
                    sf=folder/(split+'-selection');sf.mkdir(exist_ok=True)
                    sr={f:cv_candidate(f,train,sf) for f in FAMILIES}
                    split_selection=select_candidates(sr)
                    split_chosen={op:s['family'] for op,s in split_selection.items()}
                    audits[split]['training_only_selection']=split_selection
                for name,families in [('baseline',base),('selected',split_chosen)]:
                    result,pred,supported=evaluate_split(train,test,families);audits[split][name]=result
                    records=[dict(source=r['source'],gpu=r['gpu'],model=r['model'],tp=r['tp'],op=r['op'],n=r['n'],shape=r['shape'],measured_ns=r['ns'],predicted_ns=p) for r,p in zip(supported,pred)]
                    (folder/(split+'-'+name+'-predictions.json.gz')).write_bytes(gzip.compress(json.dumps(records,separators=(',',':')).encode(),mtime=0))
                    print(source,split,name,result['metrics'],flush=True)
            (folder/'audits.json').write_text(json.dumps(audits,indent=2)+'\n')
if __name__=='__main__':main()
