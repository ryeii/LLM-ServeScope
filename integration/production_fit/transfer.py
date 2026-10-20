"""Leave-one-GPU-out nested model selection, separate from within-GPU shape CV."""
import gzip,json
from pathlib import Path
import numpy as np
from .train import OUT,load_rows,fold_ids,cached_fit,predict_config,metrics,hashfile


def main():
    report=json.loads((OUT/'validation.json').read_text());rows,_=load_rows()
    signature=dict(report['signature'],transfer_sha256=hashfile(Path(__file__)))
    archive=OUT/'gpu-transfer-predictions.jsonl.gz';audits={}
    with gzip.GzipFile(filename=str(archive),mode='wb',mtime=0) as stream:
        for source in ['kaist','vidur']:
            cohort=[r for r in rows if r['source']==source]
            for gpu in sorted({r['gpu'] for r in cohort}):
                training=[r for r in cohort if r['gpu']!=gpu];test_all=[r for r in cohort if r['gpu']==gpu];by_op={};ys=[];ps=[];ops=[];unsupported=[]
                for op in sorted({r['op'] for r in test_all}):
                    train=[r for r in training if r['op']==op];test=[r for r in test_all if r['op']==op]
                    if not train:unsupported.append(dict(operator=op,count=len(test)));continue
                    x=np.array([r['x'] for r in train]);y=np.array([r['ns'] for r in train]);n=np.array([r['n'] for r in train]);_,inner=fold_ids(train)
                    fit=cached_fit(source,op,'holdout-'+gpu,x,y,n,inner,signature)
                    pred=predict_config(fit['model'],np.array([r['x'] for r in test]),np.array([r['n'] for r in test]));target=np.array([r['ns'] for r in test])
                    by_op[op]=dict(metrics=metrics(target,pred),training_count=len(train),tested_count=len(test),selection=fit)
                    ys.extend(target);ps.extend(pred);ops.extend([op]*len(test))
                    for r,p in zip(test,pred):
                        record={key:r[key] for key in ['source','gpu','model','tp','op','n']};record.update(measured_ns=r['ns'],predicted_ns=float(p))
                        stream.write((json.dumps(record,separators=(',',':'))+'\n').encode())
                audits[source+':'+gpu]=dict(held_out_gpu=gpu,source=source,training_count=len(training),tested_count=len(ys),unsupported=unsupported,metrics=metrics(ys,ps,ops),by_operator=by_op)
                print('GPU transfer',source,gpu,audits[source+':'+gpu]['metrics'],flush=True)
    (OUT/'gpu-transfer-validation.json').write_text(json.dumps(dict(signature=signature,audits=audits,prediction_archive_sha256=hashfile(archive),scope='Every target GPU is excluded from both coefficient fitting and function/scale selection; software version is confounded with GPU in KAIST.'),indent=2)+'\n')
if __name__=='__main__':main()
