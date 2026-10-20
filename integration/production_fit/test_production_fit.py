"""Runtime/offline parity and leakage checks for the production evaluation."""
import collections,gzip,json,unittest
import numpy as np
from backend.calibration import dataset,work_features
from backend.catalog import HARDWARE,MODELS
from backend.gpu_functions import evaluate,shape_features
from integration.fit_study.data import ROOT,kaist_rows
from integration.fit_study.functions import evaluate_array
from .train import OUT,fold_ids,split_key,predict_config

class ProductionFitTests(unittest.TestCase):
    def test_pure_python_matches_every_numpy_function(self):
        rng=np.random.default_rng(781);x=np.exp(rng.uniform(-3,10,(120,7)));x[:,0]=1;n=np.arange(1,121)
        for family in ['F0','F1','F2','F3','F4','F5','F6','F7','F8']:
            count=5 if family=='F6' else 4 if family in ('F5','F7') else 3
            coefficients=np.exp(rng.uniform(-3,3,count)).tolist()
            model=dict(family=family,coefficients=coefficients,scale=2.)
            expected=evaluate_array(family,coefficients,x,2.)
            actual=[evaluate(model,row,tokens) for row,tokens in zip(x,n)]
            np.testing.assert_allclose(actual,expected,rtol=1e-12)
        model=dict(family='F9',threshold_tokens=16,branches=dict(small_index=[2500,20],gather=[3000,.3]))
        # Embedding arithmetic is zero in both implementations.
        x[:,1]=0
        np.testing.assert_allclose([evaluate(model,r,k) for r,k in zip(x,n)],predict_config(model,x,n))

    def test_runtime_features_match_every_kaist_training_coordinate(self):
        for r in kaist_rows():
            m=MODELS[r['model']];h=HARDWARE[r['gpu']];name=r['op'].split(':')[0]
            base=work_features(name,m,h,r['tp'],r['n'],**r['shape'])
            actual=shape_features(name,m,h,r['tp'],r['n'],base,**r['shape'])
            np.testing.assert_allclose(actual,r['x'],rtol=1e-12,atol=1e-9)

    def test_no_work_query_crosses_folds_and_fold_counts_cover_archive(self):
        report=json.loads((OUT/'validation.json').read_text());seen={};counts=collections.Counter();total=0
        with gzip.open(OUT/'cross-validated-predictions.jsonl.gz','rt') as stream:
            for line in stream:
                r=json.loads(line);key=r['group_key'];fold=r['fold']
                if key in seen:self.assertEqual(seen[key],fold)
                else:seen[key]=fold
                counts[(r['source'],r['op'],fold)]+=1;total+=1
                # Validate a deterministic slice of actual exported predictions.
                if total%997==0:
                    model=report['groups'][r['source']][r['op']]['outer_folds'][fold]['selection']['model']
                    self.assertAlmostEqual(evaluate(model,r['features'],r['n']),r['predicted_ns'],places=6)
        self.assertEqual(total,599451)
        for source,groups in report['groups'].items():
            for op,g in groups.items():
                for f in g['outer_folds']:
                    self.assertEqual(counts[(source,op,f['fold'])],f['tested_count'])
                    self.assertEqual(g['count']-f['tested_count'],f['training_count'])

    def test_final_coefficients_equal_exported_all_data_fits(self):
        r=json.loads((OUT/'validation.json').read_text());d=dataset()
        for source,models in r['models'].items():self.assertEqual(models,d['cohorts'][source]['models'])
        self.assertEqual(sum(p['observation_count'] for p in d['domains']),599451)
        self.assertEqual(d['cohorts']['kaist']['training_count'],9411)
        self.assertEqual(d['cohorts']['vidur']['training_count'],590040)

    def test_vidur_runtime_features_for_all_gpu_model_tp_groups(self):
        mapping={'meta-llama/Llama-2-7b-hf':'vidur_llama7','meta-llama/Llama-2-70b-hf':'vidur_llama70','codellama/CodeLlama-34b-Instruct-hf':'vidur_codellama34'}
        seen=set()
        with gzip.open(OUT/'cross-validated-predictions.jsonl.gz','rt') as stream:
            for line in stream:
                r=json.loads(line)
                if r['source']!='vidur':continue
                key=(r['gpu'],r['model'],r['tp'],r['op'])
                if key in seen:continue
                seen.add(key);m=MODELS[mapping[r['model']]];h=HARDWARE[r['gpu']];name=r['op'].split(':')[0]
                base=work_features(name,m,h,r['tp'],r['n'],precision='fp16',**r['shape'])
                features=shape_features(name,m,h,r['tp'],r['n'],base,precision='fp16',query_tokens=r['query_tokens'],**r['shape'])
                np.testing.assert_allclose(features,r['features'],rtol=1e-12,atol=1e-9)
        self.assertEqual(len(seen),360)

if __name__=='__main__':unittest.main()
