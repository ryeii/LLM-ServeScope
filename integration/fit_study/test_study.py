"""Offline study checks: independent units/geometry, split leakage and baseline parity.
Run: .venv-sim/bin/python -B -m unittest integration.fit_study.test_study -v
"""
import csv,io,json,unittest
from collections import defaultdict
import numpy as np
from backend.calibration import work_features
from backend.catalog import MODELS,HARDWARE
from .data import ROOT,kaist_rows,make_row,read_archived
from .functions import FAMILIES,fit,predict,evaluate_array

class StudyTests(unittest.TestCase):
    def test_original_baseline_fit_reproduced(self):
        rows=kaist_rows();published=json.loads((ROOT/'docs/calibration-validation.json').read_text())
        self.assertEqual(len(rows),9411)
        # The baseline active-set procedure should reproduce the deployed timings.
        for op in ['qkv_proj','act_fn','attention:prefill']:
            part=[r for r in rows if r['op']==op];m=fit('F0',part)
            x=np.array([r['x'] for r in part]);old=np.asarray(published['fits'][op])
            expected=np.maximum.reduce((np.ones(len(x)),x[:,1],x[:,:3]@old))
            np.testing.assert_allclose(predict(m,x),expected,rtol=1e-8,atol=1e-5)

    def test_workload_grouping_is_independent_of_gpu_and_target(self):
        rows=kaist_rows();groups=defaultdict(set)
        for r in rows:groups[r['key']].add(r['fold'])
        self.assertTrue(all(len(folds)==1 for folds in groups.values()))
        r=rows[0];m=MODELS['llama8'];h=HARDWARE['rtx4090']
        a=make_row('kaist','gpu A','model A',1,m,h,'qkv_proj',32,1)
        b=make_row('kaist','gpu B','model B',1,m,h,'qkv_proj',32,1e9)
        self.assertEqual(a['x'],b['x']);self.assertEqual(a['fold'],b['fold']);self.assertEqual(a['key'],b['key'])

    def test_replicated_work_stays_in_one_fold_across_tp(self):
        m=MODELS['llama8'];h=HARDWARE['rtx6000']
        for op in ['layernorm','final_layernorm','sampler']:
            a=make_row('kaist','a','m',1,m,h,op,32,1)
            b=make_row('kaist','a','m',2,m,h,op,32,999)
            self.assertEqual(a['key'],b['key']);self.assertEqual(a['fold'],b['fold'])
        wider=dict(m,intermediate=2*m['intermediate'])
        a=make_row('kaist','a','m',1,m,h,'act_fn',32,1)
        b=make_row('kaist','a','m2',2,wider,h,'act_fn',32,999)
        self.assertEqual(a['key'],b['key'])

    def test_vidur_timing_unit_from_published_csv(self):
        text=read_archived('data/profiling/compute/h100/meta-llama/Llama-2-7b-hf/attention.csv')
        r=next(csv.DictReader(io.StringIO(text)))
        # Source CudaTimer stores ms; 0.007 ms is 7000 ns, not 7 ns or 7 ms.
        self.assertAlmostEqual(float(r['time_stats.attn_decode.mean'])*1e6,7000)

    def test_embedding_corrected_feature_replication(self):
        m=MODELS['llama8'];h=HARDWARE['rtx6000'];n=32
        a=make_row('kaist','a','m',1,m,h,'embedding',n,1)
        b=make_row('kaist','a','m',2,m,h,'embedding',n,1)
        self.assertEqual(a['x'][6],b['x'][6])
        expected=n*(4*4096+8)/(1.5976*1e12)*1e9
        self.assertAlmostEqual(a['x'][6],expected)
        self.assertNotEqual(a['x'][2],b['x'][2]) # Historical baseline is preserved.

    def test_cache_ratio_is_not_a_bandwidth_proxy(self):
        m=MODELS['llama8'];h=dict(HARDWARE['rtx6000'],l2_bytes=128*1024**2)
        a=make_row('kaist','a','m',1,m,h,'act_fn',512,1)
        h['dram_bandwidth_TBps']*=2
        b=make_row('kaist','a','m',1,m,h,'act_fn',512,1)
        self.assertAlmostEqual(a['x'][5],b['x'][5])
        self.assertAlmostEqual(a['x'][2],2*b['x'][2])

    def test_function_nonnegativity_and_arithmetic_floor(self):
        x=np.array([[1.,1.,2.,4.,.2,.4,2.],[1.,100.,20.,150.,2.,4.,20.]])
        for f in FAMILIES:
            count=4 if f in ('F5','F7') else 5 if f=='F6' else 3
            p=evaluate_array(f,[1.]*count,x,.5)
            self.assertTrue(np.all(p>=x[:,1]));self.assertTrue(np.all(np.isfinite(p)))

    def test_h100_and_codellama_excluded_from_selection(self):
        # Selection report is generated before audit predictions. Its row count
        # is independently documented; neither reserved group enters that corpus.
        path=ROOT/'docs/fit-study-2026-10-19/vidur/results.json'
        if not path.exists():self.skipTest('Run full offline study first')
        r=json.loads(path.read_text());self.assertEqual(r['count'],262240)
        self.assertIn('selection_oof_warning',r)
        self.assertEqual(r['vidur_adapter_info']['unique_observations'],590040)
        audit=ROOT/'docs/fit-study-2026-10-19/vidur/audits.json'
        if audit.exists():
            a=json.loads(audit.read_text())
            self.assertEqual(a['model_codellama_seen_gpus']['baseline']['tested_count'],131120)
            self.assertEqual(a['gpu_and_model_h100_codellama']['baseline']['tested_count'],65560)

if __name__=='__main__':unittest.main()
