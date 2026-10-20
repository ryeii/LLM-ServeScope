"""Validate dispatch selection, independent branch fitting and held-out provenance."""
import json, math, unittest
import numpy as np
from .data import ROOT, kaist_rows, hashfile
from .embedding_piecewise import (
    OUT, fit_piecewise, predict_piecewise, predictions, query_fold, regime,
    vidur_embeddings,
)

class PiecewiseEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((OUT/'results.json').read_text())
        cls.rows = sorted([r for r in kaist_rows() if r['op']=='embedding'],
                          key=lambda r:(r['gpu'],r['model'],r['tp'],r['n']))

    def test_two_independent_functions_recover_discontinuous_data(self):
        rows=[]
        for n in [1, 4, 8, 16, 17, 20, 32, 64]:
            memory=2.*n
            target=100.+3.*memory if n<=16 else 50.+.2*memory
            rows.append(dict(n=n, x=[1,0,0,0,0,0,memory], ns=target))
        model=fit_piecewise(rows)
        np.testing.assert_allclose(predictions(model,rows),[r['ns'] for r in rows],rtol=1e-12)
        self.assertEqual(model['parameter_count'],4)
        self.assertFalse(model['threshold_fitted'])
        self.assertLess(predict_piecewise(model,17,34),predict_piecewise(model,16,32))
        self.assertEqual(regime(16),'small_index')
        self.assertEqual(regime(17),'gather')

    def test_invalid_queries_and_modified_threshold_are_rejected(self):
        model=self.result['kaist']['model']
        for n in [0,-1,1.5,math.inf,math.nan]:
            with self.assertRaises(ValueError):predict_piecewise(model,n,1.)
        for memory in [-1,math.inf,math.nan]:
            with self.assertRaises(ValueError):predict_piecewise(model,1,memory)
        altered=dict(model,threshold_tokens=20)
        with self.assertRaises(ValueError):predict_piecewise(altered,16,1.)
        with self.assertRaises(ValueError):fit_piecewise(self.rows[:1])

    def test_nonnegative_costs_and_bandwidth_response_within_each_branch(self):
        model=self.result['kaist']['model']
        for branch in model['branches'].values():
            self.assertGreaterEqual(branch['intercept_ns'],0)
            self.assertGreaterEqual(branch['memory_scale'],0)
        for n in [1,16,17,128,2048]:
            self.assertLessEqual(predict_piecewise(model,n,50),predict_piecewise(model,n,100))
            self.assertGreaterEqual(predict_piecewise(model,n,0),1)

    def test_equal_work_groups_ignore_ids_and_vocabulary_shards(self):
        row=self.rows[0]
        altered=dict(row,gpu='unseen GPU',model='unseen model',tp=32,ns=1e12,
                     geometry=[row['geometry'][0],1,1,1,1,999999])
        self.assertEqual(query_fold(row),query_fold(altered))
        # Prediction inputs are only the branch query and hardware-derived work.
        np.testing.assert_array_equal(predictions(self.result['kaist']['model'],[row]),
                                      predictions(self.result['kaist']['model'],[altered]))

    def test_exported_predictions_use_training_only_fold_coefficients(self):
        records=self.result['kaist_oof_predictions']
        self.assertEqual(len(records),len(self.rows))
        for fold in self.result['kaist']['folds']:
            train=[r for r in self.rows if query_fold(r)!=fold['fold']]
            test=[(r,p) for r,p in zip(self.rows,records) if query_fold(r)==fold['fold']]
            self.assertEqual(len(train),fold['training_count'])
            self.assertEqual(len(test),fold['tested_count'])
            refit=fit_piecewise(train)
            for branch in ['small_index','gather']:
                for key in ['intercept_ns','memory_scale']:
                    self.assertAlmostEqual(refit['branches'][branch][key],fold['model']['branches'][branch][key])
            for row,record in test:
                self.assertEqual((row['gpu'],row['model'],row['tp'],row['n']),
                                 (record['gpu'],record['model'],record['tp'],record['n']))
                self.assertAlmostEqual(predict_piecewise(refit,row['n'],row['x'][6]),record['predicted_ns'])

    def test_external_corpus_has_expected_unique_queries_and_reservations(self):
        rows=vidur_embeddings()
        self.assertEqual(len(rows),9324)
        self.assertEqual(len({(r['gpu'],r['model'],r['tp'],r['n']) for r in rows}),9324)
        dev=[r for r in rows if r['gpu']!='h100' and not r['model'].startswith('codellama/')]
        self.assertEqual(len(dev),self.result['vidur']['count'])
        self.assertEqual(len(dev),4144)

    def test_sources_match_export_and_deployed_baseline_is_preserved(self):
        self.assertEqual(hashfile(ROOT/'integration/fit_study/embedding_piecewise.py'),
                         self.result['provenance']['implementation_sha256'])
        self.assertEqual(hashfile(ROOT/'integration/fit_study/data.py'),
                         self.result['provenance']['adapter_sha256'])
        # The original deployed baseline remains archived after the later production promotion.
        provenance=json.loads((ROOT/'docs/fit-study-2026-10-19/provenance.json').read_text())
        self.assertEqual(hashfile(ROOT/'docs/fit-study-2026-10-19/baseline-gpu-fit.json'),
                         provenance['baseline_gpu_fit_sha256'])

if __name__=='__main__':unittest.main()
