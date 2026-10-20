"""Audit values exposed for the result-formula UI against the simulation itself."""
import copy,unittest
from backend.model import DEFAULT,simulate
from backend.gpu_functions import evaluate

class FormulaAccountingTests(unittest.TestCase):
    def test_energy_and_resource_contributions_reconcile(self):
        for separated,component in [(True,False),(False,False),(False,True)]:
            c=copy.deepcopy(DEFAULT);c.update(requests=4,input_tokens=32,output_tokens=3,separated=separated)
            if component:
                c['prefill']['hardware'].update(pj_per_flop=.4,pj_per_mem_byte=3)
            r=simulate(c);a=r['accounting'];t=r['metrics']['elapsed_ms']/1000
            energy=sum(q['idle_watts']*q['gpus']*t+q['dynamic_j'] for q in a['energy_terms'])
            self.assertAlmostEqual(energy,r['metrics']['energy_j'])
            for key,value in r['component_work_ms'].items():
                reconstructed=0
                for e in r['timeline']:
                    pool=r['config']['decode' if e['phase']=='decode' else 'prefill']
                    reconstructed+=e['components'][key]*(1 if e['phase']=='transfer' else pool['tp']*pool['ep'])
                self.assertAlmostEqual(value,reconstructed)
            for key in ['hbm_bytes','network_bytes']:
                self.assertAlmostEqual(r['traffic'][key],sum(counters[key] for counters in a['phase_traffic'].values()))
            for phase,m in r['memory'].items():
                p=r['config'][phase]
                weight=(a['shared_parameters']+a['expert_parameters']/p['ep'])*2/(p['tp']*p['pp'])*1e-9
                self.assertAlmostEqual(weight,m['weights_GB'])

    def test_coefficient_metadata_reconstructs_exact_operator_time(self):
        r=simulate(DEFAULT)
        for op in r['layers']:
            if 'fit_parameters' not in op:continue
            base=evaluate(op['fit_parameters'],op['shape_features_ns'],op['fit_token_count'],op['fit_compute_floor_ns'])
            actual=max(1,op['fit_compute_floor_ns'],base*op['fit_engine_multiplier']+op['fit_engine_offset_ns']+op['fit_extra_launch_ns'])/1e6
            self.assertAlmostEqual(base,op['fit_base_ns'])
            self.assertAlmostEqual(actual,op['duration_ms'])

if __name__=='__main__':unittest.main()
