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

    def test_executed_schedule_costs_and_waits_reconcile(self):
        for mode,separated,output in [('calibrated',True,4),('roofline',False,3),('calibrated',True,1)]:
            c=copy.deepcopy(DEFAULT);c.update(requests=7,input_tokens=65,prefill_chunk=32,output_tokens=output,arrival_rate=1000,timing_mode=mode,separated=separated)
            c['prefill'].update(pp=2,batch_size=3,max_concurrent_batches=1)
            c['decode'].update(pp=2,batch_size=2,max_concurrent_batches=1)
            r=simulate(c);events={e['event_id']:e for e in r['timeline']}
            for e in events.values():
                previous=events.get(e['previous_resource_event'])
                self.assertAlmostEqual(e['start_ms'],max(e['ready_ms'],previous['end_ms'] if previous else 0))
                self.assertAlmostEqual(e['queue_ms'],e['start_ms']-e['ready_ms'])
                if 'shape_id' in e:
                    shape=r['execution_shapes'][e['shape_id']]
                    cost=sum(o['duration_ms'] for o in shape['operations'])*e['layers_per_stage']+sum(o['duration_ms'] for o in e['endpoint_timing'])+e['head_fallback_ms']
                    self.assertAlmostEqual(e['duration_ms'],cost)
                    self.assertEqual(shape['context'],e['context'])
                else:
                    cost=e['payload_bytes']/(e['link_bandwidth_GBps']*1e9)*1000+e['link_latency_us']/1000
                    self.assertAlmostEqual(e['duration_ms'],cost)
            for q in r['requests']:
                for phase,begin,end,admitted in [('prefill',q['batch_ready_ms'],q['prefill_end_ms'],q['prefill_admitted_ms']),('decode',q['first_token_ms'],q['finish_ms'],q['decode_admitted_ms']),('transfer',q['prefill_end_ms'],q['first_token_ms'],q['prefill_end_ms'])]:
                    path=[e for e in events.values() if q['id'] in e['request_ids'] and e['phase']==phase]
                    service=sum(e['duration_ms'] for e in path)
                    wait=sum(e['queue_ms'] for e in path)+admitted-begin
                    self.assertAlmostEqual(end-begin,service+wait)
            self.assertEqual(r['metrics']['elapsed_ms'],max(q['finish_ms'] for q in r['requests']))

    def test_fit_feature_inputs_reconstruct_actual_features(self):
        from backend.calibration import VECTOR
        r=simulate(DEFAULT)
        for shape in r['execution_shapes'].values():
            for o in shape['operations']:
                if 'fit_work' not in o: continue
                w=o['fit_work'];h=r['config'][shape['phase']]['hardware']
                C=w['flops']/(w['peak_tflops']*1000*h['clock_scale'])*.65/h['compute_efficiency']
                M=w['traffic_bytes']/(h['dram_bandwidth_TBps']*1000)*.8/h['memory_efficiency']
                self.assertAlmostEqual(C,o['fit_features_ns'][1])
                self.assertAlmostEqual(M,o['fit_features_ns'][2])

if __name__=='__main__':unittest.main()
