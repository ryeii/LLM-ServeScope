import copy,json,math,unittest
from pathlib import Path
from backend.model import DEFAULT,simulate
from backend.storage import latency_ms
ROOT=Path(__file__).resolve().parents[1]

def scenario(mode='ssd'):
    c=copy.deepcopy(DEFAULT)
    c.update(model_id='llama8',requests=2,input_tokens=512,output_tokens=3,arrival_rate=1000)
    for phase in ['prefill','decode']:c[phase].update(tp=1,pp=1,ep=1,batch_size=2)
    c['active_kv']['placement']=mode
    c['storage'].update(enabled=False,ssd_transfer_mode='host-staged',publication_mode='blocking')
    return c

class ActiveKVTests(unittest.TestCase):
    def test_hbm_control_preserves_existing_baseline(self):
        old=json.loads((ROOT/'examples/default-result.json').read_text());new=simulate(old['config'])
        self.assertEqual(new['metrics'],old['metrics'])
        self.assertEqual(new['timeline'],old['timeline'])
        self.assertEqual([q['token_times_ms'] for q in new['requests']],[q['token_times_ms'] for q in old['requests']])
        self.assertNotIn('active_kv',new)

    def test_exact_active_read_append_and_initialization_geometry(self):
        r=simulate(scenario());a=r['active_kv'];bpt=2*8*128*2*32
        self.assertEqual(a['stats']['attention_read_bytes'],2*bpt*(512+513))
        self.assertEqual(a['stats']['append_write_bytes'],2*bpt*2)
        self.assertEqual(a['stats']['ssd_read_bytes'],a['stats']['attention_read_bytes'])
        self.assertEqual(a['stats']['ssd_write_bytes'],2*bpt*(512+2))
        reads=[e for e in r['timeline'] if e.get('cause')=='active_read' and e['tier']=='ssd']
        self.assertEqual([e['payload_bytes'] for e in reads],[2*bpt*512,2*bpt*513])
        self.assertEqual([e['kv_tokens'] for e in reads],[512,513])
        self.assertTrue(all(e['batch_size']==len(e['request_ids'])==2 for e in reads))
        self.assertEqual(r['storage']['stats'].get('ssd_read_bytes',0),0)
        self.assertFalse(r['storage']['retention_enabled'])

    def test_ssd_service_changes_tpot_and_bandwidth_sensitivity(self):
        c=scenario();ssd=simulate(c);c['storage']['ssd_read_GBps']=.1;slow=simulate(c)
        self.assertGreater(slow['metrics']['tpot_mean_ms'],ssd['metrics']['tpot_mean_ms'])
        self.assertLess(slow['metrics']['throughput_tps'],ssd['metrics']['throughput_tps'])
        dram=simulate(scenario('dram'));hbm=simulate(scenario('hbm'))
        self.assertLess(ssd['metrics']['throughput_tps'],dram['metrics']['throughput_tps'])
        self.assertLess(dram['metrics']['throughput_tps'],hbm['metrics']['throughput_tps'])

    def test_reference_gpu_service_and_serial_io_reconcile_once(self):
        c=scenario();c.update(requests=4);c['prefill']['pp']=2;c['decode']['pp']=2
        r=simulate(c);events={e['event_id']:e for e in r['timeline']}
        for e in events.values():
            if e.get('kind')=='storage_io':
                f=e['service_model'];n=e['full_chunks'];tail=e['tail_bytes']
                self.assertAlmostEqual(e['duration_ms'],n*latency_ms(f,e['chunk_bytes'])+(latency_ms(f,tail) if tail else 0))
            elif 'active_io_event_ids' in e:
                io=sum(events[i]['queue_ms']+events[i]['duration_ms'] for i in e['active_io_event_ids'])
                expected=io+(e['active_group_wait_ms']+e['active_wire_ms'] if e.get('kind')=='active_staging' else e['gpu_service_ms'])
                self.assertAlmostEqual(e['duration_ms'],expected)
                if e.get('shape_id'):
                    shape=r['execution_shapes'][e['shape_id']]
                    reference=sum(o['duration_ms'] for o in shape['operations'])*e['layers_per_stage']+sum(o['duration_ms'] for o in e['endpoint_timing'])+e['head_fallback_ms']
                    self.assertAlmostEqual(e['gpu_service_ms'],reference)
            elif e.get('kind')=='active_handoff':
                self.assertAlmostEqual(e['duration_ms'],e['active_staging_wait_ms']+e['active_wire_ms']+e['active_initialize_ms'])
        for e in events.values():
            previous=events.get(e['previous_resource_event'])
            self.assertAlmostEqual(e['start_ms'],max(e['ready_ms'],previous['end_ms'] if previous else 0))
        self.assertEqual(len(r['requests']),4)

    def test_one_token_output_has_no_active_backing_io(self):
        c=scenario();c['output_tokens']=1;ssd=simulate(c);c['active_kv']['placement']='hbm';hbm=simulate(c)
        self.assertEqual(ssd['metrics'],hbm['metrics'])
        self.assertEqual(ssd['active_kv']['stats'].get('read_bytes',0),0)
        self.assertEqual(ssd['active_kv']['stats'].get('write_bytes',0),0)

    def test_hybrid_zero_and_full_residency(self):
        c=scenario('hbm-dram');c['active_kv']['hbm_resident_GB']=0;zero=simulate(c)
        self.assertEqual(zero['metrics'],simulate(scenario('dram'))['metrics'])
        c['active_kv']['hbm_resident_GB']=1;full=simulate(c)
        self.assertEqual(full['active_kv']['stats']['attention_read_bytes'],0)
        self.assertTrue(all(q['remote_fraction']==0 for q in full['active_kv']['allocations']))
        self.assertGreater(full['metrics']['throughput_tps'],zero['metrics']['throughput_tps'])

    def test_backing_capacity_causes_wait_and_is_released(self):
        c=scenario();c.update(requests=4);c['prefill']['batch_size']=1;c['decode']['batch_size']=1;c['active_kv']['ssd_total_GB']=.08
        r=simulate(c);a=r['active_kv']
        self.assertEqual(len(r['requests']),4)
        self.assertTrue(any(q['admission_ms']>q['admission_ready_ms'] for q in a['allocations']))
        self.assertLessEqual(a['peak_remote_bytes'],a['remote_capacity_bytes']+1e-6)
        self.assertAlmostEqual(a['ledger'][-1]['used_remote_bytes'],0)
        self.assertAlmostEqual(sum(e['remote_bytes']*(1 if e['action']=='grant' else -1) for e in a['ledger']),0)

    def test_prefix_and_active_placements_are_independent_but_share_io(self):
        c=scenario();c['storage'].update(enabled=True,placement='ssd-only',hbm_cache_GB=0,dram_cache_GB=0,ssd_cache_GB=4)
        r=simulate(c)
        self.assertEqual(r['config']['prefill']['batch_size'],2)
        self.assertEqual(r['storage']['placement'],'ssd-only');self.assertEqual(r['active_kv']['placement'],'ssd')
        self.assertGreater(r['storage']['stats']['ssd_write_bytes'],0)
        self.assertGreater(r['active_kv']['stats']['ssd_read_bytes'],0)
        lanes={}
        for e in r['timeline']:
            if e.get('kind')=='storage_io':lanes.setdefault(e['resource'],[]).append(e)
        self.assertEqual({e['scope'] for e in lanes['kv_ssd:0']},{'prefix','active'})
        for lane in lanes.values():
            for left,right in zip(lane,lane[1:]):self.assertLessEqual(left['end_ms'],right['start_ms']+1e-9)
        c['storage']['placement']='hbm-only';c['storage']['hbm_cache_GB']=1;mixed=simulate(c)
        self.assertEqual(mixed['active_kv']['placement'],'ssd')
        self.assertEqual(mixed['storage']['stats'].get('ssd_write_bytes',0),0)
        self.assertGreater(mixed['active_kv']['stats']['ssd_read_bytes'],0)

    def test_device_totals_include_prefix_reservations(self):
        c=scenario();c['storage'].update(enabled=True,placement='ssd-only',ssd_cache_GB=4)
        c['active_kv']['ssd_total_GB']=4.3;r=simulate(c);a=r['active_kv']
        self.assertAlmostEqual(a['remote_capacity_bytes'],.3e9,places=5)
        self.assertLessEqual(a['peak_remote_bytes']+a['partition']['prefix_ssd_bytes'],a['partition']['ssd_total_bytes'])
        c['active_kv']['ssd_total_GB']=4.01
        with self.assertRaisesRegex(ValueError,'microbatch cannot fit'):simulate(c)
        c['active_kv']['ssd_total_GB']=3
        with self.assertRaisesRegex(ValueError,'capacity must cover'):simulate(c)

    def test_colocated_staging_hbm_and_energy_accounting(self):
        c=scenario('hbm-ssd');c.update(separated=False,requests=4);c['active_kv']['hbm_resident_GB']=.05
        c['prefill'].update(tp=2,pp=2)
        r=simulate(c)
        self.assertEqual(len(r['requests']),4)
        for mm in r['memory'].values():self.assertLessEqual(mm['total_GB'],mm['capacity_GB']+1e-8)
        self.assertGreater(r['memory']['prefill']['staging_GB'],0)
        self.assertTrue(any(e.get('kind')=='active_staging' for e in r['timeline']))
        self.assertLessEqual(r['metrics']['gpu_utilization'],1+1e-8)
        for q in r['accounting']['energy_terms']:
            weighted=sum(r['accounting']['compute_busy_ms'].values())*r['config']['prefill']['tp']*r['config']['prefill']['ep']
            self.assertAlmostEqual(q['weighted_busy_ms'],weighted)
        energy=sum(q['idle_watts']*q['gpus']*r['metrics']['elapsed_ms']/1000+q['dynamic_j'] for q in r['accounting']['energy_terms'])
        self.assertAlmostEqual(energy,r['metrics']['energy_j'])

    def test_profiles_scope_and_prefill_hbm_constraint(self):
        c=scenario();c['storage']['profile']='local-mooncake-path';r=simulate(c)
        self.assertTrue(any(e.get('scope')=='active' and e.get('profile')=='local-mooncake-path' for e in r['timeline']))
        self.assertIn('Unvalidated',r['active_kv']['calibration'])
        c=scenario();c.update(input_tokens=70000,requests=1,output_tokens=2);c['prefill']['hardware']['main_memory_GB']=20
        with self.assertRaisesRegex(ValueError,'Prefill requires'):simulate(c)

    def test_invalid_staging_and_replica_settings_are_rejected(self):
        c=scenario();c['active_kv']['staging_GB']=.00001
        with self.assertRaisesRegex(ValueError,'staging space must fit'):simulate(c)
        c=scenario();c['decode']['replicas']=2
        with self.assertRaisesRegex(ValueError,'one replica'):simulate(c)
        c=scenario();c['active_kv']['placement']='unknown'
        with self.assertRaisesRegex(ValueError,'Unknown active'):simulate(c)

if __name__=='__main__':unittest.main()
