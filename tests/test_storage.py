import copy,json,math,unittest
from pathlib import Path
import simpy
from backend.model import DEFAULT,simulate,parameters,resolve_model,validate
from backend.storage import CacheTiers,fit_dataset,latency_ms,hit_prediction,CACHE_LAYOUTS
ROOT=Path(__file__).resolve().parents[1]

def scenario():
 c=copy.deepcopy(DEFAULT);c.update(model_id='llama8',requests=8,input_tokens=1024,output_tokens=2,arrival_rate=2)
 for phase in ['prefill','decode']:c[phase].update(batch_size=1,replicas=1,tp=1)
 c['storage'].update(enabled=True,placement='inclusive',hbm_cache_GB=0,dram_cache_GB=0,ssd_cache_GB=4,
                     ssd_transfer_mode='host-staged',publication_mode='blocking',max_inflight_publications=128)
 return c

class StorageTests(unittest.TestCase):
 def test_legacy_metrics_unchanged(self):
  for name in ['default-result','calibration-reference-result','vidur-reference-result']:
   old=json.loads((ROOT/'examples'/f'{name}.json').read_text());self.assertEqual(old['metrics'],simulate(old['config'])['metrics'])

 def test_reuse_changes_executed_shape_not_only_reported_tokens(self):
  r=simulate(scenario());self.assertEqual([q['cache']['reused_tokens'] for q in r['requests']],[0]*4+[1023]*4)
  for q in r['requests'][4:]:
   events=[e for e in r['timeline'] if e['phase']=='prefill' and q['id'] in e['request_ids']]
   self.assertEqual(len(events),1);shape=r['execution_shapes'][events[0]['shape_id']]
   self.assertEqual((shape['tokens'],shape['context']),(1,1024));self.assertEqual(q['cache']['computed_tokens'],1)
  self.assertEqual(r['workload_totals']['computed_prefill_tokens'],4*1024+4)

 def test_no_instant_hits_or_reads_of_unfinished_publications(self):
  c=scenario();c['storage']['workload']=[dict(timestamp=0,input_length=1024,output_length=1,hash_ids=[0,1]) for _ in range(4)]
  r=simulate(c);self.assertTrue(all(q['cache']['matched_blocks']==0 for q in r['requests']))
  self.assertTrue(any(e['operation']=='write' for e in r['timeline'] if e.get('kind')=='storage_io'))

 def test_disable_all_tiers_means_no_cache_io(self):
  c=scenario();c['storage'].update(hbm_cache_GB=0,dram_cache_GB=0,ssd_cache_GB=0)
  r=simulate(c);self.assertEqual(r['storage']['stats'].get('reused_tokens',0),0);self.assertFalse(any(e.get('kind')=='storage_io' for e in r['timeline']))

 def test_ssd_latency_is_on_request_path_and_explicit_fifo(self):
  c=scenario();c['storage']['ssd_read_latency_us']=100
  fast=simulate(c);c['storage']['ssd_read_latency_us']=10000;slow=simulate(c)
  self.assertGreater(slow['requests'][4]['ttft_ms'],fast['requests'][4]['ttft_ms'])
  lanes={}
  for e in slow['timeline']:
   if e.get('kind')!='storage_io':continue
   lanes.setdefault(e['resource'],[]).append(e)
   self.assertAlmostEqual(e['queue_ms'],e['start_ms']-e['ready_ms'])
   if e['tier']=='ssd':
    n=e['payload_bytes'];chunk=e['chunk_bytes'];full=int(n//chunk);tail=n-full*chunk;fit=e['service_model']
    self.assertAlmostEqual(e['duration_ms'],full*latency_ms(fit,chunk)+(latency_ms(fit,tail) if tail else 0))
  for lane in lanes.values():
   for a,b in zip(lane,lane[1:]):self.assertLessEqual(a['end_ms'],b['start_ms']+1e-9)

 def test_variable_workload_arrivals_and_tokens(self):
  c=scenario();c['storage']['workload']=[dict(timestamp=5000,input_length=1024,output_length=1,hash_ids=[1,2]),dict(timestamp=15000,input_length=1536,output_length=3,hash_ids=[1,2,3])]
  r=simulate(c);self.assertEqual([q['arrival_ms'] for q in r['requests']],[0,10000]);self.assertEqual(r['workload_totals']['output_tokens'],4)
  self.assertAlmostEqual(r['metrics']['throughput_tps'],4/(r['metrics']['elapsed_ms']/1000));self.assertEqual(r['requests'][1]['cache']['reused_tokens'],1024)

 def test_pinned_restore_source_is_not_evicted(self):
  c=validate(scenario());env=simpy.Environment();t=CacheTiers(env,c,resolve_model(c),lambda e,r:None);t.capacity['ssd']=1
  t._put('ssd','a',0);t.pins['ssd']['a']=1;t._put('ssd','b',1)
  self.assertIn('a',t.caches['ssd']);self.assertNotIn('b',t.caches['ssd']);self.assertEqual(t.drops['ssd'],1)

 def test_active_capacity_causes_wait_and_never_exceeds_budget(self):
  c=scenario();c.update(requests=4,arrival_rate=10000);c['storage'].update(ssd_cache_GB=0)
  m=resolve_model(c);shared,expert=parameters(m);weight=(shared+expert)*2;activation=512*m['hidden']*2*8
  kv=2*(1024+2-1)*m['kv_heads']*m['head_dim']*2*m['layers']
  for ph in ['prefill','decode']:c[ph]['hardware']['main_memory_GB']=(weight+activation+kv*1.1)/1e9
  r=simulate(c);self.assertTrue(any(e.get('kind')=='kv_memory_wait' for e in r['timeline']))
  for mm in r['memory'].values():self.assertLessEqual(mm['active_kv_GB']*1e9,mm['active_kv_budget_bytes']+1e-5)

 def test_fitted_functions_monotone_bounded_and_bandwidth_consistent(self):
  data=fit_dataset()
  for f in data['io'].values():
   times=[latency_ms(f,x) for x in [16384,65536,262144,1048576,2097152]];self.assertEqual(times,sorted(times));self.assertTrue(all(x>0 for x in times))
  for f in data['hit'].values():
   values=[hit_prediction(f,c,1000) for c in [0,1,10,100,1000,10000]];self.assertEqual(values,sorted(values));self.assertTrue(all(0<=x<=1 for x in values))
  records=json.loads((ROOT/'docs/storage-2026-10-20/io-observations.json').read_text())['observations']
  self.assertEqual(len(records),768)
  for row in records:self.assertAlmostEqual(row['bandwidth_MBps'],row['bytes']/row['latency_ms']/1000)

 def test_archived_cv_models_reconstruct_held_out_predictions(self):
  validation=json.loads((ROOT/'docs/storage-2026-10-20/validation.json').read_text());predictions=json.loads((ROOT/'docs/storage-2026-10-20/held-out-predictions.json').read_text())
  for op,rows in predictions['io'].items():
   for row in rows:
    f=validation['io'][op]['outer_models'][str(row['bytes'])]
    self.assertNotIn(row['bytes'],f['training_sizes'])
    self.assertAlmostEqual(latency_ms(f,row['bytes']),row['predicted_ms'])
  for source,rows in predictions['hit'].items():
   for row in rows:
    f=validation['hit'][source]['outer_models'][str(row['window'])]
    self.assertNotIn(row['window'],f['training_regions'])
    self.assertAlmostEqual(hit_prediction(f,row['capacity_blocks'],row['working_set_blocks'],row['max_hit_fraction']),row['predicted_hit'])

 def test_source_and_fit_integrity(self):
  import hashlib
  fit=fit_dataset()
  for field,path in [('training_script_sha256','integration/storage/study.py'),('runtime_sha256','backend/storage.py')]:
   self.assertEqual(fit[field],hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
  for kind in ['io','hit']:
   path=ROOT/'docs/storage-2026-10-20'/f'{kind}-observations.json';self.assertEqual(fit['sources'][kind+'_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
  for folder in ['mooncake-storage','mooncake-traces']:
   base=ROOT/'third_party'/folder;m=json.loads((base/'manifest.json').read_text())
   for file in m['files']:
    path=base/('source/'+file['path'] if folder=='mooncake-storage' else file['file'])
    self.assertEqual(file['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())

 def test_parallel_shards_and_colocated_pools_preserve_payload_and_completion(self):
  for separated in [True,False]:
   c=scenario();c.update(separated=separated,arrival_rate=1000)
   for phase in ['prefill','decode']:c[phase].update(tp=2,pp=2,ep=1)
   r=simulate(c)
   self.assertEqual(len(r['requests']),8)
   self.assertEqual(r['workload_totals']['output_tokens'],16)
   self.assertEqual(r['storage']['block_bytes'],r['storage']['per_gpu_block_bytes']*4)
   self.assertTrue(any(e.get('kind')=='pipeline_transfer' for e in r['timeline']))
   self.assertEqual(any(e['phase']=='transfer' for e in r['timeline']),separated)
   ledger=r['storage']['active_memory_ledger']
   self.assertEqual(sum(e['bytes'] if e['action']=='grant' else -e['bytes'] for e in ledger),0)
   for entry in ledger:self.assertLessEqual(entry['used_bytes'],entry['capacity_bytes']+1e-5)
   for q in r['requests']:self.assertEqual(q['cache']['computed_tokens']+q['cache']['reused_tokens'],q['input_tokens'])

 def test_single_tier_layouts_mask_unused_capacity_and_io(self):
  for tier in ['hbm','dram','ssd']:
   c=scenario();c['storage'].update(placement=tier+'-only',hbm_cache_GB=4,dram_cache_GB=4,ssd_cache_GB=4)
   r=simulate(c)
   self.assertEqual(r['storage']['placement'],tier+'-only')
   for other in {'hbm','dram','ssd'}-{tier}:self.assertEqual(r['storage']['capacity_blocks'][other],0)
   paths={e['tier'] for e in r['timeline'] if e.get('kind')=='storage_io'}
   self.assertEqual(paths,{'hbm'} if tier=='hbm' else {'pcie'} if tier=='dram' else {'ssd','pcie'})
   self.assertEqual(r['requests'][4]['cache']['hit_tiers'],{tier:2})
   if tier!='hbm':self.assertEqual(r['memory']['prefill']['retained_cache_reservation_GB'],0)
  c=scenario();c['storage'].update(placement='dram-only',hbm_cache_GB=100000,dram_cache_GB=4)
  self.assertEqual(simulate(c)['memory']['prefill']['retained_cache_reservation_GB'],0)

 def test_all_directed_spill_layouts_move_only_evicted_blocks(self):
  for placement,layout in CACHE_LAYOUTS.items():
   if layout['kind']!='spill' or len(layout['tiers'])!=2:continue
   c=validate(scenario());c['storage']['placement']=placement
   env=simpy.Environment();t=CacheTiers(env,c,resolve_model(c),lambda e,r:None)
   primary,secondary=layout['tiers'];t.capacity[primary]=1;t.capacity[secondary]=2
   a=dict(input_length=512,hash_ids=[100]);b=dict(input_length=512,hash_ids=[200]);result={}
   def run():
    yield from t.publish(a,0)
    self.assertEqual(t.stats['spill_blocks'],0)
    self.assertEqual(len(t.caches[secondary]),0)
    yield from t.publish(b,1)
    self.assertEqual(list(t.caches[primary]),t.keys(b))
    self.assertEqual(list(t.caches[secondary]),t.keys(a))
    result.update((yield from t.restore(a,2)))
   env.process(run());env.run()
   self.assertEqual(t.stats['spill_blocks'],1)
   self.assertEqual(t.stats[primary+'_to_'+secondary+'_spill_blocks'],1)
   self.assertEqual(result['hit_tiers'],{secondary:1})
   self.assertEqual(result['reused_tokens'],511)
   self.assertTrue(any(e['cause']=='spill' for e in t.events))
   self.assertFalse(set(t.caches[primary]) & set(t.caches[secondary]))

 def test_three_tier_cascade_discards_only_after_last_tier_fills(self):
  c=validate(scenario());c['storage']['placement']='hbm-dram-ssd'
  env=simpy.Environment();t=CacheTiers(env,c,resolve_model(c),lambda e,r:None)
  t.capacity=dict(hbm=1,dram=1,ssd=1)
  rows=[dict(input_length=512,hash_ids=[i]) for i in range(4)]
  def run():
   for i,row in enumerate(rows):yield from t.publish(row,i)
  env.process(run());env.run()
  for tier,row in zip(['hbm','dram','ssd'],rows[:0:-1]):self.assertEqual(list(t.caches[tier]),t.keys(row))
  self.assertEqual(t.stats['discarded_blocks'],1)
  self.assertEqual(t.stats['hbm_to_dram_spill_blocks'],3)
  self.assertEqual(t.stats['dram_to_ssd_spill_blocks'],2)
  self.assertTrue(all(len(cache)<=t.capacity[tier] for tier,cache in t.caches.items()))

 def test_spill_skips_pinned_primary_without_corrupting_restore_source(self):
  c=validate(scenario());c['storage']['placement']='hbm-ssd'
  env=simpy.Environment();t=CacheTiers(env,c,resolve_model(c),lambda e,r:None)
  t.capacity.update(hbm=1,ssd=1)
  a=dict(input_length=512,hash_ids=[10]);b=dict(input_length=512,hash_ids=[20])
  def run():
   yield from t.publish(a,0)
   key=t.keys(a)[0];t.pins['hbm'][key]=1
   yield from t.publish(b,1)
   self.assertIn(key,t.caches['hbm'])
   self.assertIn(t.keys(b)[0],t.caches['ssd'])
  env.process(run());env.run();self.assertEqual(t.drops['hbm'],1)

 def test_no_retention_keeps_tier_scheduler_without_cache_io(self):
  c=scenario();c['storage']['placement']='none';r=simulate(c)
  self.assertEqual(r['storage']['capacity_blocks'],dict(hbm=0,dram=0,ssd=0))
  self.assertEqual(r['workload_totals']['computed_prefill_tokens'],8192)
  self.assertFalse(any(e.get('kind')=='storage_io' for e in r['timeline']))

 def test_no_retention_preserves_original_batched_schedule_and_all_metrics(self):
  for prefill,decode,separated in [(4,4,True),(4,1,True),(2,2,False),(4,2,False),(8,4,True)]:
   c=copy.deepcopy(DEFAULT);c['prefill']['batch_size']=prefill;c['decode']['batch_size']=decode;c['separated']=separated
   c['storage']['enabled']=False
   baseline=simulate(c);c['storage'].update(enabled=True,placement='none');r=simulate(c)
   self.assertEqual(r['metrics'],baseline['metrics'])
   self.assertEqual(r['traffic'],baseline['traffic'])
   self.assertEqual([(q['arrival_ms'],q['batch_ready_ms'],q['token_times_ms']) for q in r['requests']],[(q['arrival_ms'],q['batch_ready_ms'],q['token_times_ms']) for q in baseline['requests']])
   self.assertEqual(len(r['requests']),c['requests'])
   self.assertEqual(sum(e['bytes'] if e['action']=='grant' else -e['bytes'] for e in r['storage']['active_memory_ledger']),0)

 def test_batched_prefix_uses_common_floor_and_preserves_actual_context(self):
  c=scenario();c['prefill']['batch_size']=2;c['decode']['batch_size']=2;c['storage'].update(placement='hbm-only',hbm_cache_GB=2,ssd_cache_GB=0)
  c['storage']['workload']=[dict(timestamp=t,input_length=1024,output_length=2,hash_ids=[h,h+1]) for t,h in [(0,10),(1,20),(5000,10),(5001,30),(10000,10),(10001,20)]]
  r=simulate(c)
  self.assertEqual(r['requests'][2]['cache']['available_reused_tokens'],1023)
  self.assertEqual(r['requests'][2]['cache']['reused_tokens'],0)
  self.assertEqual(r['requests'][3]['cache']['reused_tokens'],0)
  self.assertEqual(r['storage']['stats']['batch_recomputed_restored_tokens'],1023)
  for q in r['requests'][4:]:
   self.assertEqual(q['cache']['reused_tokens'],1023)
   e=next(e for e in r['timeline'] if e['phase']=='prefill' and q['id'] in e['request_ids'])
   shape=r['execution_shapes'][e['shape_id']]
   self.assertEqual((shape['batch'],shape['tokens'],shape['context']),(2,1,1024))
  self.assertEqual(r['requests'][0]['batch_ready_ms'],1)
  self.assertEqual(r['workload_totals']['computed_prefill_tokens'],4*1024+2)

 def test_atomic_microbatch_must_fit_hbm_before_admission(self):
  c=scenario();c['prefill']['batch_size']=2;c['storage'].update(placement='none')
  m=resolve_model(c);shared,expert=parameters(m);weight=(shared+expert)*2
  activation=2*512*m['hidden']*2*8
  kv=2*(1024+2-1)*m['kv_heads']*m['head_dim']*2*m['layers']
  c['prefill']['hardware']['main_memory_GB']=(weight+activation+kv*1.1)/1e9
  with self.assertRaisesRegex(ValueError,'Prefill requires'):simulate(c)

 def test_prefix_ssd_does_not_add_per_token_decode_io(self):
  c=scenario();c['storage']['placement']='ssd-only';c['output_tokens']=8;r=simulate(c)
  self.assertEqual(r['storage']['active_kv_location'],'hbm')
  self.assertFalse(r['storage']['active_kv_offload'])
  for e in r['timeline']:
   if e.get('kind')=='storage_io':self.assertIn(e['cause'],['restore','publish','spill'])

 def test_turning_retention_off_preserves_imported_arrivals_and_lengths(self):
  c=scenario();c['storage'].update(placement='none',workload=[dict(timestamp=5000,input_length=1024,output_length=1,hash_ids=[1,2]),dict(timestamp=15000,input_length=1536,output_length=3,hash_ids=[1,2,3])])
  control=simulate(c);c['storage'].update(enabled=False,placement='ssd-only');off=simulate(c)
  self.assertEqual(off['metrics'],control['metrics'])
  self.assertEqual(off['workload_totals'],control['workload_totals'])
  self.assertEqual(off['storage']['capacity_blocks'],dict(hbm=0,dram=0,ssd=0))
  self.assertFalse(off['storage']['retention_enabled'])
  self.assertEqual([q['arrival_ms'] for q in off['requests']],[0,10000])
  self.assertEqual([q['output_tokens'] for q in off['requests']],[1,3])

 def test_invalid_combinations_fail_explicitly(self):
  c=scenario();c['storage']['placement']='unknown'
  with self.assertRaisesRegex(ValueError,'layout'):simulate(c)
  c=scenario();c['prefill']['replicas']=2
  with self.assertRaisesRegex(ValueError,'one replica'):simulate(c)
  c=scenario();c['prefill']['batch_size']=2;c['storage']['workload']=[dict(timestamp=0,input_length=512,output_length=1,hash_ids=[1]),dict(timestamp=1,input_length=1024,output_length=1,hash_ids=[1,2])]
  with self.assertRaisesRegex(ValueError,'Variable-length.*batch size 1'):simulate(c)
  c=scenario();c['storage']['workload']=[dict(timestamp=10,input_length=4,output_length=1,hash_ids=[1,2])]
  with self.assertRaisesRegex(ValueError,'prefix hash'):simulate(c)

if __name__=='__main__':unittest.main()
