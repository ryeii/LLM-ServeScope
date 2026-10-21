import copy,json,math,unittest
from pathlib import Path
import simpy
from backend.model import DEFAULT,simulate,parameters,resolve_model,validate
from backend.storage import CacheTiers,fit_dataset,latency_ms,hit_prediction
ROOT=Path(__file__).resolve().parents[1]

def scenario():
 c=copy.deepcopy(DEFAULT);c.update(model_id='llama8',requests=8,input_tokens=1024,output_tokens=2,arrival_rate=2)
 for phase in ['prefill','decode']:c[phase].update(batch_size=1,replicas=1,tp=1)
 c['storage'].update(enabled=True,hbm_cache_GB=0,dram_cache_GB=0,ssd_cache_GB=4)
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

 def test_invalid_combinations_fail_explicitly(self):
  c=scenario();c['prefill']['batch_size']=2
  with self.assertRaisesRegex(ValueError,'batch size 1'):simulate(c)
  c=scenario();c['storage']['workload']=[dict(timestamp=10,input_length=4,output_length=1,hash_ids=[1,2])]
  with self.assertRaisesRegex(ValueError,'prefix hash'):simulate(c)

if __name__=='__main__':unittest.main()
