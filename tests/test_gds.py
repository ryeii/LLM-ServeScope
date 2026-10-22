import copy
import json
import unittest
from pathlib import Path

from backend.model import DEFAULT, simulate, validate
from backend.kv_transfer import direct_service

ROOT = Path(__file__).resolve().parents[1]


def scenario():
    c = copy.deepcopy(DEFAULT)
    c.update(model_id='llama8', requests=8, input_tokens=1024,
             output_tokens=3, arrival_rate=2)
    for phase in ('prefill', 'decode'):
        c[phase].update(tp=1, pp=1, ep=1, replicas=1, batch_size=1)
    c['storage'].update(placement='ssd-only', hbm_cache_GB=0,
                        dram_cache_GB=0, publication_mode='blocking')
    return c


class GDSTests(unittest.TestCase):
    def test_new_default_is_resident_active_kv_async_tiered_cache_and_gds(self):
        r = simulate(DEFAULT)
        self.assertEqual(r['config']['active_kv']['placement'], 'hbm')
        self.assertEqual(r['storage']['placement'], 'dynamo')
        self.assertEqual(r['storage']['publication_mode'], 'async')
        self.assertEqual(r['storage']['ssd_transfer_mode'], 'gds')
        self.assertEqual(r['schema_version'], 'servescope.behavioral.v6')
        self.assertEqual(len(r['provenance']['kv_transfer_model_sha256']), 64)
        self.assertNotIn('active_kv', r)
        self.assertFalse(any(e.get('cause') == 'active_read' for e in r['timeline']))

    def test_direct_path_counts_storage_and_link_once_and_no_host_copy(self):
        c = scenario()
        direct = simulate(c)
        events = [e for e in direct['timeline'] if e.get('kind') == 'storage_io']
        self.assertTrue(events)
        self.assertEqual({e['transfer_path'] for e in events}, {'gds'})
        for e in events:
            g = e['gds']
            self.assertAlmostEqual(e['duration_ms'], g['startup_ms'] + max(g['storage_ms'], g['link_ms']))
        c['storage']['ssd_transfer_mode'] = 'host-staged'
        staged = simulate(c)
        self.assertTrue(any(e.get('tier') == 'pcie' for e in staged['timeline']))
        self.assertEqual(direct['storage']['stats']['ssd_read_bytes'], staged['storage']['stats']['ssd_read_bytes'])
        self.assertLess(direct['requests'][4]['ttft_ms'], staged['requests'][4]['ttft_ms'])

    def test_gds_link_can_be_the_bottleneck_and_nonmanual_profile_is_rejected(self):
        c = scenario()
        fast = simulate(c)
        c['storage']['gds_GBps'] = .1
        slow = simulate(c)
        self.assertGreater(slow['requests'][4]['ttft_ms'], fast['requests'][4]['ttft_ms'])
        self.assertAlmostEqual(direct_service(1, 1e6, .1, 10)['duration_ms'], 10.01)
        c['storage']['profile'] = 'unsupported-device-fit'
        with self.assertRaisesRegex(ValueError, 'Only configured storage timing'):
            validate(c)

    def test_async_publication_preserves_source_until_both_copies_and_handoff_finish(self):
        c = scenario()
        c.update(requests=4, output_tokens=1, arrival_rate=10000)
        c['storage'].update(publication_mode='async', ssd_write_GBps=.01,
                            max_inflight_publications=2)
        r = simulate(c)
        for q in r['requests']:
            k = q['cache']
            self.assertEqual(k['publication_end_ms'], k['gpu_prefill_end_ms'])
            self.assertGreater(k['background_publication_end_ms'], q['first_token_ms'])
            self.assertGreaterEqual(k['prefill_release_ms'], k['background_publication_end_ms'])
            self.assertGreaterEqual(k['prefill_release_ms'], k['handoff_end_ms'])
        self.assertTrue(all(e['background'] for e in r['timeline'] if e.get('kind') == 'storage_io'))
        self.assertTrue(all(q['cache']['matched_blocks'] == 0 for q in r['requests']))
        self.assertEqual(r['metrics']['elapsed_ms'], max(q['finish_ms'] for q in r['requests']))
        ledger = r['storage']['active_memory_ledger']
        self.assertEqual(sum(e['bytes'] if e['action'] == 'grant' else -e['bytes'] for e in ledger), 0)
        spans = r['storage']['publications']
        for t in {p['start_ms'] for p in spans}:
            self.assertLessEqual(sum(p['start_ms'] <= t < p['end_ms'] for p in spans), 2)

    def test_dynamo_filter_and_ready_copies_restore_into_hbm(self):
        c = scenario()
        c.update(requests=3, arrival_rate=.1)
        c['storage'].update(placement='dynamo', disk_min_uses=2,
                            workload=[dict(timestamp=i*10000,input_length=1024,output_length=3,hash_ids=[1,2]) for i in range(3)])
        r = simulate(c)
        self.assertEqual(r['storage']['stats']['filtered_disk_blocks'], 2)
        self.assertEqual(r['storage']['stats']['ssd_write_bytes'], 2*r['storage']['block_bytes'])
        # With HBM/DRAM disabled, the first reference is deliberately not saved.
        self.assertEqual([q['cache']['reused_tokens'] for q in r['requests']], [0,0,1023])
        self.assertFalse(any(e.get('cause') == 'active_read' for e in r['timeline']))

    def test_active_ssd_gds_has_no_host_relay_or_duplicate_pcie(self):
        c = scenario()
        c['storage'].update(enabled=False)
        c['active_kv'].update(placement='ssd', dram_total_GB=0)
        r = simulate(c)
        self.assertEqual(r['active_kv']['partition']['relay_dram_bytes'], 0)
        self.assertFalse(any(e.get('tier') == 'pcie' for e in r['timeline']))
        self.assertGreater(r['active_kv']['stats']['gds_read_bytes'], 0)

    def test_legacy_exports_retain_timing_and_staged_blocking_semantics(self):
        for name in ('default-result', 'storage-tier-result', 'active-kv-result'):
            old = json.loads((ROOT/'examples'/f'{name}.json').read_text())
            r = simulate(old['config'])
            self.assertEqual(r['metrics'], old['metrics'])
            self.assertEqual(r['config']['storage']['ssd_transfer_mode'], 'host-staged')
            self.assertEqual(r['config']['storage']['publication_mode'], 'blocking')


if __name__ == '__main__':
    unittest.main()
