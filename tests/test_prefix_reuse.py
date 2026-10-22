"""Check evidence semantics, GPU shapes and rejection of retired paths."""
import copy
import hashlib
import json
import unittest
from pathlib import Path
from backend.model import DEFAULT,simulate,validate,catalog
from backend.prefix_reuse import evidence
from integration.analyze_prefix_reuse import analyze
ROOT=Path(__file__).resolve().parents[1]


class PrefixReuseTests(unittest.TestCase):
    def test_tied_arrivals_cannot_reuse_one_another_and_last_token_is_recomputed(self):
        rows=[dict(timestamp=t,input_length=1024,hash_ids=ids) for t,ids in [(0,[1,2]),(0,[1,2]),(1,[1,2]),(2,[3,2])]]
        result=analyze(rows)
        self.assertEqual(result['reusable_tokens'],1023)
        self.assertEqual(result['requests_with_reuse'],1)
        self.assertEqual(result['fraction'],1023/4096)

    def test_partial_blocks_and_weighted_denominator(self):
        rows=[dict(timestamp=t,input_length=n,hash_ids=ids) for t,n,ids in [(0,1024,[1,2]),(1,1536,[1,3,4]),(2,600,[1,9])]]
        r=analyze(rows)
        self.assertEqual(r['reusable_tokens'],1024)
        self.assertEqual(r['fraction'],1024/(1024+1536+600))

    def test_all_published_sources_match_exact_counts_and_hashes(self):
        d=evidence()
        self.assertEqual(d['analysis_sha256'],hashlib.sha256((ROOT/'integration/analyze_prefix_reuse.py').read_bytes()).hexdigest())
        for row in d['workloads'].values():
            raw=(ROOT/'third_party/mooncake-traces'/row['source_file']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            computed=analyze([json.loads(line) for line in raw.splitlines()])
            for key,value in computed.items():self.assertEqual(value,row[key],key)

    def test_default_has_only_hbm_no_cache_policy_or_device_events(self):
        r=simulate(DEFAULT)
        self.assertEqual(r['prefix_reuse']['source'],'none')
        self.assertEqual(r['prefix_reuse']['reused_tokens_per_request'],0)
        for key in ['storage','active_kv']:self.assertNotIn(key,r);self.assertNotIn(key,r['config'])
        self.assertTrue(all(e['phase'] in ['prefill','decode','transfer'] for e in r['timeline']))
        self.assertNotIn('storage_layouts',catalog())

    def test_reuse_reduces_queries_but_preserves_full_kv_context_decode_and_handoff(self):
        c=copy.deepcopy(DEFAULT);c.update(requests=4,output_tokens=3)
        a=simulate(c);c['prefix_reuse_source']='conversation';b=simulate(c)
        reused=b['prefix_reuse']['reused_tokens_per_request']
        self.assertGreater(reused,0)
        self.assertEqual(a['memory'],b['memory'])
        for r in [a,b]:
            q=r['requests'][0]
            pre=[e for e in r['timeline'] if e['phase']=='prefill' and e.get('shape_id') and 0 in e['request_ids']]
            self.assertEqual(sum(e['tokens'] for e in pre),q['computed_prefill_tokens'])
            self.assertEqual(pre[-1]['context'],c['input_tokens'])
            self.assertEqual(len(q['token_times_ms']),3)
        self.assertGreater(b['timeline'][0]['context'],b['timeline'][0]['tokens'])
        payload=lambda r:[e['payload_bytes'] for e in r['timeline'] if e['phase']=='transfer']
        self.assertEqual(payload(a),payload(b))
        decode=lambda r:sorted(e['context'] for e in r['timeline'] if e['phase']=='decode')
        self.assertEqual(decode(a),decode(b))
        self.assertLess(b['metrics']['ttft_mean_ms'],a['metrics']['ttft_mean_ms'])

    def test_removed_settings_fail_and_gpu_only_legacy_defaults_to_no_reuse(self):
        for key in ['storage','active_kv']:
            c=copy.deepcopy(DEFAULT);c[key]={}
            with self.assertRaisesRegex(ValueError,'retired KV placement'):validate(c)
        c=copy.deepcopy(DEFAULT);c.pop('prefix_reuse_source')
        self.assertEqual(validate(c)['prefix_reuse_source'],'none')
        c['prefix_reuse_source']='invented'
        with self.assertRaisesRegex(ValueError,'published prefix-reuse'):validate(c)
