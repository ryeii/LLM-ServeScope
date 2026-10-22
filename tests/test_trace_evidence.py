"""Published workload evidence must not become fabricated device measurements."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from backend.model import catalog, simulate

ROOT = Path(__file__).resolve().parents[1]


class TraceEvidenceTests(unittest.TestCase):
    def test_published_example_preserves_original_arrivals_lengths_and_hashes(self):
        config = json.loads((ROOT/'examples/mooncake-conversation-scenario.json').read_text())
        raw = (ROOT/'third_party/mooncake-traces/conversation_trace.jsonl').read_bytes()
        original = [json.loads(line) for line in raw.splitlines()[:4]]
        self.assertEqual(config['storage']['source_file_sha256'],hashlib.sha256(raw).hexdigest())
        for index,(row,source) in enumerate(zip(config['storage']['workload'],original)):
            self.assertEqual({key:row[key] for key in source},source)
            self.assertEqual(row['source_line'],index+1)
        result = simulate(config)
        self.assertEqual(result['storage']['timing_evidence'],'configured-assumptions-not-request-trace-measurements')
        for row,source in zip(result['requests'],original):
            self.assertEqual(row['input_tokens'],source['input_length'])
            self.assertEqual(row['output_tokens'],source['output_length'])
            self.assertEqual(row['arrival_ms'],source['timestamp']-original[0]['timestamp'])
        self.assertEqual(result['workload_totals']['output_tokens'],sum(r['output_length'] for r in original))

    def test_trace_reuse_forecast_does_not_manufacture_runtime_hits_or_timing(self):
        config = json.loads((ROOT/'examples/mooncake-conversation-scenario.json').read_text())
        config['storage']['workload'] = [dict(timestamp=i*10000,input_length=1024,output_length=2,hash_ids=[1,2]) for i in range(3)]
        with patch('backend.storage.hit_prediction',return_value=0):
            a = simulate(config)
        with patch('backend.storage.hit_prediction',return_value=1):
            b = simulate(config)
        self.assertNotEqual(a['storage']['hit_forecast']['prediction'],b['storage']['hit_forecast']['prediction'])
        self.assertEqual(a['metrics'],b['metrics'])
        self.assertEqual(a['storage']['stats'],b['storage']['stats'])

    def test_runtime_catalog_contains_only_trace_hit_fits_and_gpu_fits(self):
        c = catalog()
        self.assertNotIn('storage_profiles',c)
        self.assertEqual(set(c['prefix_hit_profiles']['hit']),{'conversation','toolagent','synthetic'})
        self.assertNotIn('io',c['prefix_hit_profiles'])
        self.assertEqual({p.name for p in (ROOT/'backend').glob('*fit.json')},{'gpu-fit.json','prefix-hit-fit.json'})


if __name__=='__main__':
    unittest.main()
