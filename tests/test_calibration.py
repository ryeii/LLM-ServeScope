"""A single fitted GPU function, hardware constraints, provenance and validation."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from backend.calibration import dataset, decode_dataset, prediction, timing, work_features
from backend.gpu_functions import evaluate
from backend.catalog import HARDWARE, MODELS
from backend.model import DEFAULT, phase_ops, simulate, validate

ROOT=Path(__file__).resolve().parents[1]


class FittedGpuTests(unittest.TestCase):
    def config(self):
        c=copy.deepcopy(DEFAULT)
        c.update(model_id="llama8",requests=4,input_tokens=128,output_tokens=8,
                 arrival_rate=10000,batch_size=4,prefill_chunk=128,separated=False)
        c["prefill"].update(hardware_id="rtx6000",hardware=copy.deepcopy(HARDWARE["rtx6000"]),tp=1)
        return validate(c)

    def test_coefficient_integrity_and_source_digests(self):
        raw=(ROOT/"backend/gpu-fit.json").read_bytes()
        d=decode_dataset(raw)
        self.assertEqual(d["schema_version"],"servescope.gpu-fit.v2")
        self.assertEqual(d["training_observations"],599451)
        for cohort in d["cohorts"].values():
            for model in cohort["models"].values():
                numbers=model["coefficients"] if model["family"]!="F9" else sum(model["branches"].values(),[])
                self.assertTrue(all(x>=0 for x in numbers))
        self.assertEqual(d["runtime_sha256"],hashlib.sha256((ROOT/"backend/calibration.py").read_bytes()).hexdigest())
        changed=json.loads(raw);changed["cohorts"]["kaist"]["models"]["qkv_proj"]["coefficients"][0]+=1
        with self.assertRaisesRegex(ValueError,"checksum"):
            decode_dataset(json.dumps(changed).encode())
        # Runtime coefficients/coverage carry no timing observations or table grids.
        self.assertNotIn("profiles",d)
        self.assertNotIn("time_us",raw.decode())
        self.assertNotIn("latency_ns",raw.decode())

    def test_profiled_and_custom_identity_use_same_function(self):
        c=self.config()
        profiled=timing("qkv_proj",c,"decode",4,1,129)
        c["decode"]["hardware_id"]="custom"
        custom=timing("qkv_proj",c,"decode",4,1,129)
        self.assertEqual(profiled["duration_ms"],custom["duration_ms"])
        self.assertEqual(profiled["timing_source"],"fitted-profiled-configuration")
        self.assertEqual(custom["timing_source"],"fitted-extrapolation")
        self.assertNotIn("profile_cost_ns",profiled)

    def test_qkv_function_matches_independently_calculated_features(self):
        c=self.config();h=c["decode"]["hardware"]
        # One GPU, 4 tokens; H=4096, Q=4096, KV=1024, BF16. Include input/output traffic.
        flops=2*4*4096*(4096+2*1024)
        byte_count=2*(4096*(4096+2*1024)+4*4096+4*(4096+2*1024))
        compute_ns=flops/(h["bf16_tensor_tflops"]*1e12)*1e9
        memory_ns=byte_count/(h["dram_bandwidth_TBps"]*1e12)*1e9
        a,b,d=dataset()["cohorts"]["kaist"]["models"]["qkv_proj"]["coefficients"]
        expected=max(compute_ns,a+b*compute_ns+d*memory_ns)/1e6
        self.assertAlmostEqual(timing("qkv_proj",c,"decode",4,1,129)["duration_ms"],expected,places=12)

    def test_causal_attention_work_includes_current_query(self):
        m=MODELS["llama8"];h=HARDWARE["rtx6000"]
        prefill=work_features("attention",m,h,1,512,pc=512,kp=64,nd=0,kv=0)
        # Four independent 128-token causal sequences: sum(1..128) per sequence.
        expected_flops=4*4096*4*128*129/2
        self.assertAlmostEqual(prefill[1],expected_flops/(h["bf16_tensor_tflops"]*1e12)*1e9)
        decode=work_features("attention",m,h,1,4,pc=0,kp=0,nd=4,kv=128)
        self.assertAlmostEqual(decode[1],(4*4096*4*129)/(h["bf16_tensor_tflops"]*1e12)*1e9)

    def test_compute_bandwidth_work_and_continuity_constraints(self):
        c=self.config()
        baseline=timing("qkv_proj",c,"decode",4,1,129)["duration_ms"]
        c["decode"]["hardware"]["clock_scale"]=2
        self.assertLess(timing("qkv_proj",c,"decode",4,1,129)["duration_ms"],baseline)
        c=self.config();c["decode"]["hardware"]["dram_bandwidth_TBps"]*=.5
        self.assertGreater(timing("qkv_proj",c,"decode",4,1,129)["duration_ms"],baseline)
        c=self.config();small=timing("qkv_proj",c,"decode",1,1,129)["duration_ms"]
        self.assertGreaterEqual(baseline,small)
        c["decode"]["hardware"]["bf16_tensor_tflops"]*=1.000001
        nearby=timing("qkv_proj",c,"decode",4,1,129)["duration_ms"]
        self.assertLess(abs(nearby/baseline-1),.000001)
        self.assertGreaterEqual(prediction([0,0,0],[1,1234,0]),1234)

    def test_offline_training_files_not_needed_by_simulation(self):
        read_bytes=Path.read_bytes
        def only_runtime_files(path):
            self.assertNotIn("third_party",path.parts)
            self.assertNotIn(".vendor",path.parts)
            return read_bytes(path)
        dataset.cache_clear()
        with patch.object(Path,"read_bytes",only_runtime_files):
            r=simulate(self.config())
        self.assertGreater(r["metrics"]["throughput_tps"],0)
        self.assertTrue(all(o["timing_source"].startswith("fitted-") for o in r["layers"]))

    def test_precision_engine_model_and_shape_bounds_are_explicit(self):
        for mutate in [
            lambda c:c["prefill"]["hardware"].update(clock_scale=2),
            lambda c:c["prefill"].update(hardware_id="custom"),
            lambda c:c.update(precision="fp8"),
            lambda c:c.update(kv_precision="fp8"),
            lambda c:c.update(inference_engine="trtllm"),
        ]:
            c=self.config();mutate(c);c=validate(c)
            self.assertEqual(timing("qkv_proj",c,"decode",4,1,129)["timing_source"],"fitted-extrapolation")
        self.assertEqual(timing("qkv_proj",self.config(),"prefill",4,4096,4096)["timing_source"],"fitted-extrapolation")

    def test_independent_validation_splits_are_recorded_separately(self):
        d=dataset();v=d["validation"]
        self.assertEqual(v["overall"]["count"],599451)
        self.assertEqual(v["by_source"]["kaist"]["count"],9411)
        self.assertEqual(v["by_source"]["vidur"]["count"],590040)
        self.assertEqual(v["protocol"]["outer_folds"],5)
        self.assertEqual(v["protocol"]["inner_folds"],3)
        self.assertLess(v["overall"]["mape_pct"],v["baseline"]["mape_pct"])
        self.assertEqual(dataset()["cohorts"]["kaist"]["models"]["embedding"]["family"],"F9")

    def test_precision_cohorts_and_missing_cache_fallback_are_explicit(self):
        c=self.config();c["precision"]=c["kv_precision"]="fp16"
        fp16=timing("qkv_proj",c,"decode",4,1,129)
        self.assertEqual(fp16["fit_cohort"],"vidur")
        c=self.config();c["decode"]["hardware"]["l2_cache_MiB"]=0
        fallback=timing("act_fn",c,"decode",4,1,129)
        self.assertEqual(fallback["fit_family"],"F0")
        self.assertIn("L2",fallback["fit_fallback_reason"])

    def test_embedding_dispatch_boundary_is_used_by_runtime(self):
        c=self.config()
        before=timing("embedding",c,"decode",16,1,129)
        after=timing("embedding",c,"decode",17,1,129)
        self.assertEqual(before["fit_family"],"F9")
        self.assertLess(after["duration_ms"],before["duration_ms"])

    def test_legacy_roofline_reproducible_and_new_outputs_are_predictions(self):
        c=copy.deepcopy(DEFAULT);c["timing_mode"]="roofline"
        self.assertAlmostEqual(simulate(c)["metrics"]["throughput_tps"],191.4542607691084)
        r=simulate(DEFAULT)
        self.assertIn("Fitted GPU function",r["calibration"]["summary"])
        self.assertIn("extrapolation",r["calibration"]["summary"])
        self.assertEqual(len(r["calibration"]["dataset"]["profiles"]),41)
        c["timing_mode"]="invalid"
        with self.assertRaisesRegex(ValueError,"Timing model"):
            validate(c)


if __name__=="__main__":
    unittest.main()
