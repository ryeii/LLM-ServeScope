import copy, json, math, unittest
from backend.model import DEFAULT, HARDWARE, fabric, simulate, validate


class BehavioralModelTests(unittest.TestCase):
    def config(self):
        c = copy.deepcopy(DEFAULT)
        c.update(
            requests=8,
            input_tokens=128,
            output_tokens=16,
            arrival_rate=10000,
            batch_size=4,
            prefill_chunk=128,
        )
        return c

    def test_deterministic_finite_and_token_accounting(self):
        c = self.config()
        a = simulate(c)
        b = simulate(c)
        self.assertEqual(a, b)
        json.dumps(a, allow_nan=False)
        self.assertEqual(len(a["requests"]), 8)
        for r in a["requests"]:
            self.assertEqual(len(r["token_times_ms"]), 16)
            self.assertGreaterEqual(r["ttft_ms"], 0)
            self.assertAlmostEqual(r["latency_ms"], r["ttft_ms"] + r["tpot_ms"] * 15)
        self.assertAlmostEqual(
            a["metrics"]["throughput_tps"], 128 / (a["metrics"]["elapsed_ms"] / 1000)
        )

    def test_slower_hbm_increases_service_time(self):
        fast = self.config()
        slow = copy.deepcopy(fast)
        for phase in ("prefill", "decode"):
            slow[phase]["hardware"]["dram_bandwidth_TBps"] = 0.25
        a, b = simulate(fast), simulate(slow)
        self.assertGreater(b["metrics"]["elapsed_ms"], a["metrics"]["elapsed_ms"])
        self.assertLess(b["metrics"]["throughput_tps"], a["metrics"]["throughput_tps"])

    def test_spine_transfer_is_a_ttft_floor(self):
        fast = self.config()
        slow = copy.deepcopy(fast)
        slow["network"]["spine_bw_GBps"] = 0.05
        a, b = simulate(fast), simulate(slow)
        self.assertGreater(b["metrics"]["ttft_mean_ms"], a["metrics"]["ttft_mean_ms"])
        self.assertTrue(any(e["phase"] == "transfer" for e in b["timeline"]))

    def test_sram_removes_kv_reads_but_not_weights_or_writes(self):
        c = self.config()
        c["input_tokens"] = 2048
        c["prefill_chunk"] = 2048
        for p in ("prefill", "decode"):
            c[p]["hardware"]["sram_cache_MB"] = 0
        a = simulate(c)
        for p in ("prefill", "decode"):
            c[p]["hardware"]["sram_cache_MB"] = 100000
        b = simulate(c)
        op = lambda r: next(
            o for o in r["phases"]["decode"]["operations"] if o["name"] == "Attention"
        )
        self.assertGreater(op(a)["kv_hbm_bytes"], op(b)["kv_hbm_bytes"])
        self.assertGreater(op(b)["kv_hbm_bytes"], 0)  # Appending KV still writes HBM.
        self.assertEqual(
            a["memory"], b["memory"]
        )  # Cache changes traffic, not HBM residency.

    def test_queues_never_overlap_one_resource(self):
        c = self.config()
        c["prefill"]["pp"] = 2
        c["decode"]["pp"] = 2
        r = simulate(c)
        for resource in {e["resource"] for e in r["timeline"]}:
            rows = sorted(
                [e for e in r["timeline"] if e["resource"] == resource],
                key=lambda e: e["start_ms"],
            )
            for a, b in zip(rows, rows[1:]):
                self.assertLessEqual(a["end_ms"], b["start_ms"] + 1e-9)
        self.assertTrue(0 < r["metrics"]["gpu_utilization"] <= 1)

    def test_colocation_shares_gpu_capacity_and_has_no_handoff(self):
        c = self.config()
        c["separated"] = False
        r = simulate(c)
        self.assertEqual(r["metrics"]["gpu_count"], 4)
        self.assertFalse(any(e["phase"] == "transfer" for e in r["timeline"]))
        self.assertTrue(all(e["resource"].startswith("shared:") for e in r["timeline"]))

    def test_capacity_and_invalid_inputs_rejected(self):
        c = self.config()
        c["prefill"]["hardware"]["main_memory_GB"] = 1
        with self.assertRaisesRegex(ValueError, "above"):
            simulate(c)
        for key, value in [
            ("requests", True),
            ("arrival_rate", float("nan")),
            ("precision", "madeup"),
        ]:
            c = self.config()
            c[key] = value
            with self.assertRaises(ValueError):
                simulate(c)
        c = self.config()
        c["precision"] = "fp4"
        with self.assertRaisesRegex(ValueError, "no fp4"):
            simulate(c)

    def test_network_tiers_and_moe_routes(self):
        h = copy.deepcopy(HARDWARE["h100"])
        h["same_rail_gpus"] = 16
        n = DEFAULT["network"]
        self.assertEqual(fabric(8, h, n)["name"], "Intra-node")
        self.assertEqual(fabric(16, h, n)["name"], "Same rail / TOR")
        self.assertEqual(fabric(32, h, n)["name"], "Scale-out NIC")
        self.assertEqual(fabric(128, h, n)["name"], "DC spine / WAN")
        self.assertLess(fabric(128, h, n)["bw"], fabric(8, h, n)["bw"])
        c = self.config()
        c["model_id"] = "qwen30"
        c["prefill"]["ep"] = 2
        c["decode"]["ep"] = 2
        r = simulate(c)
        names = [o["name"] for o in r["phases"]["decode"]["operations"]]
        self.assertIn("EP dispatch", names)
        self.assertIn("EP combine", names)

    def test_energy_and_cost_bounds(self):
        c = self.config()
        r = simulate(c)
        m = r["metrics"]
        self.assertLessEqual(
            m["energy_j"], m["gpu_count"] * 700 * m["elapsed_ms"] / 1000
        )
        self.assertGreater(m["energy_j"], m["gpu_count"] * 120 * m["elapsed_ms"] / 1000)
        self.assertAlmostEqual(
            m["cost_usd"], m["gpu_count"] * 3 * m["elapsed_ms"] / 3.6e6
        )
        for p in ("prefill", "decode"):
            c[p]["hardware"].update(
                pj_per_flop=0.2, pj_per_mem_byte=10, pj_per_net_byte=20, pj_per_vop=0.5
            )
        p = simulate(c)["metrics"]
        self.assertTrue(math.isfinite(p["energy_j"]))
        self.assertLessEqual(
            p["energy_j"], p["gpu_count"] * 700 * p["elapsed_ms"] / 1000
        )

    def test_single_output_token_has_no_decode_interval(self):
        c = self.config()
        c["output_tokens"] = 1
        r = simulate(c)
        self.assertEqual(r["metrics"]["tpot_mean_ms"], 0)
        self.assertFalse(any(e["phase"] == "decode" for e in r["timeline"]))


if __name__ == "__main__":
    unittest.main()
