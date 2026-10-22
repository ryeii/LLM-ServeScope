"""Behavioral checks for configurable model geometry and independent server pools."""

import copy
import json
import unittest
from pathlib import Path
from backend.model import (
    DEFAULT,
    MODELS,
    HARDWARE,
    catalog,
    parameters,
    phase_ops,
    simulate,
    validate,
)


class ConfigurationTests(unittest.TestCase):
    def config(self):
        c = copy.deepcopy(DEFAULT)
        c['storage']['enabled'] = False  # General serving tests, including multiple replicas.
        c.update(
            model_id="llama8",
            requests=12,
            input_tokens=128,
            output_tokens=8,
            arrival_rate=10000,
            batch_size=4,
            prefill_chunk=64,
        )
        return c

    def test_all_presets_run_and_retain_model_sources(self):
        for model_id in MODELS:
            if model_id == "custom":
                continue
            with self.subTest(model=model_id):
                c = self.config()
                c["model_id"] = model_id
                if model_id == "qwen235":
                    c["prefill"]["tp"] = c["decode"]["tp"] = 8
                r = simulate(c)
                self.assertGreater(r["metrics"]["throughput_tps"], 0)
                self.assertTrue(r["model"]["source_url"].startswith("https://"))
                names = [o["name"] for o in r["phases"]["decode"]["operations"]]
                self.assertEqual("Expert router" in names, r["model"]["experts"] > 1)
                self.assertEqual("QK normalization" in names, r["model"]["qk_norm"])
        for hardware_id, h in HARDWARE.items():
            with self.subTest(hardware=hardware_id):
                c = self.config()
                for ph in ("prefill", "decode"):
                    c[ph].update(
                        hardware_id=hardware_id, hardware=copy.deepcopy(h), tp=1
                    )
                self.assertGreater(simulate(c)["metrics"]["throughput_tps"], 0)

    def test_new_models_match_frozen_publisher_configs(self):
        sources = json.loads(
            (Path(__file__).parents[1] / "docs/model-sources.json").read_text()
        )
        for item in sources:
            m = next(x for x in MODELS.values() if x["source"] == item["repo"])
            raw = item["config"]
            for k, source_k in [
                ("hidden", "hidden_size"),
                ("layers", "num_hidden_layers"),
                ("heads", "num_attention_heads"),
                ("kv_heads", "num_key_value_heads"),
                ("head_dim", "head_dim"),
                ("tie_embeddings", "tie_word_embeddings"),
            ]:
                self.assertEqual(m[k], raw[source_k])
            self.assertEqual(
                m["intermediate"],
                raw.get("moe_intermediate_size", raw["intermediate_size"]),
            )
            self.assertEqual(m["source_revision"], item["revision"])

    def test_custom_mha_mqa_moe_and_tied_weights(self):
        c = self.config()
        c.update(model_id="custom", custom_model=copy.deepcopy(MODELS["llama8"]))
        for kv_heads, experts, top_k in [(32, 1, 1), (1, 1, 1), (8, 8, 2)]:
            c["custom_model"].update(kv_heads=kv_heads, experts=experts, top_k=top_k)
            r = simulate(c)
            self.assertEqual(r["model"]["kv_heads"], kv_heads)
            self.assertEqual(r["model"]["top_k"], top_k)
        m = c["custom_model"]
        a = parameters(m)[0]
        m["tie_embeddings"] = True
        self.assertEqual(a - parameters(m)[0], m["hidden"] * m["vocab"])
        for key, val in [
            ("kv_heads", 3),
            ("top_k", 9),
            ("layers", True),
            ("max_context", 100),
        ]:
            bad = copy.deepcopy(c)
            bad["custom_model"][key] = val
            with self.assertRaises(ValueError):
                simulate(bad)

    def test_phase_batches_split_with_exact_token_accounting(self):
        c = self.config()
        c["decode"].update(batch_size=2, replicas=2)
        r = simulate(c)
        self.assertEqual([x["id"] for x in r["requests"]], list(range(12)))
        self.assertEqual(len({x["prefill_batch_id"] for x in r["requests"]}), 3)
        self.assertEqual(len({x["decode_batch_id"] for x in r["requests"]}), 6)
        for request in r["requests"]:
            self.assertEqual(len(request["token_times_ms"]), 8)
            self.assertAlmostEqual(
                request["latency_ms"], request["ttft_ms"] + request["tpot_ms"] * 7
            )
        for e in r["timeline"]:
            self.assertEqual(len(e["request_ids"]), e["batch_size"])
            if e["phase"] == "decode":
                self.assertLessEqual(e["batch_size"], 2)
        self.assertEqual(r["layers"][-1]["batch"], 2)
        self.assertAlmostEqual(
            r["metrics"]["throughput_tps"], 96e3 / r["metrics"]["elapsed_ms"]
        )

    def test_concurrency_slot_held_across_all_decode_steps(self):
        c = self.config()
        c["decode"].update(batch_size=2, max_concurrent_batches=1, pp=2)
        r = simulate(c)
        intervals = {}
        for e in r["timeline"]:
            if e["phase"] == "decode":
                intervals.setdefault(e["batch_id"], []).append(e)
        spans = sorted(
            (min(e["start_ms"] for e in rows), max(e["end_ms"] for e in rows))
            for rows in intervals.values()
        )
        for a, b in zip(spans, spans[1:]):
            self.assertLessEqual(a[1], b[0] + 1e-9)
        relaxed = copy.deepcopy(c)
        relaxed["decode"]["max_concurrent_batches"] = 128
        self.assertLessEqual(
            simulate(relaxed)["metrics"]["elapsed_ms"], r["metrics"]["elapsed_ms"]
        )

    def test_kv_precision_changes_kv_capacity_writes_and_handoff(self):
        c = self.config()
        a = simulate(c)
        c["kv_precision"] = "fp8"
        b = simulate(c)
        for ph in ("prefill", "decode"):
            self.assertAlmostEqual(
                a["memory"][ph]["kv_GB"], b["memory"][ph]["kv_GB"] * 2
            )
            self.assertEqual(
                a["memory"][ph]["weights_GB"], b["memory"][ph]["weights_GB"]
            )
        traffic = lambda r: sum(
            e["duration_ms"] - c["network"]["spine_latency_us"] / 1000
            for e in r["timeline"]
            if e["phase"] == "transfer"
        )
        self.assertAlmostEqual(traffic(a), 2 * traffic(b))

    def test_memory_allocation_limits_and_runtime_reservation(self):
        c = self.config()
        c["decode"].update(memory_utilization=0.9, runtime_reserved_GB=2)
        r = simulate(c)
        self.assertEqual(r["memory"]["decode"]["capacity_GB"], 70)
        self.assertEqual(r["memory"]["decode"]["physical_capacity_GB"], 80)
        c["decode"]["runtime_reserved_GB"] = 73
        with self.assertRaisesRegex(ValueError, "reservation"):
            simulate(c)
        c["decode"]["runtime_reserved_GB"] = 69
        with self.assertRaisesRegex(ValueError, "usable memory"):
            simulate(c)

    def test_phase_engine_clock_and_launch_controls_change_service(self):
        c = validate(self.config())
        a = phase_ops(c, "decode", 4, 1, 128)[0]
        c["decode"]["engine"] = "eager"
        b = phase_ops(c, "decode", 4, 1, 128)[0]
        self.assertGreater(
            sum(o["duration_ms"] for o in b), sum(o["duration_ms"] for o in a)
        )
        c["decode"]["engine"] = "inherit"
        c["decode"]["hardware"]["clock_scale"] = 0.5
        d = phase_ops(c, "decode", 4, 1, 128)[0]
        self.assertAlmostEqual(d[1]["compute_ms"], a[1]["compute_ms"] * 2)
        c["decode"]["hardware"]["launch_overhead_us"] = 10
        e = phase_ops(c, "decode", 4, 1, 128)[0]
        self.assertAlmostEqual(e[1]["duration_ms"] - d[1]["duration_ms"], 0.01)

    def test_parallel_kv_links_have_distinct_nonoverlapping_lanes(self):
        c = self.config()
        c.update(input_tokens=1024, prefill_chunk=1024)
        c["prefill"]["replicas"] = 4
        c["network"]["spine_bw_GBps"] = 0.01
        a = simulate(c)
        c["network"]["handoff_channels"] = 4
        b = simulate(c)
        self.assertLess(b["metrics"]["elapsed_ms"], a["metrics"]["elapsed_ms"])
        lanes = {}
        for e in b["timeline"]:
            if e["phase"] == "transfer":
                lanes.setdefault(e["resource"], []).append(e)
        self.assertEqual(len(lanes), 3)
        for rows in lanes.values():
            for x, y in zip(rows, rows[1:]):
                self.assertLessEqual(x["end_ms"], y["start_ms"])

    def test_shared_pool_component_energy_includes_both_phases(self):
        c = self.config()
        c["separated"] = False
        c["prefill"]["hardware"].update(
            idle_watts=0, tdp_watts=1000000, pj_per_mem_byte=1
        )
        r = simulate(c)
        self.assertAlmostEqual(
            r["metrics"]["energy_j"], r["traffic"]["hbm_bytes"] * 1e-12
        )

    def test_legacy_schema_defaults_and_colocated_pool_resolution(self):
        c = self.config()
        c["decode"].update(batch_size=1, engine="eager", memory_utilization=0.5)
        c["separated"] = False
        resolved = validate(c)
        self.assertEqual(resolved["prefill"], resolved["decode"])
        self.assertEqual(resolved["kv_precision"], "bf16")
        self.assertEqual(resolved["prefill"]["batch_size"], 4)
        self.assertEqual(catalog()["default_config"]["network"]["handoff_channels"], 1)
        self.assertEqual(validate(resolved), resolved)


if __name__ == "__main__":
    unittest.main()
