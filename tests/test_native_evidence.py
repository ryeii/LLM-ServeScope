"""Verify retained native results against their original CSV and graph artifacts."""

import csv
import hashlib
import json
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    (ROOT / "runs/native-demo-tp1/result.json").exists(),
    "Retained native runs are optional; execute integration/native_runner.py first.",
)
class NativeEvidenceTests(unittest.TestCase):
    def load(self, name):
        return json.loads((ROOT / "runs" / name / "result.json").read_text())

    def test_summary_matches_native_csv_and_actual_dag(self):
        r = self.load("native-demo-tp1")
        with (ROOT / "runs/native-demo-tp1/requests.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(
            r["metrics"]["mean_ttft_ms"],
            statistics.mean(int(v["TTFT"]) for v in rows) / 1e6,
        )
        self.assertAlmostEqual(
            r["metrics"]["mean_tpot_ms"],
            statistics.mean(int(v["TPOT"]) for v in rows) / 1e6,
        )
        for g in r["graphs"]:
            ids = {n["id"] for n in g["nodes"]}
            self.assertEqual(len(ids), len(g["nodes"]))
            self.assertEqual(len(ids), 294)
            for e in g["edges"]:
                self.assertIn(e["source"], ids)
                self.assertIn(e["target"], ids)
        self.assertTrue(
            all(v["tracked_source_clean"] for v in r["provenance"]["sources"].values())
        )

    def test_native_link_bandwidth_changes_real_completion(self):
        fast = self.load("native-demo-tp2")
        slow = self.load("native-demo-tp2-link1")
        for k in [
            "hardware",
            "input_tokens",
            "output_tokens",
            "requests",
            "tp",
            "link_latency_ns",
        ]:
            self.assertEqual(fast["config"][k], slow["config"][k])
        self.assertEqual(fast["config"]["link_bandwidth_gibps"], 16)
        self.assertEqual(slow["config"]["link_bandwidth_gibps"], 1)
        self.assertGreater(
            slow["metrics"]["simulated_duration_ms"],
            fast["metrics"]["simulated_duration_ms"],
        )
        self.assertTrue(
            any(n["type"] == "COMM_COLL_NODE" for n in fast["graphs"][0]["nodes"])
        )

    def test_original_artifact_hashes(self):
        for name in ["native-demo-tp1", "native-demo-tp2", "native-demo-tp2-link1"]:
            r = self.load(name)
            for artifact in r["artifacts"]:
                p = ROOT / artifact["path"]
                self.assertEqual(p.stat().st_size, artifact["bytes"])
                self.assertEqual(
                    hashlib.sha256(p.read_bytes()).hexdigest(), artifact["sha256"]
                )


if __name__ == "__main__":
    unittest.main()
