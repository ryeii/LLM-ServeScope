"""Compare unmodified ServeScope analytical timing with the pinned KAIST backend.

Run from the repository with .venv/bin/python integration/compare_roofline_profiles.py.
Raw runs are retained under runs/. Original roofline is the default; calibrated
mode evaluates the fitted GPU function. No new GPU inference is run.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.catalog import HARDWARE
from backend.model import DEFAULT, phase_ops, simulate, validate

DEVICES = [("RTXPRO6000", "rtx6000", 1, 96, 1597.6),
           ("RTXPRO6000", "rtx6000", 2, 96, 1597.6),
           ("RTX4090", "rtx4090", 1, 24, 1008.0)]
WORKLOADS = [(1, 128, 8), (4, 128, 8), (4, 512, 32)]
OP_NAMES = ["Input RMSNorm", "QKV projection", "Rotary position", "Attention",
            "Output projection", "FFN RMSNorm", "Gate + up projection", "SwiGLU",
            "Down projection"]
NATIVE_NAMES = ["layernorm", "qkv_proj", "rotary_emb", "attention", "o_proj",
                "layernorm", "gate_up_proj", "act_fn", "down_proj"]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/roofline-profile-comparison-2026-10-19")
    parser.add_argument("--timing-mode", choices=["roofline", "calibrated"], default="roofline")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT / "runs")
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    rows, operators, evidence = [], [], []
    verified_artifacts = 0
    for hardware, hardware_id, tp, capacity, bandwidth in DEVICES:
        for batch, prompt, generation in WORKLOADS:
            name = f"{hardware.lower()}-tp{tp}-b{batch}-p{prompt}-o{generation}"
            native_dir = output / name
            command = [str(ROOT / ".venv-sim/bin/python"), str(ROOT / "integration/native_runner.py"),
                       "--hardware", hardware, "--tp", str(tp), "--requests", str(batch),
                       "--input-tokens", str(prompt), "--output-tokens", str(generation),
                       "--memory-gib", str(capacity), "--memory-bandwidth-gbps", str(bandwidth),
                       "--link-bandwidth-gibps", "16", "--link-latency-ns", "1000",
                       "--arrival-interval-ms", "0", "--output", str(native_dir)]
            if not (native_dir / "result.json").exists():
                print(f"Running {name}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            native = json.loads((native_dir / "result.json").read_text())
            assert native["status"] == "completed", native["status"]
            for key, value in dict(hardware=hardware, tp=tp, requests=batch,
                                   input_tokens=prompt, output_tokens=generation,
                                   memory_gib=capacity, memory_bandwidth_gbps=bandwidth,
                                   link_bandwidth_gibps=16, link_latency_ns=1000,
                                   arrival_interval_ms=0).items():
                assert native["config"][key] == value, (key, native["config"][key], value)
            assert len(native["requests"]) == batch
            assert all(r["input"] == prompt and r["output"] == generation for r in native["requests"])
            for artifact in native["artifacts"]:
                path = ROOT / artifact["path"]
                assert path.stat().st_size == artifact["bytes"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
                verified_artifacts += 1
            with (native_dir / "requests.csv").open() as stream:
                original = list(csv.DictReader(stream))
            span_s = (max(int(r["end_time"]) for r in original)
                      - min(int(r["arrival"]) for r in original)) / 1e9
            csv_throughput = sum(int(r["output"]) for r in original) / span_s
            assert abs(csv_throughput - native["metrics"]["request_span_output_tokens_per_second"]) < 1e-9

            config = copy.deepcopy(DEFAULT)
            config.update(model_id="llama8", timing_mode=args.timing_mode, precision="bf16", kv_precision="bf16",
                          inference_engine="vllm", separated=False, requests=batch,
                          input_tokens=prompt, output_tokens=generation, batch_size=batch,
                          prefill_chunk=prompt, arrival_rate=10000)
            h = copy.deepcopy(HARDWARE[hardware_id])
            h.update(main_memory_GB=capacity * 2**30 / 1e9,
                     dram_bandwidth_TBps=bandwidth / 1000,
                     intra_node_bw_GBps=16 * 2**30 / 1e9, network_latency_us=1)
            config["prefill"] = dict(hardware_id=hardware_id, hardware=h,
                                     tp=tp, pp=1, ep=1, replicas=1, batch_size=batch,
                                     memory_utilization=0.9)
            analytical = simulate(config)
            dump(native_dir / "analytical-config.json", config)
            dump(native_dir / "analytical-result.json", analytical)
            am, nm = analytical["metrics"], native["metrics"]
            row = dict(case=name, hardware=hardware, tp=tp, batch=batch,
                       input_tokens=prompt, output_tokens=generation)
            for label, a, n in [
                ("throughput_tps", am["throughput_tps"], nm["request_span_output_tokens_per_second"]),
                ("ttft_ms", am["ttft_mean_ms"], nm["mean_ttft_ms"]),
                ("tpot_ms", am["tpot_mean_ms"], nm["mean_tpot_ms"]),
                ("elapsed_ms", am["elapsed_ms"], nm["request_span_ms"]),
            ]:
                row[f"analytical_{label}"] = a
                row[f"native_{label}"] = n
                row[f"delta_{label}_pct"] = (a / n - 1) * 100
            row["native_total_clock_throughput_tps"] = nm["output_tokens_per_second"]
            row["native_control_tail_ms"] = nm["simulated_duration_ms"] - nm["request_span_ms"]
            row["analytical_batch_ready_ms"] = (batch - 1) / 10000 * 1000
            rows.append(row)

            c = validate(config)
            for graph_index, phase, tokens, context in [(0, "prefill", prompt, prompt),
                                                         (1, "decode", 1, prompt + 1)]:
                graph = native["graphs"][graph_index]
                comp = [n for n in graph["nodes"] if n["type"] == "COMP_NODE"]
                names = [re.sub(r"_\d+$", "", n["name"].removeprefix("COMP_NODE_")) for n in comp]
                assert names[0] == "embedding", names[:12]
                assert names[1:10] == NATIVE_NAMES, names[:12]
                ops = {o["name"]: o for o in phase_ops(c, phase, batch, tokens, context)[0]}
                for op_name, node in zip(OP_NAMES, comp[1:10]):
                    a = ops[op_name]["duration_ms"] * 1000
                    n = node["profile_cost_ns"] / 1000
                    operators.append(dict(case=name, phase=phase, operator=op_name,
                                          analytical_us=a, native_profile_us=n,
                                          delta_pct=(a / n - 1) * 100,
                                          analytical_bound=ops[op_name]["bound"],
                                          native_node=node["name"]))
            evidence.append(dict(case=name, native_command=command,
                                 native_result=str((native_dir / "result.json").relative_to(ROOT)),
                                 native_result_sha256=hashlib.sha256((native_dir / "result.json").read_bytes()).hexdigest(),
                                 provenance=native["provenance"]))
            print(json.dumps(row), flush=True)

    result = dict(kind="analytical_vs_native_profile_simulation", source_revision=revision,
                  working_tree_source_sha256={str(p): hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                              for p in [Path("backend/model.py"),Path("backend/catalog.py"),Path("backend/calibration.py"),Path("backend/gpu_functions.py"),Path("backend/gpu-fit.json")]},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  methodology={
                      "comparison": f"ServeScope {args.timing_mode} timing vs executed pinned KAIST/ASTRA backend; not measured end-to-end inference.",
                      "workload": "Llama 3.1 8B BF16 weights/activations and BF16 KV; colocated; one replica; PP1/EP1; one prefill chunk; max batched tokens 2048; no prefix sharing/offload.",
                      "throughput": "Output tokens / earliest-arrival-to-last-completion window for both. Native total-clock throughput retained separately; includes control tail.",
                      "arrivals": "Native requests arrive simultaneously. Analytical public schema requires positive rate; use maximum 10000 requests/s (0.1 ms spacing). Batch-four initial wait is 0.3 ms and mean formation delay 0.15 ms. Single-request cases match exactly. No production code patched.",
                      "hardware": "Capacity and memory/link bandwidth units aligned. Link =16 GiB/s, latency=1000 ns. Compute/memory efficiency inputs remain at their reference settings. Calibrated mode uses the fitted GPU function; original mode uses the uncalibrated tile/roofline assumptions.",
                      "software": "RTXPRO6000 profiles report vLLM0.28.0/CUDA13.0; RTX4090 vLLM0.19.0/CUDA12.8. Fitted mode learns kernel baselines from these snapshots; other engine/version changes are unvalidated. Original mode uses an assumed generic engine factor.",
                      "operator_comparison": "First transformer block COMP_NODE costs in native prefill and first decode graphs vs matching nine analytical operators. COMP_NODE cost is whole profiled kernel elapsed time, including its internal memory effects; not arithmetic-only time. Embedding/final norm/head/communications excluded from matched block comparison, retained in end-to-end simulations.",
                      "scope": "End-to-end differences include fitted operator residuals (or original roofline assumptions), operator coverage, traffic/cache estimates, scheduling and communication. This small suite is not a universal accuracy assessment. No energy comparison because accounting scopes differ. No model weights or hardware inference executed.",
                  }, statistics={metric: {
                      "min_signed_delta_pct": min(r[f"delta_{metric}_pct"] for r in rows),
                      "max_signed_delta_pct": max(r[f"delta_{metric}_pct"] for r in rows),
                      "mean_absolute_relative_difference_pct": statistics.mean(abs(r[f"delta_{metric}_pct"]) for r in rows),
                  } for metric in ["throughput_tps", "ttft_ms", "tpot_ms"]},
                  verified_native_artifact_count=verified_artifacts,
                  results=rows, operators=operators, evidence=evidence)
    dump(output / "comparison.json", result)
    for filename, data in [("comparison.csv", rows), ("operators.csv", operators)]:
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    print(f"Saved {output / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
