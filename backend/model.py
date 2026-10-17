"""System-level behavioral model. These equations are our extension, not KAIST timings."""

from __future__ import annotations
import copy, hashlib, json, math, statistics
from pathlib import Path
import simpy

ROOT = Path(__file__).resolve().parents[1]
PIN = "b09c7aaab4b072d73136641841cc2688616c49eb"
MODELS = {
    "llama8": dict(
        name="Llama 3.1 8B",
        source="meta-llama/Llama-3.1-8B",
        hidden=4096,
        intermediate=14336,
        layers=32,
        heads=32,
        kv_heads=8,
        head_dim=128,
        vocab=128256,
        experts=1,
        top_k=1,
    ),
    "llama70": dict(
        name="Llama 3.1 70B",
        source="meta-llama/Llama-3.1-70B",
        hidden=8192,
        intermediate=28672,
        layers=80,
        heads=64,
        kv_heads=8,
        head_dim=128,
        vocab=128256,
        experts=1,
        top_k=1,
    ),
    "qwen30": dict(
        name="Qwen3 30B-A3B",
        source="Qwen/Qwen3-30B-A3B-Instruct-2507",
        hidden=2048,
        intermediate=768,
        layers=48,
        heads=32,
        kv_heads=4,
        head_dim=128,
        vocab=151936,
        experts=128,
        top_k=8,
    ),
}
BASE = dict(
    fp4_tensor_tflops=0,
    fp8_tensor_tflops=1979,
    bf16_tensor_tflops=989,
    fp16_tensor_tflops=989,
    tf32_tensor_tflops=494,
    int8_tensor_tflops=1979,
    fp32_tflops=67,
    main_memory_GB=80,
    dram_bandwidth_TBps=3.35,
    sram_cache_MB=50,
    num_sms=132,
    tensor_width=64,
    intra_node_bw_GBps=450,
    chip_cluster_size=8,
    same_rail_bw_GBps=50,
    same_rail_gpus=64,
    scale_out_bw_GBps=50,
    default_GPUs_per_rack=64,
    cluster_default_racks=8,
    tdp_watts=700,
    idle_watts=120,
    compute_efficiency=0.65,
    memory_efficiency=0.8,
    network_latency_us=2,
    collective_allreduce_fraction=0.75,
    collective_alltoall_fraction=0.65,
    collective_alltoall_offset=3,
    pj_per_flop=0,
    pj_per_vop=0,
    pj_per_mem_byte=0,
    pj_per_net_byte=0,
    hourly_cost=3.0,
)
HARDWARE = {
    "h100": dict(
        BASE,
        name="H100 SXM · illustrative",
        provenance="Illustrative strategy preset; peak inputs require calibration. Dense, non-sparse compute convention.",
    ),
    "rtx6000": dict(
        BASE,
        name="RTX PRO 6000 · profile reference",
        main_memory_GB=102,
        dram_bandwidth_TBps=1.5976,
        sram_cache_MB=134.217728,
        num_sms=188,
        bf16_tensor_tflops=503,
        fp16_tensor_tflops=503,
        fp8_tensor_tflops=1006,
        fp4_tensor_tflops=2012,
        fp32_tflops=125,
        intra_node_bw_GBps=30.85,
        chip_cluster_size=2,
        tdp_watts=600,
        idle_watts=90,
        hourly_cost=2.75,
        provenance="Capacity/BW/L2/SM/TDP from upstream RTXPRO6000 hardware.yaml; compute peaks and efficiency are explicit illustrative inputs. Native mode uses measured operation profiles.",
    ),
    "custom": dict(
        BASE,
        name="Custom accelerator",
        provenance="User-supplied architecture assumptions; not measured unless separately calibrated.",
    ),
}
DEFAULT = dict(
    model_id="llama70",
    precision="bf16",
    inference_engine="vllm",
    requests=16,
    input_tokens=2048,
    output_tokens=64,
    arrival_rate=8,
    batch_size=4,
    prefill_chunk=512,
    separated=True,
    prefill=dict(
        hardware_id="h100",
        hardware=copy.deepcopy(HARDWARE["h100"]),
        tp=4,
        pp=1,
        ep=1,
        replicas=1,
    ),
    decode=dict(
        hardware_id="h100",
        hardware=copy.deepcopy(HARDWARE["h100"]),
        tp=4,
        pp=1,
        ep=1,
        replicas=1,
    ),
    network=dict(spine_bw_GBps=25, spine_latency_us=10),
)
BYTES = {"fp4": 0.5, "fp8": 1, "int8": 1, "bf16": 2, "fp16": 2, "tf32": 4, "fp32": 4}
PEAK = {p: p + "_tensor_tflops" for p in ["fp4", "fp8", "int8", "bf16", "fp16", "tf32"]}
PEAK["fp32"] = "fp32_tflops"


def number(value, name, low, high, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
        or (integer and int(value) != value)
    ):
        raise ValueError(
            f'{name} must be {"an integer " if integer else ""}between {low} and {high}.'
        )
    return int(value) if integer else float(value)


def validate(config):
    c = copy.deepcopy(config)
    if c.get("model_id") not in MODELS:
        raise ValueError("Choose a supported model architecture.")
    if c.get("precision") not in BYTES:
        raise ValueError("Unsupported precision.")
    if c.get("inference_engine") not in ("vllm", "trtllm", "eager"):
        raise ValueError("Unsupported engine policy.")
    for name, lo, hi in [
        ("requests", 1, 128),
        ("input_tokens", 1, 32768),
        ("output_tokens", 1, 256),
        ("batch_size", 1, 32),
        ("prefill_chunk", 1, 8192),
    ]:
        c[name] = number(c.get(name), name, lo, hi, True)
    c["arrival_rate"] = number(c.get("arrival_rate"), "arrival_rate", 0.01, 10000)
    if type(c.get("separated")) is not bool:
        raise ValueError("separated must be a boolean.")
    for phase in ("prefill", "decode"):
        p = c[phase]
        for name in ("tp", "pp", "ep", "replicas"):
            p[name] = number(p.get(name), phase + "." + name, 1, 16, True)
        m = MODELS[c["model_id"]]
        if m["heads"] % p["tp"] or m["layers"] % p["pp"]:
            raise ValueError(
                "TP must divide attention heads and PP must divide transformer layers."
            )
        if m["experts"] == 1 and p["ep"] != 1:
            raise ValueError("Expert parallelism requires a mixture-of-experts model.")
        if m["experts"] % p["ep"]:
            raise ValueError("EP must divide the expert count.")
        if not isinstance(p.get("hardware"), dict):
            raise ValueError("Hardware inputs are required.")
        for key in BASE:
            lo = 0.001
            hi = 1e7
            if key in PEAK.values():
                lo = 0
            if key.startswith("pj_") or key in (
                "idle_watts",
                "collective_alltoall_offset",
                "sram_cache_MB",
            ):
                lo = 0
            if key in (
                "compute_efficiency",
                "memory_efficiency",
                "collective_allreduce_fraction",
                "collective_alltoall_fraction",
            ):
                hi = 1
            if key in (
                "chip_cluster_size",
                "same_rail_gpus",
                "default_GPUs_per_rack",
                "cluster_default_racks",
                "num_sms",
                "tensor_width",
            ):
                lo = 1
                hi = 65536
            p["hardware"][key] = number(
                p["hardware"].get(key),
                phase + "." + key,
                lo,
                hi,
                key
                in (
                    "chip_cluster_size",
                    "same_rail_gpus",
                    "default_GPUs_per_rack",
                    "cluster_default_racks",
                    "num_sms",
                    "tensor_width",
                ),
            )
        h = p["hardware"]
        if h[PEAK[c["precision"]]] <= 0:
            raise ValueError(
                f'{phase.capitalize()} hardware has no {c["precision"]} compute profile.'
            )
        if h["idle_watts"] > h["tdp_watts"]:
            raise ValueError("Idle power cannot exceed TDP.")
        if (
            not h["chip_cluster_size"]
            <= h["same_rail_gpus"]
            <= h["default_GPUs_per_rack"]
        ):
            raise ValueError(
                "Network domains must nest: chip domain ≤ same rail ≤ GPUs per rack."
            )
        if (
            p["tp"] * p["ep"] * p["pp"] * p["replicas"]
            > h["default_GPUs_per_rack"] * h["cluster_default_racks"]
        ):
            raise ValueError("Requested GPU count exceeds the configured cluster.")
    c["network"]["spine_bw_GBps"] = number(
        c["network"].get("spine_bw_GBps"), "Spine bandwidth", 0.001, 1e6
    )
    c["network"]["spine_latency_us"] = number(
        c["network"].get("spine_latency_us"), "Spine latency", 0, 1e6
    )
    if not c["separated"]:
        c["decode"] = copy.deepcopy(c["prefill"])
    budget = math.ceil(c["requests"] / c["batch_size"]) * (
        math.ceil(c["input_tokens"] / c["prefill_chunk"]) * c["prefill"]["pp"]
        + c["output_tokens"] * c["decode"]["pp"]
    )
    if budget > 40000:
        raise ValueError(
            "Scenario exceeds the interactive limit. Reduce requests, output length, or pipeline depth."
        )
    return c


def parameters(m):
    h = m["hidden"]
    q = m["heads"] * m["head_dim"]
    kv = m["kv_heads"] * m["head_dim"]
    shared = m["layers"] * (h * (q + 2 * kv) + q * h + 2 * h) + 2 * h * m["vocab"] + h
    expert = m["layers"] * 3 * h * m["intermediate"] * m["experts"]
    return shared, expert


def fabric(group, h, network):
    if group <= h["chip_cluster_size"]:
        return dict(
            name="Intra-node",
            bw=h["intra_node_bw_GBps"],
            latency=h["network_latency_us"],
        )
    if group <= h["same_rail_gpus"]:
        return dict(
            name="Same rail / TOR",
            bw=min(
                h["intra_node_bw_GBps"], h["same_rail_bw_GBps"], h["scale_out_bw_GBps"]
            ),
            latency=h["network_latency_us"] * 2,
        )
    if group <= h["default_GPUs_per_rack"]:
        return dict(
            name="Scale-out NIC",
            bw=min(h["same_rail_bw_GBps"], h["scale_out_bw_GBps"]),
            latency=h["network_latency_us"] * 3,
        )
    return dict(
        name="DC spine / WAN",
        bw=min(h["scale_out_bw_GBps"], network["spine_bw_GBps"]),
        latency=network["spine_latency_us"],
    )


def phase_ops(c, phase, batch, tokens, context):
    m = MODELS[c["model_id"]]
    p = c[phase]
    h = p["hardware"]
    tp = p["tp"]
    ep = p["ep"]
    dtype = BYTES[c["precision"]]
    hidden = m["hidden"]
    q = m["heads"] * m["head_dim"]
    kv = m["kv_heads"] * m["head_dim"]
    inter = m["intermediate"]
    n = batch * tokens
    peak = h[PEAK[c["precision"]]] * 1e12
    # Engine knobs are disclosed scenario assumptions, not measured engine deltas.
    engine = {"vllm": (0.97, 2.5), "trtllm": (1, 2), "eager": (0.8, 8)}[
        c["inference_engine"]
    ]
    active_experts = (
        1
        if m["experts"] == 1
        else m["experts"] * (1 - (1 - m["top_k"] / m["experts"]) ** n)
    )
    kv_layer = 2 * batch * context * max(1, m["kv_heads"] / tp) * m["head_dim"] * 2
    kv_write = 2 * batch * tokens * max(1, m["kv_heads"] / tp) * m["head_dim"] * 2
    sram_per_layer = h["sram_cache_MB"] * 1e6 / (m["layers"] / p["pp"])
    spill = max(0, kv_layer - sram_per_layer)
    attention_flops = (
        4
        * batch
        * m["heads"]
        * m["head_dim"]
        * (tokens * max(0, context - tokens) + tokens * (tokens + 1) / 2)
        / tp
    )
    ops = []

    def op(name, flops, weights, activation=0, extra=0, out=hidden, vector=False):
        traffic = weights + activation + extra
        tiles = max(1, math.ceil(n / 64)) * math.ceil(out / (tp * h["tensor_width"]))
        tile_util = min(1, max(0.08, tiles / (h["num_sms"] * 2))) if not vector else 1
        effective = (
            (h["fp32_tflops"] * 1e12 if vector else peak)
            * h["compute_efficiency"]
            * tile_util
            * engine[0]
        )
        compute = flops / effective * 1000
        memory = (
            traffic / (h["dram_bandwidth_TBps"] * 1e12 * h["memory_efficiency"]) * 1000
        )
        launch = engine[1] / 1000
        ops.append(
            dict(
                id=name.lower().replace(" ", "_"),
                name=name,
                flops=flops,
                hbm_bytes=traffic,
                weight_bytes=weights,
                activation_bytes=activation,
                kv_hbm_bytes=extra,
                compute_ms=compute,
                memory_ms=memory,
                network_ms=0,
                launch_ms=launch,
                duration_ms=max(compute, memory) + launch,
                bound="Compute" if compute >= memory else "Memory",
                arithmetic_intensity=flops / max(1, traffic),
                achieved_tflops=flops / max((max(compute, memory) + launch) * 1e9, 1),
                tile_utilization=tile_util,
            )
        )

    op(
        "Input RMSNorm",
        5 * n * hidden / tp,
        hidden * dtype / tp,
        2 * n * hidden * dtype / tp,
        vector=True,
    )
    op(
        "QKV projection",
        2 * n * hidden * (q + 2 * kv) / tp,
        hidden * (q + 2 * kv) * dtype / tp,
        2 * n * hidden * dtype / tp,
        out=q + 2 * kv,
    )
    op("Rotary position", 6 * n * q / tp, 0, 2 * n * q * dtype / tp, vector=True)
    op("Attention", attention_flops, 0, 2 * n * q * dtype / tp, spill + kv_write, out=q)
    op(
        "Output projection",
        2 * n * q * hidden / tp,
        q * hidden * dtype / tp,
        2 * n * hidden * dtype / tp,
    )
    op(
        "FFN RMSNorm",
        5 * n * hidden / tp,
        hidden * dtype / tp,
        2 * n * hidden * dtype / tp,
        vector=True,
    )
    mlp_div = tp * ep
    op(
        "Gate + up projection",
        4 * n * hidden * inter * m["top_k"] / mlp_div,
        2 * hidden * inter * active_experts * dtype / mlp_div,
        3 * n * inter * dtype * m["top_k"] / mlp_div,
        out=2 * inter,
    )
    op(
        "SwiGLU",
        5 * n * inter * m["top_k"] / mlp_div,
        0,
        3 * n * inter * dtype * m["top_k"] / mlp_div,
        vector=True,
        out=inter,
    )
    op(
        "Down projection",
        2 * n * hidden * inter * m["top_k"] / mlp_div,
        hidden * inter * active_experts * dtype / mlp_div,
        2 * n * hidden * dtype / tp,
    )
    net = fabric(tp * ep, h, c["network"])

    def collective(name, group, amount, eff, latency):
        if group <= 1:
            return
        is_reduce = "all-reduce" in name
        phases = 2 * (group - 1) if is_reduce else group - 1
        volume = amount * (
            2 * (group - 1) / group if is_reduce else (group - 1) / group
        )
        ms = (
            volume / (net["bw"] * 1e9 * eff) * 1000
            + phases * net["latency"] / 1000
            + latency / 1000
        )
        ops.append(
            dict(
                id=name.lower().replace(" ", "_"),
                name=name,
                flops=0,
                hbm_bytes=0,
                weight_bytes=0,
                activation_bytes=0,
                kv_hbm_bytes=0,
                compute_ms=0,
                memory_ms=0,
                network_ms=ms,
                launch_ms=0,
                duration_ms=ms,
                bound="Network",
                arithmetic_intensity=0,
                achieved_tflops=0,
                tile_utilization=0,
                network_bytes=volume,
                fabric=net["name"],
            )
        )

    collective(
        "Attention all-reduce",
        tp,
        n * hidden * dtype,
        h["collective_allreduce_fraction"],
        0,
    )
    collective(
        "FFN all-reduce", tp, n * hidden * dtype, h["collective_allreduce_fraction"], 0
    )
    collective(
        "EP dispatch",
        ep,
        n * hidden * dtype * m["top_k"] / tp,
        h["collective_alltoall_fraction"],
        h["collective_alltoall_offset"],
    )
    collective(
        "EP combine",
        ep,
        n * hidden * dtype * m["top_k"] / tp,
        h["collective_alltoall_fraction"],
        h["collective_alltoall_offset"],
    )
    order = [
        "Input RMSNorm",
        "QKV projection",
        "Rotary position",
        "Attention",
        "Output projection",
        "Attention all-reduce",
        "FFN RMSNorm",
        "EP dispatch",
        "Gate + up projection",
        "SwiGLU",
        "Down projection",
        "EP combine",
        "FFN all-reduce",
    ]
    ops.sort(key=lambda o: order.index(o["name"]))
    # One transformer stage plus amortized embedding/head work. Head is charged only on the last pipeline stage by the scheduler.
    return ops, dict(
        kv_read_bytes=kv_layer,
        kv_spill_bytes=spill,
        sram_hit_fraction=min(1, sram_per_layer / max(1, kv_layer)),
        fabric=net,
        peak_tflops=peak / 1e12,
    )


def percentile(values, p):
    if not values:
        return 0
    a = sorted(values)
    x = (len(a) - 1) * p
    lo = math.floor(x)
    hi = math.ceil(x)
    return a[lo] + (a[hi] - a[lo]) * (x - lo)


def simulate(config):
    c = validate(config)
    m = MODELS[c["model_id"]]
    shared, expert = parameters(m)
    dtype = BYTES[c["precision"]]
    max_batch = min(c["batch_size"], c["requests"])
    memory = {}
    warnings = []
    for phase in ("prefill", "decode"):
        p = c[phase]
        h = p["hardware"]
        group = p["tp"] * p["pp"] * p["ep"]
        weight = (shared + expert / p["ep"]) * dtype / (p["tp"] * p["pp"])
        # Conservatively reserve max-context KV for every request assigned to one replica, including queued batches.
        batches_per_replica = math.ceil(
            math.ceil(c["requests"] / max_batch) / p["replicas"]
        )
        concurrent = min(c["requests"], max_batch * batches_per_replica)
        kv = (
            2
            * concurrent
            * (c["input_tokens"] + c["output_tokens"] - 1)
            * max(1, m["kv_heads"] / p["tp"])
            * m["head_dim"]
            * 2
            * m["layers"]
            / p["pp"]
        )
        activation = (
            max_batch
            * min(c["input_tokens"], c["prefill_chunk"])
            * m["hidden"]
            * dtype
            * 8
            / p["tp"]
            if phase == "prefill"
            else max_batch * m["hidden"] * dtype * 8 / p["tp"]
        )
        used = weight + kv + activation
        capacity = h["main_memory_GB"] * 1e9
        memory[phase] = dict(
            weights_GB=weight / 1e9,
            kv_GB=kv / 1e9,
            activation_GB=activation / 1e9,
            total_GB=used / 1e9,
            capacity_GB=h["main_memory_GB"],
            headroom_GB=(capacity - used) / 1e9,
            gpus=group * p["replicas"],
        )
        if used > capacity:
            raise ValueError(
                f'{phase.capitalize()} requires about {used/1e9:.1f} GB/GPU, above {h["main_memory_GB"]:g} GB. Increase TP/PP/EP or memory capacity, or reduce batch/context.'
            )
    env = simpy.Environment()
    resources = {}
    pipeline_links = {}
    busy = {}
    timeline = []
    requests = []
    traffic = {"hbm_bytes": 0, "network_bytes": 0, "flops": 0}
    component_work = {"Compute": 0, "Memory": 0, "Network": 0, "Launch": 0}
    op_cache = {}
    phase_traffic = {
        ph: dict(tensor_flops=0, vector_flops=0, hbm_bytes=0, network_bytes=0)
        for ph in ("prefill", "decode")
    }
    for phase in ("prefill", "decode"):
        p = c[phase]
        if phase == "decode" and not c["separated"]:
            resources[phase] = resources["prefill"]
            continue
        resources[phase] = [
            [simpy.Resource(env, capacity=1) for _ in range(p["pp"])]
            for _ in range(p["replicas"])
        ]
    transfer = simpy.Resource(env, capacity=1)
    for phase in ("prefill", "decode"):
        shared_phase = "prefill" if not c["separated"] else phase
        for replica in range(c[phase]["replicas"]):
            for stage in range(c[phase]["pp"] - 1):
                pipeline_links.setdefault(
                    (shared_phase, replica, stage), simpy.Resource(env, capacity=1)
                )

    def work(phase, batch, tokens, context, replica, batch_id, chunk_index):
        p = c[phase]
        layers = m["layers"] / p["pp"]
        hardware = p["hardware"]
        key = (phase, batch, tokens, context)
        if key not in op_cache:
            op_cache[key] = phase_ops(c, phase, batch, tokens, context)
        ops, detail = op_cache[key]
        for stage, resource in enumerate(resources[phase][replica]):
            with resource.request() as ticket:
                yield ticket
                begin = env.now
                component = dict(
                    Compute=sum(o["compute_ms"] for o in ops) * layers,
                    Memory=sum(o["memory_ms"] for o in ops) * layers,
                    Network=sum(o["network_ms"] for o in ops) * layers,
                    Launch=sum(o["launch_ms"] for o in ops) * layers,
                )
                duration = sum(o["duration_ms"] for o in ops) * layers
                # Logits: last pipeline stage, sampled once at prefill end / each decode iteration.
                last_prefill = phase == "prefill" and context >= c["input_tokens"]
                if stage == p["pp"] - 1 and (phase == "decode" or last_prefill):
                    head_flops = 2 * batch * m["hidden"] * m["vocab"] / p["tp"]
                    head_bytes = m["hidden"] * m["vocab"] * dtype / p["tp"]
                    hc = (
                        head_flops
                        / (
                            hardware[PEAK[c["precision"]]]
                            * 1e12
                            * hardware["compute_efficiency"]
                        )
                        * 1000
                    )
                    hm = (
                        head_bytes
                        / (
                            hardware["dram_bandwidth_TBps"]
                            * 1e12
                            * hardware["memory_efficiency"]
                        )
                        * 1000
                    )
                    duration += max(hc, hm)
                    component["Compute"] += hc
                    component["Memory"] += hm
                    traffic["flops"] += head_flops * p["tp"] * p["ep"]
                    traffic["hbm_bytes"] += head_bytes * p["tp"] * p["ep"]
                    phase_traffic[phase]["tensor_flops"] += (
                        head_flops * p["tp"] * p["ep"]
                    )
                    phase_traffic[phase]["hbm_bytes"] += head_bytes * p["tp"] * p["ep"]
                yield env.timeout(max(duration, 1e-9))
                label_phase = phase if c["separated"] else "shared"
                rid = f"{label_phase}:{replica}:{stage}"
                busy[rid] = busy.get(rid, 0) + duration
                timeline.append(
                    dict(
                        phase=phase,
                        resource=rid,
                        batch_id=batch_id,
                        chunk=chunk_index,
                        start_ms=begin,
                        end_ms=env.now,
                        duration_ms=duration,
                        batch_size=batch,
                        context=context,
                        components=component,
                    )
                )
                for o in ops:
                    traffic["flops"] += o["flops"] * layers * p["tp"] * p["ep"]
                    traffic["hbm_bytes"] += o["hbm_bytes"] * layers * p["tp"] * p["ep"]
                    traffic["network_bytes"] += (
                        o.get("network_bytes", 0) * layers * p["tp"] * p["ep"]
                    )
                    energy_key = (
                        "vector_flops"
                        if o["name"]
                        in ("Input RMSNorm", "FFN RMSNorm", "Rotary position", "SwiGLU")
                        else "tensor_flops"
                    )
                    phase_traffic[phase][energy_key] += (
                        o["flops"] * layers * p["tp"] * p["ep"]
                    )
                    phase_traffic[phase]["hbm_bytes"] += (
                        o["hbm_bytes"] * layers * p["tp"] * p["ep"]
                    )
                    phase_traffic[phase]["network_bytes"] += (
                        o.get("network_bytes", 0) * layers * p["tp"] * p["ep"]
                    )
                for k, v in component.items():
                    component_work[k] += v * p["tp"] * p["ep"]
            if stage < p["pp"] - 1:
                shared_phase = "prefill" if not c["separated"] else phase
                with pipeline_links[
                    (shared_phase, replica, stage)
                ].request() as link_ticket:
                    yield link_ticket
                    begin = env.now
                    link = fabric(p["tp"] * p["ep"] * p["pp"], hardware, c["network"])
                    payload = batch * tokens * m["hidden"] * dtype / p["tp"]
                    latency = (
                        payload / (link["bw"] * 1e9) * 1000 + link["latency"] / 1000
                    )
                    yield env.timeout(latency)
                    timeline.append(
                        dict(
                            phase=phase,
                            kind="pipeline_transfer",
                            resource=f"{'shared' if not c['separated'] else phase}:{replica}:link{stage}",
                            batch_id=batch_id,
                            chunk=chunk_index,
                            start_ms=begin,
                            end_ms=env.now,
                            duration_ms=latency,
                            batch_size=batch,
                            context=context,
                            components={
                                "Compute": 0,
                                "Memory": 0,
                                "Network": latency,
                                "Launch": 0,
                            },
                        )
                    )
                    traffic["network_bytes"] += payload * p["tp"] * p["ep"]
                    phase_traffic[phase]["network_bytes"] += payload * p["tp"] * p["ep"]
                    component_work["Network"] += latency * p["tp"] * p["ep"]

    def run_batch(ids, bid):
        batch = len(ids)
        arrivals = [i / c["arrival_rate"] * 1000 for i in ids]
        yield env.timeout(max(0, max(arrivals) - env.now))
        start = env.now
        done = 0
        while done < c["input_tokens"]:
            n = min(c["prefill_chunk"], c["input_tokens"] - done)
            done += n
            yield env.process(
                work(
                    "prefill", batch, n, done, bid % c["prefill"]["replicas"], bid, done
                )
            )
        prefill_end = env.now
        kv_bytes = (
            2
            * batch
            * c["input_tokens"]
            * m["kv_heads"]
            * m["head_dim"]
            * 2
            * m["layers"]
        )
        if c["separated"]:
            with transfer.request() as ticket:
                yield ticket
                ts = env.now
                transfer_ms = (
                    kv_bytes / (c["network"]["spine_bw_GBps"] * 1e9) * 1000
                    + c["network"]["spine_latency_us"] / 1000
                )
                yield env.timeout(transfer_ms)
                timeline.append(
                    dict(
                        phase="transfer",
                        resource="spine",
                        batch_id=bid,
                        chunk=0,
                        start_ms=ts,
                        end_ms=env.now,
                        duration_ms=transfer_ms,
                        batch_size=batch,
                        context=c["input_tokens"],
                        components={
                            "Compute": 0,
                            "Memory": 0,
                            "Network": transfer_ms,
                            "Launch": 0,
                        },
                    )
                )
                traffic["network_bytes"] += kv_bytes
                component_work["Network"] += transfer_ms
                phase_traffic["prefill"]["network_bytes"] += kv_bytes
        first = env.now
        times = [first]
        for step in range(c["output_tokens"] - 1):
            yield env.process(
                work(
                    "decode",
                    batch,
                    1,
                    c["input_tokens"] + step + 1,
                    bid % c["decode"]["replicas"],
                    bid,
                    step,
                )
            )
            times.append(env.now)
        for identifier, arrival in zip(ids, arrivals):
            requests.append(
                dict(
                    id=identifier,
                    arrival_ms=arrival,
                    batch_ready_ms=start,
                    prefill_end_ms=prefill_end,
                    first_token_ms=first,
                    finish_ms=env.now,
                    ttft_ms=first - arrival,
                    tpot_ms=(env.now - first) / max(1, c["output_tokens"] - 1),
                    latency_ms=env.now - arrival,
                    token_times_ms=times,
                )
            )

    for bid, offset in enumerate(range(0, c["requests"], c["batch_size"])):
        env.process(
            run_batch(
                list(range(offset, min(c["requests"], offset + c["batch_size"]))), bid
            )
        )
    env.run()
    requests.sort(key=lambda r: r["id"])
    elapsed = max(r["finish_ms"] for r in requests)
    n_gpus = memory["prefill"]["gpus"] + (
        memory["decode"]["gpus"] if c["separated"] else 0
    )
    gpu_busy = sum(busy.values()) / max(1, elapsed)
    energy_j = 0
    cost = 0
    for phase in ("prefill", "decode"):
        if phase == "decode" and not c["separated"]:
            continue
        p = c[phase]
        h = p["hardware"]
        prefix = phase if c["separated"] else "shared"
        b = (
            sum(v for k, v in busy.items() if k.startswith(prefix + ":"))
            * p["tp"]
            * p["ep"]
        )
        active_watts = min(
            h["tdp_watts"], h["idle_watts"] + (h["tdp_watts"] - h["idle_watts"]) * 0.85
        )
        # Optional component coefficients replace the activity increment, then cap at TDP.
        coeffs = any(
            h[k] > 0
            for k in ("pj_per_flop", "pj_per_vop", "pj_per_mem_byte", "pj_per_net_byte")
        )
        if coeffs:
            pt = phase_traffic[phase]
            dynamic = (
                pt["tensor_flops"] * h["pj_per_flop"]
                + pt["vector_flops"] * h["pj_per_vop"]
                + pt["hbm_bytes"] * h["pj_per_mem_byte"]
                + pt["network_bytes"] * h["pj_per_net_byte"]
            ) * 1e-12
            dynamic = min(dynamic, (h["tdp_watts"] - h["idle_watts"]) * b / 1000)
            energy_j += (
                h["idle_watts"] * memory[phase]["gpus"] * elapsed / 1000 + dynamic
            )
        else:
            energy_j += (
                h["idle_watts"] * memory[phase]["gpus"] * elapsed / 1000
                + (active_watts - h["idle_watts"]) * b / 1000
            )
        cost += memory[phase]["gpus"] * h["hourly_cost"] * elapsed / 3.6e6
    sample_phase = {}
    layer_rows = []
    for phase in ("prefill", "decode"):
        tokens = min(c["prefill_chunk"], c["input_tokens"]) if phase == "prefill" else 1
        context = (
            tokens
            if phase == "prefill"
            else c["input_tokens"] + (c["output_tokens"] - 1) / 2
        )
        ops, detail = phase_ops(c, phase, max_batch, tokens, context)
        sample_phase[phase] = dict(operations=ops, detail=detail)
        for o in ops:
            layer_rows.append(
                dict(o, phase=phase, batch=max_batch, tokens=tokens, context=context)
            )
    metrics = dict(
        throughput_tps=c["requests"] * c["output_tokens"] / (elapsed / 1000),
        request_throughput=c["requests"] / (elapsed / 1000),
        prompt_tps=c["requests"] * c["input_tokens"] / (elapsed / 1000),
        ttft_mean_ms=statistics.mean(r["ttft_ms"] for r in requests),
        ttft_p95_ms=percentile([r["ttft_ms"] for r in requests], 0.95),
        tpot_mean_ms=statistics.mean(r["tpot_ms"] for r in requests),
        latency_p95_ms=percentile([r["latency_ms"] for r in requests], 0.95),
        elapsed_ms=elapsed,
        gpu_count=n_gpus,
        energy_j=energy_j,
        tokens_per_joule=c["requests"] * c["output_tokens"] / max(energy_j, 1e-12),
        tokens_per_gpu_s=c["requests"] * c["output_tokens"] / (elapsed / 1000) / n_gpus,
        cost_usd=cost,
        cost_per_million_tokens=cost / (c["requests"] * c["output_tokens"]) * 1e6,
        gpu_utilization=sum(
            v
            * (
                c["prefill"]["tp"] * c["prefill"]["ep"]
                if k.startswith(("prefill:", "shared:"))
                else c["decode"]["tp"] * c["decode"]["ep"]
            )
            for k, v in busy.items()
        )
        / (elapsed * n_gpus),
    )
    warnings = [
        "Behavioral estimates, not measured hardware performance or a KAIST validation claim.",
        "Fixed microbatches; FIFO phase resources with pipeline overlap. This extension does not implement native vLLM continuous batching.",
        "Ideal KV SRAM read reuse; KV writes still reach HBM. HBM stores the complete KV payload.",
        "Tensor tiling, uniform MoE routing, engine efficiency, activation traffic and activity-based energy are explicit assumptions. Quantization conversion overhead and numerical quality are excluded.",
        "Pipeline activation messages use the limiting fabric tier of the entire pool as a conservative placement approximation.",
        "Network tiers use the limiting end-to-end bandwidth plus ring/dispatch overhead; they do not model individual packets or switches.",
        "TTFT includes batch formation, queueing, prefill and KV handoff; the first token is released after the handoff.",
    ]
    if c["output_tokens"] == 1:
        warnings.append(
            "TPOT is zero because no inter-token interval exists for a one-token output."
        )
    return dict(
        schema_version="memanalysis.behavioral.v1",
        engine="MemAnalysis behavioral / SimPy",
        config=c,
        model=m,
        metrics=metrics,
        memory=memory,
        traffic=traffic,
        component_work_ms=component_work,
        requests=requests,
        timeline=sorted(timeline, key=lambda e: e["start_ms"]),
        layers=layer_rows,
        phases=sample_phase,
        warnings=warnings,
        provenance=dict(
            kind="behavioral_extension",
            upstream_url="https://github.com/casys-kaist/LLMServingSim",
            upstream_revision=PIN,
            model_architecture_source="Upstream configs/model; formulas implemented independently in backend/model.py",
            config_sha256=hashlib.sha256(
                json.dumps(c, sort_keys=True).encode()
            ).hexdigest(),
            timing_source="Closed-form roofline + hierarchical network + SimPy resource contention; not native LLMServingSim clocks",
            precision_note="Peak TFLOPS must use dense/non-sparse values. Weight precision changes bytes; KV is BF16 (2 bytes). GB/TB are decimal.",
        ),
    )


def catalog():
    return dict(
        models=MODELS, hardware=HARDWARE, default_config=DEFAULT, upstream_revision=PIN
    )
