"""Resolved behavioral presets. Vendor peaks are dense; other defaults are assumptions."""

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

# Peak specs are not kernel timing profiles. Links are one-direction per GPU.
# Idle power, efficiency, rental rates, cache reservations and topology remain
# editable scenario assumptions, including the generic SM/CU tiling abstraction.
HARDWARE["h100"]["source_url"] = "https://www.nvidia.com/en-us/data-center/h100/"
HARDWARE["h100"]["name"] = "H100 SXM 80 GB"
HARDWARE["rtx6000"][
    "source_url"
] = "https://github.com/casys-kaist/LLMServingSim/blob/b09c7aaab4b072d73136641841cc2688616c49eb/profiler/perf/RTXPRO6000/hardware.yaml"
HARDWARE["a100"] = dict(
    BASE,
    name="A100 SXM 80 GB",
    main_memory_GB=80,
    dram_bandwidth_TBps=2.039,
    bf16_tensor_tflops=312,
    fp16_tensor_tflops=312,
    tf32_tensor_tflops=156,
    fp8_tensor_tflops=0,
    fp4_tensor_tflops=0,
    int8_tensor_tflops=624,
    fp32_tflops=19.5,
    num_sms=108,
    sram_cache_MB=0,
    intra_node_bw_GBps=300,
    tdp_watts=400,
    idle_watts=70,
    hourly_cost=2,
    source_url="https://www.nvidia.com/en-us/data-center/a100/",
)
HARDWARE["h200"] = dict(
    BASE,
    name="H200 SXM 141 GB",
    main_memory_GB=141,
    dram_bandwidth_TBps=4.8,
    bf16_tensor_tflops=989.5,
    fp16_tensor_tflops=989.5,
    tf32_tensor_tflops=494.5,
    sram_cache_MB=0,
    hourly_cost=4,
    source_url="https://www.nvidia.com/en-us/data-center/h200/",
)
HARDWARE["b200"] = dict(
    BASE,
    name="B200 SXM 180 GB",
    main_memory_GB=180,
    dram_bandwidth_TBps=7.7,
    bf16_tensor_tflops=2250,
    fp16_tensor_tflops=2250,
    tf32_tensor_tflops=1100,
    fp8_tensor_tflops=4500,
    fp4_tensor_tflops=9000,
    int8_tensor_tflops=4500,
    fp32_tflops=75,
    num_sms=148,
    sram_cache_MB=0,
    intra_node_bw_GBps=900,
    tdp_watts=1000,
    idle_watts=150,
    hourly_cost=6,
    source_url="https://lenovopress.lenovo.com/lp2226.pdf",
)
HARDWARE["l40s"] = dict(
    BASE,
    name="L40S 48 GB",
    main_memory_GB=48,
    dram_bandwidth_TBps=0.864,
    bf16_tensor_tflops=362.05,
    fp16_tensor_tflops=362.05,
    tf32_tensor_tflops=183,
    fp8_tensor_tflops=733,
    fp4_tensor_tflops=0,
    int8_tensor_tflops=733,
    fp32_tflops=91.6,
    num_sms=142,
    sram_cache_MB=0,
    intra_node_bw_GBps=31.5,
    chip_cluster_size=1,
    tdp_watts=350,
    idle_watts=50,
    hourly_cost=1.5,
    source_url="https://www.nvidia.com/en-us/data-center/l40s/",
)
HARDWARE["rtx4090"] = dict(
    BASE,
    name="RTX 4090 24 GB",
    main_memory_GB=24,
    dram_bandwidth_TBps=1.008,
    bf16_tensor_tflops=165.2,
    fp16_tensor_tflops=165.2,
    tf32_tensor_tflops=82.6,
    fp8_tensor_tflops=330.3,
    fp4_tensor_tflops=0,
    int8_tensor_tflops=660.6,
    fp32_tflops=82.6,
    num_sms=128,
    sram_cache_MB=0,
    intra_node_bw_GBps=31.5,
    chip_cluster_size=1,
    tdp_watts=450,
    idle_watts=40,
    hourly_cost=0.75,
    source_url="https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf",
)
HARDWARE["mi300x"] = dict(
    BASE,
    name="AMD MI300X 192 GB",
    main_memory_GB=192,
    dram_bandwidth_TBps=5.3,
    bf16_tensor_tflops=1307.4,
    fp16_tensor_tflops=1307.4,
    tf32_tensor_tflops=653.7,
    fp8_tensor_tflops=2614.9,
    fp4_tensor_tflops=0,
    int8_tensor_tflops=2614.9,
    fp32_tflops=163.4,
    num_sms=304,
    sram_cache_MB=0,
    intra_node_bw_GBps=448,
    tdp_watts=750,
    idle_watts=100,
    hourly_cost=3,
    source_url="https://rocmdocs.amd.com/en/docs-7.2.4/conceptual/gpu-arch/mi300.html",
)
for key, h in HARDWARE.items():
    h.update(clock_scale=1.0, launch_overhead_us=0.0)
    h.setdefault(
        "provenance",
        "Published capacity, bandwidth and dense compute peaks; generic tiling, fabric topology, idle power, efficiency and cost are scenario assumptions. KV SRAM is a reserved budget, not the entire physical cache. No measured latency profile.",
    )
BASE.update(clock_scale=1.0, launch_overhead_us=0.0)

MODELS["qwen06"] = {
    "name": "Qwen3 0.6B",
    "source": "Qwen/Qwen3-0.6B",
    "source_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "source_url": "https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json",
    "hidden": 1024,
    "intermediate": 3072,
    "layers": 28,
    "heads": 16,
    "kv_heads": 8,
    "head_dim": 128,
    "vocab": 151936,
    "experts": 1,
    "top_k": 1,
    "max_context": 40960,
    "tie_embeddings": True,
    "qk_norm": True,
}

MODELS["qwen4"] = {
    "name": "Qwen3 4B",
    "source": "Qwen/Qwen3-4B",
    "source_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "source_url": "https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/config.json",
    "hidden": 2560,
    "intermediate": 9728,
    "layers": 36,
    "heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "vocab": 151936,
    "experts": 1,
    "top_k": 1,
    "max_context": 40960,
    "tie_embeddings": True,
    "qk_norm": True,
}

MODELS["qwen8"] = {
    "name": "Qwen3 8B",
    "source": "Qwen/Qwen3-8B",
    "source_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "source_url": "https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json",
    "hidden": 4096,
    "intermediate": 12288,
    "layers": 36,
    "heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "vocab": 151936,
    "experts": 1,
    "top_k": 1,
    "max_context": 40960,
    "tie_embeddings": False,
    "qk_norm": True,
}

MODELS["qwen14"] = {
    "name": "Qwen3 14B",
    "source": "Qwen/Qwen3-14B",
    "source_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "source_url": "https://huggingface.co/Qwen/Qwen3-14B/blob/40c069824f4251a91eefaf281ebe4c544efd3e18/config.json",
    "hidden": 5120,
    "intermediate": 17408,
    "layers": 40,
    "heads": 40,
    "kv_heads": 8,
    "head_dim": 128,
    "vocab": 151936,
    "experts": 1,
    "top_k": 1,
    "max_context": 40960,
    "tie_embeddings": False,
    "qk_norm": True,
}

MODELS["qwen32"] = {
    "name": "Qwen3 32B",
    "source": "Qwen/Qwen3-32B",
    "source_revision": "9216db5781bf21249d130ec9da846c4624c16137",
    "source_url": "https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json",
    "hidden": 5120,
    "intermediate": 25600,
    "layers": 64,
    "heads": 64,
    "kv_heads": 8,
    "head_dim": 128,
    "vocab": 151936,
    "experts": 1,
    "top_k": 1,
    "max_context": 40960,
    "tie_embeddings": False,
    "qk_norm": True,
}

MODELS["qwen235"] = {
    "name": "Qwen3 235B-A22B",
    "source": "Qwen/Qwen3-235B-A22B",
    "source_revision": "8efa61729e24bd65b1d152b5ab5409052aa80e65",
    "source_url": "https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json",
    "hidden": 4096,
    "intermediate": 1536,
    "layers": 94,
    "heads": 64,
    "kv_heads": 4,
    "head_dim": 128,
    "vocab": 151936,
    "experts": 128,
    "top_k": 8,
    "max_context": 40960,
    "tie_embeddings": False,
    "qk_norm": True,
}

MODELS["mixtral8x7"] = dict(
    name="Mixtral 8x7B",
    source="mistralai/Mixtral-8x7B-v0.1",
    hidden=4096,
    intermediate=14336,
    layers=32,
    heads=32,
    kv_heads=8,
    head_dim=128,
    vocab=32000,
    experts=8,
    top_k=2,
    max_context=32768,
)
for key, m in MODELS.items():
    m.setdefault("source_revision", "b09c7aaab4b072d73136641841cc2688616c49eb")
    m.setdefault(
        "source_url",
        f"https://github.com/casys-kaist/LLMServingSim/blob/{m['source_revision']}/configs/model/{m['source']}.json",
    )
    m.setdefault("max_context", 262144 if key == "qwen30" else 131072)
    m.setdefault("qk_norm", key == "qwen30")
    m.setdefault("tie_embeddings", False)
MODELS["custom"] = dict(
    MODELS["llama8"],
    name="Custom decoder",
    source="User-defined decoder geometry",
    source_url="",
    source_revision="user",
)

# Physical L2 for shape-informed timing; independent of the editable KV SRAM reservation.
for _gpu, _mib in {"rtx6000":128,"rtx4090":72,"a100":40,"h100":50}.items():
    HARDWARE[_gpu]["l2_cache_MiB"]=_mib
HARDWARE["a40"]=dict(BASE,name="A40 48 GB · Vidur reference",main_memory_GB=48,
    dram_bandwidth_TBps=.696,bf16_tensor_tflops=149.7,fp16_tensor_tflops=149.7,
    fp32_tflops=37.4,fp8_tensor_tflops=0,fp4_tensor_tflops=0,num_sms=84,
    l2_cache_MiB=6,sram_cache_MB=0,intra_node_bw_GBps=56.25,tdp_watts=300,
    idle_watts=50,hourly_cost=1.0,chip_cluster_size=2,
    source_url="https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf",
    provenance="Vidur FP16 profile reference: peak/BW/cache inputs are nominal, not measured run clocks. Fabric, idle power and cost are scenario assumptions.")

# Match the published profiler geometry, including its padded 32768 vocabulary.
# These entries are explicitly profile references rather than new inference measurements.
for _id,_source,_hidden,_inter,_heads,_kv,_layers,_context in [
    ("vidur_llama7","meta-llama/Llama-2-7b-hf",4096,11008,32,32,32,4096),
    ("vidur_llama70","meta-llama/Llama-2-70b-hf",8192,28672,64,8,80,4096),
    ("vidur_codellama34","codellama/CodeLlama-34b-Instruct-hf",8192,22016,64,8,48,16384),
]:
    MODELS[_id]=dict(name={"vidur_llama7":"Llama 2 7B","vidur_llama70":"Llama 2 70B","vidur_codellama34":"CodeLlama 34B"}[_id]+" · Vidur geometry",
        source=_source,source_revision="abae7f63aa857300f5cdc6f5e0d27860cd24721b",
        source_url="https://github.com/microsoft/vidur/blob/abae7f63aa857300f5cdc6f5e0d27860cd24721b/vidur/config/model_config.py",
        hidden=_hidden,intermediate=_inter,heads=_heads,kv_heads=_kv,head_dim=_hidden//_heads,
        layers=_layers,vocab=32768,experts=1,top_k=1,max_context=_context,qk_norm=False,tie_embeddings=False)
