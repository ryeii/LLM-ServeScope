"""Fitted GPU service-time function. Runtime reads coefficients, never timing tables.

Training/validation uses KAIST and Vidur's MIT-licensed profiles offline. The fitted model
and its feature/constraint choices are a ServeScope extension, not KAIST validation.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from .gpu_functions import shape_features, evaluate

OPERATOR_NAMES = {
    "Input RMSNorm": "layernorm", "FFN RMSNorm": "layernorm",
    "QKV projection": "qkv_proj", "QK normalization": "qk_norm",
    "Rotary position": "rotary_emb", "Attention": "attention",
    "Output projection": "o_proj", "Gate + up projection": "gate_up_proj",
    "SwiGLU": "act_fn", "Down projection": "down_proj",
}
TP_STABLE = {"layernorm", "final_layernorm", "sampler"}
VECTOR = {"layernorm", "final_layernorm", "rotary_emb", "qk_norm", "act_fn", "sampler", "embedding"}
GEOMETRY = ("hidden", "intermediate", "heads", "kv_heads", "head_dim", "vocab", "experts", "top_k", "qk_norm")
RATE_FIELDS = ("bf16_tensor_tflops", "fp32_tflops", "dram_bandwidth_TBps")



def decode_dataset(raw):
    data = json.loads(raw)
    payload = dict(data)
    expected = payload.pop("payload_sha256")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if actual != expected:
        raise ValueError("GPU fit dataset checksum mismatch.")
    return data


@lru_cache(maxsize=1)
def dataset():
    return decode_dataset(Path(__file__).with_name("gpu-fit.json").read_bytes())


def work_features(name, m, h, tp, n, *, pc=0, kp=0, nd=0, kv=0, precision="bf16", kv_precision="bf16", explain=False):
    """Ideal arithmetic/data-movement times in ns for the transfer regression.

    These nominal traffic features are not measured DRAM transactions. No SM-wave
    heuristic is multiplied into published peak FLOPS; empirical kernel effects
    are learned from profiles instead. TP-stable norms/sampler stay replicated.
    """
    dtype = {"fp4": .5, "fp8": 1, "int8": 1, "bf16": 2, "fp16": 2,
             "tf32": 4, "fp32": 4}[precision]
    hidden, inter = m["hidden"], m["intermediate"]
    q, k = m["heads"]*m["head_dim"], m["kv_heads"]*m["head_dim"]
    effective_tp = 1 if name in TP_STABLE else tp
    top_k = m.get("top_k", 1)
    active = 1 if m.get("experts", 1) == 1 else m["experts"] * (1-(1-top_k/m["experts"])**n)
    flops = traffic = 0
    if name in ("layernorm", "final_layernorm"):
        flops, traffic = 5*n*hidden, (2*n*hidden+hidden)*dtype
    elif name == "qkv_proj":
        flops = 2*n*hidden*(q+2*k)/tp
        traffic = (hidden*(q+2*k)/tp+n*hidden+n*(q+2*k)/tp)*dtype
    elif name == "qk_norm":
        flops, traffic = 5*n*(q+k)/tp, (2*n*(q+k)/tp+2*m["head_dim"])*dtype
    elif name == "rotary_emb":
        flops, traffic = 6*n*q/tp, 2*n*(q+k)*dtype/tp
    elif name == "attention":
        flops = 4*q/tp*(pc*(kp+.5)+nd*(kv+1))
        kv_dtype = {"bf16": 2, "fp16": 2, "fp8": 1}[kv_precision]
        traffic = (dtype*2*(pc+nd)*q+kv_dtype*2*k*(pc+kp+nd*(kv+1)))/tp
    elif name == "o_proj":
        flops = 2*n*q*hidden/tp
        traffic = (q*hidden/tp+n*q/tp+n*hidden)*dtype
    elif name == "gate_up_proj":
        flops = 4*n*hidden*inter*top_k/tp
        traffic = (2*hidden*inter*active/tp+n*hidden+2*n*inter*top_k/tp)*dtype
    elif name == "act_fn":
        flops, traffic = 5*n*inter*top_k/tp, 3*n*inter*top_k*dtype/tp
    elif name == "down_proj":
        flops = 2*n*hidden*inter*top_k/tp
        traffic = (hidden*inter*active/tp+n*inter*top_k/tp+n*hidden)*dtype
    elif name == "embedding":
        traffic = n*(hidden*dtype*2/tp+4)
    elif name == "lm_head":
        flops = 2*n*hidden*m["vocab"]/tp
        traffic = (hidden*m["vocab"]/tp+n*(hidden+m["vocab"]/tp))*dtype
    elif name == "sampler":
        flops, traffic = 5*n*m["vocab"], n*m["vocab"]*4
    else:
        raise KeyError(name)
    peak = h["fp32_tflops"] if name in VECTOR else h[precision+"_tensor_tflops"] if precision != "fp32" else h["fp32_tflops"]
    compute = flops/max(1, peak*1e12*h.get("clock_scale", 1))*1e9
    memory = traffic/max(1, h["dram_bandwidth_TBps"]*1e12)*1e9
    # Preserve the meaning of user edits relative to calibration defaults.
    compute *= .65/h.get("compute_efficiency", .65)
    memory *= .8/h.get("memory_efficiency", .8)
    features = [1.0, compute, memory]
    if explain:
        return dict(features=features, flops=flops, traffic_bytes=traffic, peak_tflops=peak,
                    effective_tp=effective_tp, n=n, pc=pc, kp=kp, nd=nd, kv=kv,
                    active_experts=active, dtype_bytes=dtype)
    return features


def prediction(coefficients, features, compute_floor=None):
    # Monotone, nonnegative fitted service time with an arithmetic speed limit.
    floor = features[1] if compute_floor is None else compute_floor
    return max(1.0, floor, sum(a*b for a, b in zip(coefficients, features)))



def timing(name, config, phase, batch, tokens, context):
    """One fitted function for every GPU/input, including profiled identities.

    Identity is used only to label whether the prediction falls within profiled
    configurations. It never selects a timing table, residual, or correction.
    """
    if config.get("timing_mode", "calibrated") == "roofline":
        return None
    from .catalog import MODELS
    pool = config[phase]
    h, tp = pool["hardware"], pool["tp"]
    m = config["custom_model"] if config["model_id"] == "custom" else MODELS[config["model_id"]]
    effective_tp = tp*pool["ep"] if name in ("gate_up_proj", "act_fn", "down_proj") else tp
    engine = config["inference_engine"] if pool["engine"] == "inherit" else pool["engine"]
    data = dataset()
    fit_key = name+":"+phase if name == "attention" else name
    cohort_name = "vidur" if config["precision"] == "fp16" else "kaist"
    cohort = data["cohorts"][cohort_name]
    fit = cohort["models"].get(fit_key)
    fallback_reason = None
    if fit is None:
        cohort_name, cohort = "kaist", data["cohorts"]["kaist"]
        fit = cohort["models"].get(fit_key)
        fallback_reason = "Operator absent from precision-specific corpus; KAIST function extrapolated."
    if fit is None:
        return None
    if fit["family"] == "F7" and h.get("l2_cache_MiB", 0) <= 0:
        fit = cohort["fallback_models"][fit_key]
        fallback_reason = "Physical L2 capacity unspecified; use fitted additive fallback."
    n = batch if name in ("lm_head", "sampler") else batch*tokens
    shape = dict(pc=batch*tokens if phase == "prefill" else 0,
                 kp=max(0, context-tokens)+tokens/2 if phase == "prefill" else 0,
                 nd=batch if phase == "decode" else 0,
                 kv=max(0, context-1) if phase == "decode" else 0)
    fit_work = work_features(name,m,h,effective_tp,n,precision=config["precision"],
                             kv_precision=config["kv_precision"],explain=True,**shape)
    features = fit_work["features"]
    arithmetic_floor = features[1]*h.get("compute_efficiency",.65)/.65
    extended = shape_features(name,m,h,effective_tp,n,features,precision=config["precision"],
                              query_tokens=tokens if cohort_name=="vidur" and phase=="prefill" else None,**shape)
    ns = evaluate(fit, extended, n, arithmetic_floor)
    ns *= .97/{"vllm":.97,"trtllm":1.0,"eager":.8}[engine]
    ns += ({"vllm":2.5,"trtllm":2.0,"eager":8.0}[engine]-2.5)*1000
    ns += h.get("launch_overhead_us",0)*1000
    # Bounds are descriptive coverage, not a claim that unseen combinations
    # inside a bounding box have been measured or independently validated.
    covered = False
    for domain in data["domains"]:
        if (domain["hardware_id"] != pool["hardware_id"] or domain["model_id"] != config["model_id"]
                or domain["tp"] != effective_tp or engine != "vllm"
                or domain.get("cohort", "kaist") != cohort_name
                or config["precision"] != domain.get("precision", "bf16")
                or config["kv_precision"] != domain.get("precision", "bf16")):
            continue
        if any(m.get(k,False) != domain["model"].get(k,False) for k in GEOMETRY):
            continue
        expected=dict(domain["hardware"],clock_scale=1,compute_efficiency=.65,memory_efficiency=.8,launch_overhead_us=0)
        if any(not math.isclose(h.get(k,expected.get(k,0)),expected.get(k,0),rel_tol=1e-10,abs_tol=1e-12)
               for k in (*RATE_FIELDS,config["precision"]+"_tensor_tflops","sram_cache_MB","num_sms","tensor_width","clock_scale","compute_efficiency","memory_efficiency","launch_overhead_us","l2_cache_MiB")):
            continue
        bounds=domain["bounds"].get(fit_key)
        query=shape if name=="attention" else {"n":n}
        if bounds and all(lo <= query[k] <= hi for k,(lo,hi) in bounds.items()):
            covered=True
            break
    return dict(duration_ms=max(1,arithmetic_floor,ns)/1e6,
                timing_source="fitted-profiled-configuration" if covered else "fitted-extrapolation",
                fit_key=fit_key, fit_features_ns=features, shape_features_ns=extended,
                fit_work=fit_work, fit_batch=batch, fit_tokens=tokens, fit_context=context,
                fit_family=fit["family"], fit_cohort=cohort_name, fit_fallback_reason=fallback_reason,
                fit_parameters={key:fit[key] for key in ("family","coefficients","scale","branches","threshold_tokens") if key in fit},
                fit_compute_floor_ns=arithmetic_floor, fit_base_ns=evaluate(fit,extended,n,arithmetic_floor),
                fit_engine_multiplier=.97/{"vllm":.97,"trtllm":1.0,"eager":.8}[engine],
                fit_engine_offset_ns=({"vllm":2.5,"trtllm":2.0,"eager":8.0}[engine]-2.5)*1000,
                fit_extra_launch_ns=h.get("launch_overhead_us",0)*1000, fit_token_count=n,
                calibration_scope="Fitted GPU operator service time; serving/network/energy remain modeled")


def manifest():
    d=dataset()
    return dict(revision=d["upstream_revision"],data_sha256=d["payload_sha256"],
                profiles=[dict(id=p["id"],model_id=p["model_id"],hardware_id=p["hardware_id"],tp=p["tp"],software=p["software"])
                          for p in d["domains"]],
                validation=d["validation"],
                model="Nested-CV selected shape-informed functions; separate KAIST BF16 / Vidur FP16 fits; no runtime timing-table lookup",
                accuracy_report=d["accuracy_report"], training_observations=d["training_observations"],
                functions={source:{op:model["family"] for op,model in cohort["models"].items()} for source,cohort in d["cohorts"].items()})
