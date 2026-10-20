"""Fit a GPU service-time function offline and validate withheld hardware/model/shapes.

Run with .venv-sim/bin/python integration/build_calibration.py. No GPU required.
Only uniform dense decoder profiles supported by ServeScope are exported. Fused
MoE/MLA/hybrid profiles are not silently assigned to incompatible operators.
"""
from __future__ import annotations
import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".vendor/LLMServingSim"
sys.path[:0] = [str(ROOT), str(UPSTREAM)]
from backend.catalog import HARDWARE, MODELS
from backend.calibration import prediction, work_features
from integration.profile_reference import lookup
from serving.core import trace_generator as native
native._PROFILER_ROOT_REL = str(UPSTREAM / "profiler")

PIN = "b09c7aaab4b072d73136641841cc2688616c49eb"
BUNDLES = [("RTXPRO6000", "rtx6000", "llama8", 1),
           ("RTXPRO6000", "rtx6000", "llama8", 2),
           ("RTX4090", "rtx4090", "llama8", 1),
           ("RTXPRO6000", "rtx6000", "qwen32", 1),
           ("RTXPRO6000", "rtx6000", "qwen32", 2)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_nonnegative(rows):
    """Three-term nonnegative relative least squares; enumerate all active sets.

    L + a * ideal-compute-ns + b * nominal-memory-ns. Coefficients describe
    empirical scaling, not a physical decomposition of kernel elapsed time.
    """
    x = np.array([r["features"] for r in rows], dtype=float)
    y = np.array([r["ns"] for r in rows], dtype=float)
    x = x/y[:, None]
    target = np.ones(len(y))
    scales = np.maximum(np.linalg.norm(x, axis=0), 1e-30)
    design = x/scales
    best, loss = np.zeros(3), float("inf")
    for size in range(1, 4):
        for subset in itertools.combinations(range(3), size):
            values = np.linalg.lstsq(design[:, subset], target, rcond=None)[0]
            if np.any(values < -1e-10):
                continue
            trial = np.zeros(3)
            trial[list(subset)] = np.maximum(0, values)/scales[list(subset)]
            candidate = np.mean((x @ trial-target)**2)
            if candidate < loss:
                best, loss = trial, candidate
    return best.tolist()


def fit_all(rows):
    return {key: fit_nonnegative([r for r in rows if r["fit_key"] == key])
            for key in sorted({r["fit_key"] for r in rows})}


def summary(errors):
    a = np.abs(np.array(errors, dtype=float))*100
    return dict(count=len(a), mean_abs_relative_difference_pct=float(a.mean()),
                median_abs_relative_difference_pct=float(np.median(a)),
                p95_abs_relative_difference_pct=float(np.percentile(a, 95)))


def main():
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip()
    assert actual == PIN
    assert not subprocess.check_output(["git", "diff", "--ignore-submodules"], cwd=UPSTREAM, text=True)
    profiles, sources, observations = {}, {}, []
    parity = dict(lookups=0, max_absolute_difference_ns=0)
    for gpu, hardware_id, model_id, tp in BUNDLES:
        m, h = copy.deepcopy(MODELS[model_id]), copy.deepcopy(HARDWARE[hardware_id])
        if hardware_id == "rtx6000":
            h["dram_bandwidth_TBps"] = 1.5976
        model = m["source"]
        db = native._load_perf_db(gpu, model, "bf16", {tp}, "llama" if model_id == "llama8" else "qwen3")
        profile_id = f"{hardware_id}:{model_id}:tp{tp}"
        profile = dict(id=profile_id, hardware_id=hardware_id, model_id=model_id,
                       tp=tp, hardware=h, model=m,
                       software={k: db["meta"].get(k) for k in ["vllm_version", "cuda_version", "profiled_at"]},
                       dense={}, per_sequence={}, attention={})
        for category in ("dense", "per_sequence"):
            for name in native._tp_tables(db, tp)[category]:
                effective_tp = native._effective_tp(db, category, name, tp)
                table = native._tp_tables(db, effective_tp)[category][name]
                points = [[int(x), int(y)] for x,y in zip(table["keys"], table["values"])]
                profile[category][name] = points
                for n, ns in points:
                    observations.append(dict(profile_id=profile_id, hardware_id=hardware_id,
                                             model_id=model_id, tp=tp, name=name, fit_key=name,
                                             n=n, phase="decode", shape=dict(pc=0,kp=0,nd=0,kv=0),
                                             features=work_features(name,m,h,tp,n), ns=ns))
        table = native._tp_tables(db, tp)["attention_by_layer"]["attention"][1]
        prefill, decode = [], []
        for pc, nd in table["pc_nd_pairs"]:
            if pc > 0 and nd == 0:
                sl = table["slices"][(pc, nd)]
                points = [[float(kp), int(native._attn_slice_lookup(table, pc, nd, kp, 0))]
                          for kp in sl["prefill_key_vals"]]
                prefill.append([pc, points])
                for kp, ns in points:
                    shape = dict(pc=pc,kp=kp,nd=0,kv=0)
                    observations.append(dict(profile_id=profile_id, hardware_id=hardware_id,
                                             model_id=model_id,tp=tp,name="attention",fit_key="attention:prefill",
                                             n=pc,phase="prefill",shape=shape,
                                             features=work_features("attention",m,h,tp,pc,**shape),ns=ns))
            elif pc == 0 and nd > 0:
                sl = table["slices"][(pc, nd)]
                row = sl["rows"][0]
                points = [[int(kv), int(ns)] for kv,ns in zip(row["keys"], row["values"])]
                decode.append([nd, points])
                for kv, ns in points:
                    shape = dict(pc=0,kp=0,nd=nd,kv=kv)
                    observations.append(dict(profile_id=profile_id,hardware_id=hardware_id,
                                             model_id=model_id,tp=tp,name="attention",fit_key="attention:decode",
                                             n=nd,phase="decode",shape=shape,
                                             features=work_features("attention",m,h,tp,nd,**shape),ns=ns))
        origin = native._lookup_attention(db,tp,0,0,0,0)
        profile["attention"] = dict(prefill=[[0, [[0, origin]]]]+sorted(prefill), decode=sorted(decode))
        profiles[profile_id] = profile
        for path in [Path(db["root"])/"meta.yaml", UPSTREAM/"profiler/models"/("llama.yaml" if model_id=="llama8" else "qwen3.yaml")]:
            sources[str(path.relative_to(UPSTREAM))] = digest(path)
        for directory in {tp,1}:
            for name in ["dense.csv", "per_sequence.csv", "attention.csv"]:
                path=Path(db["root"])/f"tp{directory}"/name
                sources[str(path.relative_to(UPSTREAM))] = digest(path)
        # Independent upstream parity oracle on grid and non-grid queries,
        # including below/above-grid and chunked-prefill histories.
        cases = [r for r in observations if r["profile_id"]==profile_id]
        for r in cases:
            ours = lookup(profile,r["name"],r["n"],r["phase"],**r["shape"])[0]
            if r["name"] == "attention":
                s=r["shape"]
                theirs=native._lookup_attention(db,tp,s["pc"],s["kp"],s["nd"],s["kv"])
            else:
                function=native._lookup_per_sequence if r["name"] in profile["per_sequence"] else native._lookup_dense
                theirs=function(db,r["name"],tp,r["n"])
            parity["lookups"] += 1
            parity["max_absolute_difference_ns"] = max(parity["max_absolute_difference_ns"],abs(ours-theirs))
            assert ours==theirs,(profile_id,r,ours,theirs)
        for phase in ("prefill","decode"):
            for n in (1,3,7,17,63,129,513,2049):
                for key in (0,31.5,129,2049,17000):
                    s=dict(pc=n,kp=key,nd=0,kv=0) if phase=="prefill" else dict(pc=0,kp=0,nd=n,kv=int(key))
                    ours=lookup(profile,"attention",n,phase,**s)[0]
                    theirs=native._lookup_attention(db,tp,s["pc"],s["kp"],s["nd"],s["kv"])
                    parity["lookups"]+=1
                    parity["max_absolute_difference_ns"]=max(parity["max_absolute_difference_ns"],abs(ours-theirs))
                    assert ours==theirs,(profile_id,s,ours,theirs)
    fits=fit_all(observations)
    training=[r for r in observations if r["hardware_id"]!="rtx4090"]
    heldout=[r for r in observations if r["hardware_id"]=="rtx4090"]
    transfer_fits=fit_all(training)
    errors=[]
    weighted_absolute_ns=total_measured_ns=0.0
    heldout_rows=[]
    by_operator={}
    for r in heldout:
        target=profiles[r["profile_id"]]
        fit=transfer_fits[r["fit_key"]]
        predicted=prediction(fit,r["features"])
        error=predicted/r["ns"]-1
        errors.append(error)
        weighted_absolute_ns+=abs(predicted-r["ns"])
        total_measured_ns+=r["ns"]
        heldout_rows.append(dict(operator=r["fit_key"],n=r["n"],phase=r["phase"],shape=r["shape"],measured_ns=r["ns"],predicted_ns=predicted))
        by_operator.setdefault(r["fit_key"],[]).append(error)
    # Compare the actual legacy operator implementation on the same held-out
    # transformer-block samples. Endpoint kernels had no legacy counterpart.
    sys.path.insert(0,str(ROOT/"third_party/simpy-4.1.1-py3-none-any.whl"))
    from backend.model import DEFAULT, validate, phase_ops
    from backend.calibration import OPERATOR_NAMES
    names={v:k for k,v in OPERATOR_NAMES.items()}
    raw_errors=[]
    calibrated_errors=[]
    raw_cache={}
    c=copy.deepcopy(DEFAULT)
    c.update(model_id="llama8",timing_mode="roofline",separated=False,requests=1,batch_size=1)
    c["prefill"].update(hardware_id="rtx4090",hardware=copy.deepcopy(HARDWARE["rtx4090"]),tp=1)
    c=validate(c)
    for r,predicted_row in zip(heldout,heldout_rows):
        if r["name"] not in names:
            continue
        if r["name"]=="attention":
            s=r["shape"]
            if r["phase"]=="decode":
                batch,tokens,context=s["nd"],1,s["kv"]+1
            else:
                if s["kp"] < s["pc"]/2:
                    continue  # Legacy single-sequence formula cannot encode this batch geometry.
                batch,tokens,context=1,s["pc"],s["kp"]+s["pc"]/2
        else:
            batch,tokens,context=1,r["n"],r["n"]
        key=(r["phase"],batch,tokens,context)
        if key not in raw_cache:
            raw_cache[key]={o["name"]:o for o in phase_ops(c,r["phase"],batch,tokens,context)[0]}
        raw=raw_cache[key][names[r["name"]]]["duration_ms"]*1e6
        raw_errors.append(raw/r["ns"]-1)
        calibrated_errors.append(predicted_row["predicted_ns"]/r["ns"]-1)
    validation=dict(training_data_extraction=parity,
                    held_out_gpu=dict(hardware="RTX4090",excluded_from_fit=True,
                                      summary=summary(errors),
                                      latency_weighted_absolute_difference_pct=weighted_absolute_ns/total_measured_ns*100,
                                      same_block_samples=dict(roofline=summary(raw_errors),fitted_function=summary(calibrated_errors)),
                                      by_operator={k:summary(v) for k,v in by_operator.items()}),
                    fit_observations=len(observations),
                    scope="Operator latency comparisons, not measured end-to-end accuracy. Two GPU families; cross-vendor, quantization, MoE and clock changes are unvalidated extrapolations.")
    in_sample=[prediction(fits[r["fit_key"]],r["features"])/r["ns"]-1 for r in observations]
    validation["training_fit"]=summary(in_sample)
    def validate_split(train,test):
        coefficients=fit_all(train)
        errors=[prediction(coefficients[r["fit_key"]],r["features"])/r["ns"]-1 for r in test if r["fit_key"] in coefficients]
        return dict(training_rows=len(train),tested_rows=len(errors),summary=summary(errors),
                    unsupported_operator_groups=sorted({r["fit_key"] for r in test}-{r["fit_key"] for r in train}))
    validation["held_out_model"]=dict(model="qwen32",excluded_from_fit=True,
        **validate_split([r for r in observations if r["model_id"]!="qwen32"],
                         [r for r in observations if r["model_id"]=="qwen32"]))
    def shape_fold(r):
        # Group identical operator/shape coordinates across ALL GPUs/models/TPs.
        # No corresponding measurement of that shape leaks into another split.
        shape=dict(fit_key=r["fit_key"],n=r["n"],shape=r["shape"])
        return int(hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest()[:8],16)%5
    validation["held_out_shapes"]=dict(split="SHA256(operator, tensor coordinates) mod 5 == 0 across devices/models/TPs",
        **validate_split([r for r in observations if shape_fold(r)!=0],
                         [r for r in observations if shape_fold(r)==0]))
    validation["scope"]="Direct fitted-function validation. No runtime table anchors or residual overrides. Two GPU families; empirical residual percentiles are not confidence bounds for an unseen chip."
    domains=[]
    for p in profiles.values():
        bounds={}
        for r in observations:
            if r["profile_id"]!=p["id"]:
                continue
            coordinates=r["shape"] if r["name"]=="attention" else {"n":r["n"]}
            entry=bounds.setdefault(r["fit_key"],{})
            for k,v in coordinates.items():
                entry[k]=[min(entry.get(k,[v,v])[0],v),max(entry.get(k,[v,v])[1],v)]
        domains.append({**{k:p[k] for k in ["id","hardware_id","model_id","tp","hardware","model","software"]},"bounds":bounds})
    provenance=dict(upstream_revision=PIN,source_files_sha256=sources,
                    exporter_sha256=digest(Path(__file__)),runtime_sha256=digest(ROOT/"backend/calibration.py"),
                    catalog_sha256=digest(ROOT/"backend/catalog.py"))
    training_payload=dict(profiles=profiles,**provenance)
    offline=ROOT/"third_party/kaist-dense-profile-grids.json.gz"
    offline.write_bytes(gzip.compress(json.dumps(training_payload,sort_keys=True,separators=(",",":")).encode(),mtime=0))
    payload=dict(schema_version="servescope.gpu-fit.v1",**provenance,
                 training_data_sha256=digest(offline),fits=fits,domains=domains,validation=validation,
                 fit_method="Nonnegative relative least squares: L+a*ideal_compute_ns+b*nominal_memory_ns, with arithmetic floor. Same function on every hardware and shape. No runtime timing-table lookup.")
    encoded=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    payload["payload_sha256"]=hashlib.sha256(encoded).hexdigest()
    out=ROOT/"backend/gpu-fit.json"
    out.write_text(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
    print(json.dumps(validation,indent=2))
    print(f"Wrote {out}: {out.stat().st_size} bytes")
    report=ROOT/"docs/calibration-validation.json"
    report.write_text(json.dumps(dict(validation=validation,fits=fits,
                                     source_files_sha256=sources,payload_sha256=payload["payload_sha256"],
                                     upstream_revision=PIN,held_out_gpu_samples=heldout_rows),indent=2)+"\n")


if __name__=="__main__":
    main()
