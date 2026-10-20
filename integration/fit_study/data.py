"""Deterministic profile adapters and shape features; raw units stay auditable."""
from pathlib import Path
import csv,gzip,hashlib,io,json,math,statistics
import numpy as np
from backend.calibration import work_features
ROOT=Path(__file__).resolve().parents[2]
VIDUR=ROOT/'third_party/vidur-profiles'

def hashfile(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def make_row(source,gpu,model_id,tp,m,h,op,n,ns,shape=None,query_tokens=None):
    shape=shape or dict(pc=0,kp=0,nd=0,kv=0)
    raw_op=op.split(':')[0]
    precision='bf16' if source=='kaist' else 'fp16'
    _,c,mem=work_features(raw_op,m,h,tp,n,precision=precision,**shape)
    S=h['num_sms'];H=m['hidden'];I=m['intermediate'];Q=m['heads']*m['head_dim'];K=m['kv_heads']*m['head_dim']
    widths={'qkv_proj':(Q+2*K)/tp,'o_proj':H,'gate_up_proj':2*I/tp,'down_proj':H,'lm_head':m['vocab']/tp}
    if raw_op in widths:
        width=widths[raw_op]
        blocks=math.ceil(n/128)*math.ceil(width/128)
        padded=(math.ceil(n/128)*128/n)*(math.ceil(width/128)*128/width)
        wave=c*padded*max(1,S/blocks)
        waves=n*width/(128*128*S)
    elif raw_op=='attention':
        width=m['heads']/tp
        q=query_tokens or (shape['pc'] if shape['pc'] else 1)
        seq=(shape['pc']/q) if shape['pc'] else shape['nd']
        blocks=seq*math.ceil(q/128)*width
        padded=math.ceil(q/128)*128/q if shape['pc'] else 1
        wave=c*padded*max(1,S/max(1,blocks))
        waves=seq*width*max(1,q/128)/S
    else:
        width=I/tp if raw_op=='act_fn' else (Q+K)/tp if raw_op in ('rotary_emb','qk_norm') else m['vocab'] if raw_op=='sampler' else H/(tp if raw_op=='embedding' else 1)
        blocks=math.ceil(n*width/1024)
        wave=mem*max(1,S/blocks)
        waves=n*width/(1024*S)
    geometry=[m.get(k,0) for k in ['hidden','intermediate','heads','kv_heads','head_dim','vocab']]
    # Canonical executed geometry, not model identity or nominal TP degree.
    # Replicated norms/sampler must remain together across all TP values.
    if raw_op in widths:
        reduction=Q/tp if raw_op=='o_proj' else I/tp if raw_op=='down_proj' else H
        executed=dict(M=n,N=widths[raw_op],K=reduction)
    elif raw_op=='attention':
        executed=dict(q_heads=m['heads']/tp,kv_heads=m['kv_heads']/tp,head_dim=m['head_dim'])
    elif raw_op in ('layernorm','final_layernorm'):
        executed=dict(rows=n,width=H)
    elif raw_op=='sampler':
        executed=dict(rows=n,width=m['vocab'])
    elif raw_op=='embedding':
        executed=dict(rows=n,width=H,vocabulary_shard=m['vocab']/tp)
    elif raw_op=='act_fn':
        executed=dict(rows=n,width=I/tp)
    else:
        executed=dict(rows=n,q_width=Q/tp,kv_width=K/tp,head_dim=m['head_dim'])
    coord=dict(op=op,executed=executed,n=n,shape=shape,query_tokens=query_tokens)
    key=json.dumps(coord,sort_keys=True,separators=(',',':'))
    digest=hashlib.sha256(key.encode()).hexdigest()
    return dict(source=source,gpu=gpu,model=model_id,tp=tp,op=op,n=n,shape=shape,
      query_tokens=query_tokens,geometry=geometry,key=key,fold=int(digest[:8],16)%5,
      inner_fold=int(digest[8:16],16)%3,ns=float(ns),x=[1.,c,mem,wave,waves,mem*h["dram_bandwidth_TBps"]*1000/max(1,h.get("l2_bytes",1)),
                    (n*(4*H+8)/(h["dram_bandwidth_TBps"]*1e12)*1e9) if raw_op=="embedding" else mem],repeat_count=1)

def kaist_rows():
    data=json.load(gzip.open(ROOT/'third_party/kaist-dense-profile-grids.json.gz','rt'))
    rows=[]
    for p in data['profiles'].values():
        h=dict(p['hardware'],l2_bytes=(72 if p['hardware_id']=='rtx4090' else 128)*1024**2)
        args=('kaist',p['hardware_id'],p['model_id'],p['tp'],p['model'],h)
        for category in ['dense','per_sequence']:
            for name,points in p[category].items():
                for n,ns in points: rows.append(make_row(*args,name,n,ns))
        for phase,sets in p['attention'].items():
            for n,points in sets:
                if n==0: continue
                for k,ns in points:
                    shape=dict(pc=n,kp=k,nd=0,kv=0) if phase=='prefill' else dict(pc=0,kp=0,nd=n,kv=k)
                    rows.append(make_row(*args,'attention:'+phase,n,ns,shape))
    return rows

# Vidur names the memory capacities but not the physical SKU/interface in CSV.
# These explicit reference inputs must not be confused with measured clocks/BW.
# Source URLs and alternative variants are recorded in the study report.
VIDUR_HARDWARE={
 'a100':dict(fp16_tensor_tflops=312,bf16_tensor_tflops=312,fp32_tflops=19.5,dram_bandwidth_TBps=2.039,num_sms=108,l2_bytes=40*1024**2),
 'a40':dict(fp16_tensor_tflops=149.7,bf16_tensor_tflops=149.7,fp32_tflops=37.4,dram_bandwidth_TBps=.696,num_sms=84,l2_bytes=6*1024**2),
 'h100':dict(fp16_tensor_tflops=989,bf16_tensor_tflops=989,fp32_tflops=67,dram_bandwidth_TBps=3.35,num_sms=132,l2_bytes=50*1024**2)}
MAP={'emb':'embedding','input_layernorm':'layernorm','attn_pre_proj':'qkv_proj','attn_rope':'rotary_emb',
 'attn_post_proj':'o_proj','mlp_up_proj':'gate_up_proj','mlp_act':'act_fn','mlp_down_proj':'down_proj'}

def read_archived(path):
    meta=json.loads((VIDUR/'manifest.json').read_text())
    entry=next(f for f in meta['files'] if f['path']==path)
    raw=gzip.decompress((VIDUR/entry['archive']).read_bytes())
    assert hashlib.sha256(raw).hexdigest()==entry['sha256'],path
    return raw.decode()

def vidur_rows():
    meta=json.loads((VIDUR/'manifest.json').read_text()); rows=[]; exclusions={}
    for entry in meta['files']:
        path=entry['path']
        if not path.endswith('.csv'): continue
        parts=path.split('/'); gpu=parts[3];model='/'.join(parts[4:-1])
        for record in csv.DictReader(io.StringIO(read_archived(path))):
            attention=path.endswith('attention.csv')
            heads=int(record['n_q_head'] if attention else record['n_head'])
            H=int(record['n_embd']); tp=int(record['num_tensor_parallel_workers'])
            # Attention CSV omits FFN/vocab. Use known Llama/CodeLlama geometry;
            # these fields do not enter attention arithmetic/memory features.
            I=int(record.get('n_expanded_embd') or (11008 if H==4096 else 28672 if H==8192 else 22016))
            vocab=int(record.get('vocab_size') or 32000)
            m=dict(hidden=H,intermediate=I,heads=heads,kv_heads=int(record['n_kv_head']),head_dim=H//heads,vocab=vocab,experts=1,top_k=1)
            args=('vidur',gpu,model,tp,m,VIDUR_HARDWARE[gpu])
            if attention:
                pre=record['is_prefill']=='True';b=int(record['batch_size']);q=int(record['prefill_chunk_size']);kv=int(record['kv_cache_size'])
                op='attention:prefill' if pre else 'attention:decode'
                col='time_stats.attn_prefill.mean' if pre else 'time_stats.attn_decode.mean'
                if not record.get(col): exclusions['blank_inactive_attention']=exclusions.get('blank_inactive_attention',0)+1;continue
                ns=float(record[col])*1e6
                shape=dict(pc=b*q,kp=kv+q/2,nd=0,kv=0) if pre else dict(pc=0,kp=0,nd=b,kv=kv)
                if ns<=0: exclusions['nonpositive_time']=exclusions.get('nonpositive_time',0)+1;continue
                rows.append(make_row(*args,op,b*q if pre else b,ns,shape,query_tokens=q if pre else 1))
            else:
                if record['use_gated_mlp']!='True': raise ValueError('Adapter supports gated dense MLP only')
                for key,op in MAP.items():
                    ns=float(record['time_stats.'+key+'.mean'])*1e6
                    if ns<=0: exclusions['nonpositive_time']=exclusions.get('nonpositive_time',0)+1;continue
                    rows.append(make_row(*args,op,int(record['num_tokens']),ns))
    # Deduplicate repeated published coordinates with median of the recorded means.
    grouped={}
    for r in rows:
        # Preserve independent TP/model measurements; collapse only repeated
        # exact published coordinates. Fold grouping is deliberately broader.
        record=(r['gpu'],r['model'],r['tp'],r['op'],r['n'],json.dumps(r['shape'],sort_keys=True),r['query_tokens'])
        grouped.setdefault(record,[]).append(r)
    out=[]
    for group in grouped.values():
        r=dict(group[0]);r['ns']=statistics.median(x['ns'] for x in group);r['repeat_count']=len(group);out.append(r)
    return sorted(out,key=lambda r:(r['gpu'],r['op'],r['key'])),dict(raw_observations=len(rows),unique_observations=len(out),exclusions=exclusions)

def arrays(rows): return np.array([r['x'] for r in rows]),np.array([r['ns'] for r in rows])
