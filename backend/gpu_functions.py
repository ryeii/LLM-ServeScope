"""Pure-Python shape-informed GPU functions shared by local Python and Pyodide.

No fitting, profile targets, GPU/model-ID corrections, NumPy or SciPy at runtime.
"""
import math


def shape_features(name,m,h,tp,n,base,*,pc=0,kp=0,nd=0,kv=0,precision='bf16',query_tokens=None):
    c,mem=base[1:3];H=m['hidden'];I=m['intermediate'];Q=m['heads']*m['head_dim'];K=m['kv_heads']*m['head_dim'];S=h['num_sms']
    widths={'qkv_proj':(Q+2*K)/tp,'o_proj':H,'gate_up_proj':2*I/tp,'down_proj':H,'lm_head':m['vocab']/tp}
    if name in widths:
        width=widths[name];blocks=math.ceil(n/128)*math.ceil(width/128)
        padded=(math.ceil(n/128)*128/n)*(math.ceil(width/128)*128/width)
        wave=c*padded*max(1,S/blocks);waves=n*width/(128*128*S)
    elif name=='attention':
        width=m['heads']/tp;q=query_tokens or (pc if pc else 1);seq=pc/q if pc else nd
        blocks=seq*math.ceil(q/128)*width;padded=math.ceil(q/128)*128/q if pc else 1
        wave=c*padded*max(1,S/max(1,blocks));waves=seq*width*max(1,q/128)/S
    else:
        width=I/tp if name=='act_fn' else (Q+K)/tp if name in ('rotary_emb','qk_norm') else m['vocab'] if name=='sampler' else H/(tp if name=='embedding' else 1)
        blocks=math.ceil(n*width/1024);wave=mem*max(1,S/blocks);waves=n*width/(1024*S)
    cache=h.get('l2_cache_MiB',0)*1024**2
    # Remove the scenario efficiency multiplier: nominal capacity pressure is bytes/cache.
    nominal_bytes=mem*h['dram_bandwidth_TBps']*1000*h.get('memory_efficiency',.8)/.8
    ratio=nominal_bytes/cache if cache>0 else 0.
    dtype={'fp4':.5,'fp8':1,'int8':1,'bf16':2,'fp16':2,'tf32':4,'fp32':4}[precision]
    corrected=(n*(2*dtype*H+8)/(h['dram_bandwidth_TBps']*1000)*.8/h.get('memory_efficiency',.8)) if name=='embedding' else mem
    return [1.,c,mem,wave,waves,ratio,corrected]


def evaluate(model,x,n,compute_floor=None):
    family=model['family'];c,m,w,waves,ratio,corrected=x[1:]
    floor=c if compute_floor is None else compute_floor
    if family=='F9':
        branch=model['branches']['small_index' if n<=model['threshold_tokens'] else 'gather']
        return max(1.,floor,branch[0]+branch[1]*corrected)
    scale=model.get('scale',1.);g=waves/(waves+scale)
    if family=='F8':basis=[1.,c,corrected]
    elif family=='F7':basis=[1.,c,m,m*ratio/(ratio+scale)]
    elif family=='F4':basis=[1.,w,m]
    elif family=='F5':basis=[1.,c,m*(1-g),m*g]
    elif family=='F6':basis=[1.,c*(1-g),c*g,m*(1-g),m*g]
    else:basis=[1.,c,m]
    z=[a*b for a,b in zip(model['coefficients'],basis)]
    if family=='F2':pred=z[0]+max(z[1],z[2])
    elif family=='F3':pred=z[0]+math.hypot(z[1],z[2])
    else:pred=sum(z)
    return max(1.,floor,pred)
