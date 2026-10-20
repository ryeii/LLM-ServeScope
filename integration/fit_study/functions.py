"""Small physically motivated function space, fitted without timing lookups."""
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.optimize import least_squares,nnls

FAMILIES={
 'F0':dict(name='Baseline additive / relative loss',parameters=3,formula='L + a C + b M',merge='sum',loss='relative'),
 'F1':dict(name='Additive / log loss',parameters=3,formula='L + a C + b M',merge='sum',loss='log'),
 'F2':dict(name='Fitted roofline',parameters=3,formula='L + max(a C, b M)',merge='max',loss='log'),
 'F3':dict(name='Smooth overlap',parameters=3,formula='L + sqrt((a C)^2 + (b M)^2)',merge='quadratic',loss='log'),
 'F4':dict(name='Tile-wave additive',parameters=3,formula='L + a W + b M',merge='wave',loss='log'),
 'F5':dict(name='One bounded shape transition',parameters=5,formula='L + a C + M [b0 (1-g) + b1 g]',merge='memory_gate',loss='log'),
 'F6':dict(name='Two bounded shape transitions',parameters=6,formula='L + C [a0 (1-g) + a1 g] + M [b0 (1-g) + b1 g]',merge='both_gate',loss='log'),
 'F7':dict(name='Cache-capacity transition',parameters=5,formula='L + a C + b M + d M g_cache',merge='cache_gate',loss='log'),
 'F8':dict(name='Corrected embedding geometry',parameters=3,formula='L + a C + b M_correct',merge='corrected',loss='relative')}
SCALES=(.125,.5,2.,8.,32.)

def basis(family,x,scale=1.):
    c,m,w,waves=x[:,1],x[:,2],x[:,3],x[:,4]
    g=waves/(waves+scale)
    if family=='F8': return np.column_stack((x[:,0],c,x[:,6]))
    if family=='F7':
        ratio=x[:,5];gc=ratio/(ratio+scale)
        return np.column_stack((x[:,0],c,m,m*gc))
    if family=='F4': return np.column_stack((x[:,0],w,m))
    if family=='F5': return np.column_stack((x[:,0],c,m*(1-g),m*g))
    if family=='F6': return np.column_stack((x[:,0],c*(1-g),c*g,m*(1-g),m*g))
    return x[:,:3]

def evaluate_array(family,coefficients,x,scale=1.):
    z=basis(family,x,scale)*np.asarray(coefficients)
    if family=='F2': pred=z[:,0]+np.maximum(z[:,1],z[:,2])
    elif family=='F3': pred=z[:,0]+np.hypot(z[:,1],z[:,2])
    else: pred=np.sum(z,axis=1)
    return np.maximum.reduce((np.ones(len(x)),x[:,1],pred))

def fit_fixed(family,x,y,scale=1.):
    b=basis(family,x,scale)
    if family in ('F0','F8'):
        # Preserve the baseline's exact active-set algorithm and tie behavior.
        relative=b/y[:,None];norm=np.maximum(np.linalg.norm(relative,axis=0),1e-30);design=relative/norm
        best=np.zeros(3);loss=float('inf')
        for size in range(1,4):
            for subset in itertools.combinations(range(3),size):
                values=np.linalg.lstsq(design[:,subset],np.ones(len(y)),rcond=None)[0]
                if np.any(values < -1e-10):continue
                trial=np.zeros(3);trial[list(subset)]=np.maximum(0,values)/norm[list(subset)]
                candidate=np.mean((relative@trial-1)**2)
                if candidate<loss:best,loss=trial,candidate
        return dict(family=family,coefficients=best.tolist(),scale=scale)
    # Parameter scaling keeps ns intercept and dimensionless slopes well conditioned.
    normalizers=np.maximum(np.sqrt(np.mean((b/y[:,None])**2,axis=0)),1e-15)
    design=b/normalizers
    relative=design/y[:,None]
    initial=nnls(relative,np.ones(len(y)))[0]
    if family in ('F0','F8'):
        return dict(family=family,coefficients=(initial/normalizers).tolist(),scale=scale)
    def residual(z):
        pred=evaluate_array(family,z/normalizers,x,scale)
        return np.log(pred/y)
    best=None
    # One deterministic nonzero start plus a second for overlap functions; no
    # target-specific/manual parameter editing. Floors are evaluated inside loss.
    starts=[np.maximum(initial,1e-5)]
    if family in ('F2','F3'): starts.append(np.maximum(initial,1/len(initial)))
    for start in starts:
        result=least_squares(residual,start,bounds=(0,np.inf),max_nfev=160,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        loss=float(np.mean(result.fun**2))
        if best is None or loss<best[0]: best=(loss,result)
    result=best[1]
    return dict(family=family,coefficients=(result.x/normalizers).tolist(),scale=scale,
                optimizer_success=bool(result.success),optimizer_nfev=int(result.nfev),training_log_mse=best[0])

def predict(model,x): return evaluate_array(model['family'],model['coefficients'],x,model['scale'])

def fit(family,rows):
    x=np.array([r['x'] for r in rows]);y=np.array([r['ns'] for r in rows])
    if family not in ('F5','F6','F7'): return fit_fixed(family,x,y)
    # Select the transition location inside this training set. Outer validation
    # targets never choose this scale, even during candidate family evaluation.
    folds=np.array([r['inner_fold'] for r in rows]); candidates=[]
    for scale in SCALES:
        errors=[]
        for fold in range(3):
            train=folds!=fold;test=~train
            if not np.any(test) or not np.any(train): continue
            m=fit_fixed(family,x[train],y[train],scale)
            errors.extend(np.abs(predict(m,x[test])/y[test]-1).tolist())
        candidates.append((float(np.mean(errors)),scale))
    _,scale=min(candidates)
    model=fit_fixed(family,x,y,scale)
    model['scale_inner_cv']=[dict(scale=s,mape_pct=e*100) for e,s in candidates]
    return model

def metrics(y,pred,ops=None):
    y=np.asarray(y);pred=np.asarray(pred);err=np.abs(pred/y-1)*100
    result=dict(count=len(y),mape_pct=float(np.mean(err)),median_ape_pct=float(np.median(err)),
      p95_ape_pct=float(np.percentile(err,95)),latency_weighted_ape_pct=float(np.sum(abs(pred-y))/np.sum(y)*100),
      signed_bias_pct=float(np.mean((pred/y-1)*100)),rms_log_error=float(np.sqrt(np.mean(np.log(pred/y)**2))))
    if ops is not None:
        result['macro_operator_mape_pct']=float(np.mean([np.mean(err[np.asarray(ops)==op]) for op in sorted(set(ops))]))
    return result
