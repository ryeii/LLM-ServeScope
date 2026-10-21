"""Behavioral prefix-KV tier model. No random hits or fabricated device measurements.

Retained caches use explicit single-tier, exclusive spill or inclusive-copy layouts,
separate from pinned active-request KV.
Publication is write-through before handoff; cache I/O shares explicit FIFO links.
"""
from collections import OrderedDict, Counter
from functools import lru_cache
from pathlib import Path
import copy, hashlib, json, math
import simpy

CACHE_LAYOUTS = {
    'none': dict(label='No retained prefix cache', tiers=[], kind='single', description='Compute every prompt; retain no prefix blocks. Active request KV stays in HBM.'),
    'hbm-only': dict(label='HBM only', tiers=['hbm'], kind='single', description='Retain prefixes in GPU HBM; discard least-recently-used blocks when full.'),
    'dram-only': dict(label='DRAM only', tiers=['dram'], kind='single', description='Retain prefixes in host DRAM; restores and publication use the host–GPU link.'),
    'ssd-only': dict(label='SSD only', tiers=['ssd'], kind='single', description='Retain prefixes on SSD; charge SSD service and host–GPU transfers.'),
    'hbm-dram': dict(label='HBM → DRAM spill', tiers=['hbm','dram'], kind='spill', description='Place new blocks in HBM; spill LRU victims to DRAM, then discard when DRAM is full.'),
    'dram-hbm': dict(label='DRAM → HBM spill', tiers=['dram','hbm'], kind='spill', description='Place new blocks in DRAM; spill LRU victims to HBM, then discard when HBM is full.'),
    'hbm-ssd': dict(label='HBM → SSD spill', tiers=['hbm','ssd'], kind='spill', description='Place new blocks in HBM; spill LRU victims to SSD, then discard when SSD is full.'),
    'dram-ssd': dict(label='DRAM → SSD spill', tiers=['dram','ssd'], kind='spill', description='Place new blocks in DRAM; spill LRU victims to SSD, then discard when SSD is full.'),
    'ssd-hbm': dict(label='SSD → HBM spill', tiers=['ssd','hbm'], kind='spill', description='Place new blocks on SSD; spill LRU victims to HBM, then discard when HBM is full.'),
    'ssd-dram': dict(label='SSD → DRAM spill', tiers=['ssd','dram'], kind='spill', description='Place new blocks on SSD; spill LRU victims to DRAM, then discard when DRAM is full.'),
    'hbm-dram-ssd': dict(label='HBM → DRAM → SSD spill', tiers=['hbm','dram','ssd'], kind='spill', description='Place new blocks in HBM; cascade LRU victims through DRAM to SSD.'),
    'inclusive': dict(label='Custom inclusive copies · original policy', tiers=['hbm','dram','ssd'], kind='inclusive', description='Write every new block to every tier with nonzero capacity; restore from the fastest ready copy. This is replication, not spill.'),
}

def cache_capacity_GB(s, tier):
    return s[tier+'_cache_GB'] if s.get('enabled') and tier in CACHE_LAYOUTS[s.get('placement','inclusive')]['tiers'] else 0

DEFAULT_STORAGE=dict(enabled=False,placement='inclusive',profile='manual',hbm_cache_GB=0.125,dram_cache_GB=0.25,ssd_cache_GB=4,
                     ssd_read_GBps=3,ssd_write_GBps=2,ssd_read_latency_us=100,ssd_write_latency_us=200,
                     pcie_GBps=24,pcie_latency_us=10,io_chunk_KiB=1024,block_tokens=512,
                     source_id='synthetic-demo',source_label='Generated repeated-prefix workload',workload=[])

@lru_cache(maxsize=1)
def fit_dataset():
    p=Path(__file__).with_name('storage-fit.json')
    if not p.exists():return None
    data=json.loads(p.read_text());payload=dict(data);expected=payload.pop('payload_sha256')
    actual=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if actual!=expected:raise ValueError('Storage fit checksum mismatch.')
    return data

def latency_ms(fit,nbytes):
    x=nbytes/1e6;c=fit['coefficients'];f=fit['family']
    y=c[0]+c[1]*x
    if f=='sqrt-affine':y+=c[2]*math.sqrt(x)
    if f=='two-slope':y+=c[2]*max(0,x-.25)
    return max(1e-9,y)

def hit_prediction(fit,capacity_blocks,unique_blocks,max_hit_fraction=1):
    a,s=fit['coefficients'][:2];x=capacity_blocks/max(1,unique_blocks)
    if fit['family']=='capped-linear':return max_hit_fraction*a*min(1,x/s)
    p=fit['coefficients'][2] if fit['family']=='weibull' else 1
    return max_hit_fraction*a*(-math.expm1(-min(700,(x/s)**p)))

def validate_storage(c,number):
    raw=c.get('storage') or {}
    if not isinstance(raw,dict):raise ValueError('storage must be an object.')
    s=dict(copy.deepcopy(DEFAULT_STORAGE),**raw);c['storage']=s
    if type(s['enabled']) is not bool:raise ValueError('KV storage enabled must be boolean.')
    if s['placement'] not in CACHE_LAYOUTS:raise ValueError('Unknown KV cache layout.')
    if s['profile'] not in ('manual','local-mooncake-path'):raise ValueError('Unknown storage profile.')
    for k in ['hbm_cache_GB','dram_cache_GB','ssd_cache_GB']:
        s[k]=number(s[k],k,0,1e6)
    for k in ['ssd_read_GBps','ssd_write_GBps','pcie_GBps']:
        s[k]=number(s[k],k,.001,1e5)
    for k in ['ssd_read_latency_us','ssd_write_latency_us','pcie_latency_us']:
        s[k]=number(s[k],k,0,1e6)
    s['io_chunk_KiB']=number(s['io_chunk_KiB'],'I/O chunk KiB',16,2048,True)
    if s['block_tokens']!=512:raise ValueError('Prefix storage currently uses 512-token blocks.')
    if not s['enabled'] and not s.get('workload') and c.get('active_kv',{}).get('placement','hbm')=='hbm':return
    phases=['prefill','decode'] if c.get('separated',True) else ['prefill']
    for phase in phases:
        if number(c[phase].get('replicas'),phase+'.replicas',1,64,True)!=1:
            raise ValueError('Retained prefix mode currently supports one replica per pool. Change replicas explicitly; selecting a layout never changes scheduling.')
    if s['profile']=='local-mooncake-path' and not fit_dataset():raise ValueError('Local storage fit unavailable.')
    rows=s.get('workload',[])
    if not isinstance(rows,list) or len(rows)>128:raise ValueError('Workload must contain at most 128 request records.')
    if rows:
        normalized=[]
        for i,row in enumerate(rows):
            if not isinstance(row,dict):raise ValueError(f'Workload record {i} must be an object.')
            timestamp=number(row.get('timestamp'),f'Workload timestamp #{i}',0,1e12)
            length=number(row.get('input_length'),f'Workload input #{i}',1,131072,True)
            output=number(row.get('output_length'),f'Workload output #{i}',1,4096,True)
            keys=row.get('hash_ids')
            if not isinstance(keys,list) or len(keys)>math.ceil(length/512):raise ValueError(f'Invalid prefix hash list at workload record {i}.')
            keys=[number(key,'Prefix hash',0,2**53-1,True) for key in keys]
            normalized.append(dict(timestamp=timestamp,input_length=length,output_length=output,hash_ids=keys,source_line=row.get('source_line',i+1)))
        if any(normalized[i]['timestamp']<normalized[i-1]['timestamp'] for i in range(1,len(normalized))):raise ValueError('Replay records must be in timestamp order; tied arrivals retain source order.')
        s['workload']=normalized;c['requests']=len(rows);c['input_tokens']=max(r['input_length'] for r in rows);c['output_tokens']=max(r['output_length'] for r in rows)
    if rows and len({(r['input_length'],r['output_length']) for r in s['workload']})>1:
        for phase in phases:
            if number(c[phase].get('batch_size',c['batch_size']),phase+'.batch_size',1,32,True)!=1:
                raise ValueError('Variable-length replay currently requires batch size 1 per pool. Set it explicitly; uniform-length requests support fixed batching.')
    for key in ['source_id','source_label']:
        if not isinstance(s[key],str) or len(s[key])>200:raise ValueError('Invalid workload source label.')
    s['source_sha256']=hashlib.sha256(json.dumps(s['workload'],sort_keys=True,separators=(',',':')).encode()).hexdigest()

def workload(c):
    s=c.get('storage',{})
    if not s.get('enabled') and not s.get('workload') and c.get('active_kv',{}).get('placement','hbm')=='hbm':return None
    if s['workload']:
        origin=s['workload'][0]['timestamp']
        return [dict(row,arrival_ms=row['timestamp']-origin) for row in s['workload']]
    return [dict(timestamp=i/c['arrival_rate']*1000,arrival_ms=i/c['arrival_rate']*1000,input_length=c['input_tokens'],output_length=c['output_tokens'],hash_ids=[(i%4)*100000+k for k in range(c['input_tokens']//512)],source_line=i+1) for i in range(c['requests'])]

class CacheTiers:
    def __init__(self,env,c,m,record):
        self.env,self.c,self.s,self.m,self.record=env,c,c['storage'],m,record
        p=c['prefill'];d={'bf16':2,'fp16':2,'fp8':1}[c['kv_precision']]
        self.per_gpu_block_bytes=2*512*max(1,m['kv_heads']/p['tp'])*m['head_dim']*d*m['layers']/p['pp']
        self.block_bytes=self.per_gpu_block_bytes*p['tp']*p['pp']*p['ep']
        self.placement=self.s['placement'] if self.s['enabled'] else 'none'
        self.layout=CACHE_LAYOUTS[self.placement]
        self.capacity={tier:int(cache_capacity_GB(self.s,tier)*1e9/(self.per_gpu_block_bytes if tier=='hbm' else self.block_bytes)) for tier in ['hbm','dram','ssd']}
        self.placement_lock=simpy.Resource(env,1)
        self.transitions=[]
        self.caches={k:OrderedDict() for k in self.capacity};self.pins={k:Counter() for k in self.capacity};self.pending={}
        self.links={k:simpy.Resource(env,1) for k in ['ssd','pcie','hbm']}
        self.stats=Counter();self.active_io_stats=Counter();self.peaks=Counter();self.evictions=Counter();self.drops=Counter();self.events=[];self.snapshots=[]
        self.fit=fit_dataset() if self.s['profile']=='local-mooncake-path' and (self.capacity['ssd'] or c.get('active_kv',{}).get('placement') in ('ssd','hbm-ssd')) else None

    def keys(self,row):
        parent='';keys=[]
        for h in row['hash_ids'][:row['input_length']//512]:
            parent=hashlib.sha256((parent+':'+str(h)).encode()).hexdigest();keys.append(parent)
        return keys

    def _put(self,tier,key,request):
        cache=self.caches[tier];limit=self.capacity[tier]
        if not limit:return
        if key in cache:cache.move_to_end(key);return
        while len(cache)>=limit:
            victim=next((k for k in cache if not self.pins[tier][k]),None)
            if victim is None:self.drops[tier]+=1;return
            cache.pop(victim);self.evictions[tier]+=1
        cache[key]=None;self.peaks[tier]=max(self.peaks[tier],len(cache))
        self.snapshots.append(dict(time_ms=self.env.now,tier=tier,blocks=len(cache),request_id=request))

    def _io(self,tier,op,nbytes,identifier,cause,scope='prefix',model_phase='prefill',**metadata):
        if nbytes<=0:return
        counter=self.stats if scope=='prefix' else self.active_io_stats
        ready=self.env.now;resource=self.links[tier]
        with resource.request() as ticket:
            yield ticket;begin=self.env.now
            chunk=self.s['io_chunk_KiB']*1024;full=int(nbytes//chunk);tail=nbytes-full*chunk
            if tier=='ssd':
                if self.fit:model=self.fit['io'][op]
                else:model=dict(family='affine',coefficients=[self.s['ssd_'+op+'_latency_us']/1000,1/self.s['ssd_'+op+'_GBps']])
                duration=full*latency_ms(model,chunk)+(latency_ms(model,tail) if tail else 0)
                extrapolated=bool(self.fit and ((full and not model['min_bytes']<=chunk<=model['max_bytes']) or (tail and not model['min_bytes']<=tail<=model['max_bytes'])))
                if extrapolated:counter['extrapolated_io_events']+=1
            else:
                bw=self.s['pcie_GBps'] if tier=='pcie' else self.c[model_phase]['hardware']['dram_bandwidth_TBps']*1000*self.c[model_phase]['tp']*self.c[model_phase]['pp']*self.c[model_phase]['ep']
                latency=self.s['pcie_latency_us']/1000 if tier=='pcie' else 0
                duration=nbytes/(bw*1e9)*1000+latency
                model=dict(family='affine',coefficients=[latency,1/bw]);full=0;tail=nbytes;extrapolated=False
            yield self.env.timeout(duration)
            event=dict(phase='storage',kind='storage_io',tier=tier,operation=op,cause=cause,resource='kv_'+tier+':0',batch_id=identifier,request_ids=[identifier],chunk=0,batch_size=1,context=0,start_ms=begin,end_ms=self.env.now,duration_ms=duration,payload_bytes=nbytes,service_model=model,profile=self.s['profile'] if tier=='ssd' else 'configured-link-assumption',chunk_bytes=chunk if tier=='ssd' else nbytes,full_chunks=full,tail_bytes=tail,extrapolated=extrapolated,components=dict(Compute=0,Memory=0,Network=0,Launch=0))
            event.update(scope=scope,model_phase=model_phase,**metadata)
            self.record(event,ready);self.events.append(event);counter[tier+'_'+op+'_bytes']+=nbytes;counter[tier+'_service_ms']+=duration;counter[tier+'_queue_ms']+=begin-ready
            return event

    def restore(self,row,identifier):
        keys=self.keys(row);selected=[]
        for key in keys:
            tier=next((t for t in self.layout['tiers'] if key in self.caches[t]),None)
            if tier is None:break
            selected.append((key,tier));self.pins[tier][key]+=1
            for t in self.caches:
                if key in self.caches[t]:self.caches[t].move_to_end(key)
        counts=Counter(t for _,t in selected)
        self.stats['eligible_blocks']+=len(keys);self.stats['hit_blocks']+=len(selected)
        try:
            for key,tier in selected:
                if tier=='ssd':yield from self._io('ssd','read',self.block_bytes,identifier,'restore')
                if tier!='hbm':yield from self._io('pcie','read',self.block_bytes,identifier,'restore')
                else:yield from self._io('hbm','read',self.block_bytes,identifier,'restore')
                self.stats[tier+'_hit_blocks']+=1
        finally:
            for key,tier in selected:self.pins[tier][key]-=1
        reused=min(len(selected)*512,row['input_length']-1)
        self.stats['reused_tokens']+=reused
        return dict(matched_blocks=len(selected),eligible_blocks=len(keys),reused_tokens=reused,hit_tiers=dict(counts),computed_tokens=row['input_length']-reused)

    def _copy_between(self,source,destination,identifier,cause):
        # Source 'active' denotes the pinned request's computed GPU KV.
        if source=='ssd':yield from self._io('ssd','read',self.block_bytes,identifier,cause)
        if destination=='hbm':
            if source in ('dram','ssd'):yield from self._io('pcie','read',self.block_bytes,identifier,cause)
            else:yield from self._io('hbm','write',self.block_bytes,identifier,cause)
        elif source in ('active','hbm'):
            yield from self._io('pcie','write',self.block_bytes,identifier,cause)
        if destination=='ssd':yield from self._io('ssd','write',self.block_bytes,identifier,cause)
        if cause=='spill':
            self.stats['spill_blocks']+=1
            self.stats[source+'_to_'+destination+'_spill_blocks']+=1
            self.transitions.append(dict(time_ms=self.env.now,source=source,destination=destination,request_id=identifier))

    def _admit_exclusive(self,key,index,identifier,source='active'):
        chain=self.layout['tiers']
        if index>=len(chain):
            self.stats['discarded_blocks']+=1
            return
        tier=chain[index]
        if not self.capacity[tier]:
            yield from self._admit_exclusive(key,index+1,identifier,source)
            return
        cache=self.caches[tier]
        if len(cache)>=self.capacity[tier]:
            victim=next((k for k in cache if not self.pins[tier][k]),None)
            if victim is None:
                self.drops[tier]+=1
                yield from self._admit_exclusive(key,index+1,identifier,source)
                return
            # Remove from lookup before moving. In-flight victims cannot be hit
            # until their destination copy finishes; pinned restores are skipped.
            cache.pop(victim);self.evictions[tier]+=1
            self.snapshots.append(dict(time_ms=self.env.now,tier=tier,blocks=len(cache),request_id=identifier))
            yield from self._admit_exclusive(victim,index+1,identifier,tier)
        yield from self._copy_between(source,tier,identifier,'publish' if source=='active' else 'spill')
        cache[key]=None
        self.peaks[tier]=max(self.peaks[tier],len(cache))
        self.snapshots.append(dict(time_ms=self.env.now,tier=tier,blocks=len(cache),request_id=identifier))

    def publish(self,row,identifier):
        for key in self.keys(row):
            if key in self.pending:
                self.stats['coalesced_publications']+=1;yield self.pending[key];continue
            done=self.env.event();self.pending[key]=done
            try:
                if self.layout['kind']!='inclusive':
                    with self.placement_lock.request() as ticket:
                        yield ticket
                        existing=next((t for t in self.layout['tiers'] if key in self.caches[t]),None)
                        if existing:self.caches[existing].move_to_end(key)
                        elif self.layout['tiers']:yield from self._admit_exclusive(key,0,identifier)
                    continue
                if self.capacity['hbm'] and key not in self.caches['hbm']:
                    yield from self._io('hbm','write',self.block_bytes,identifier,'publish')
                self._put('hbm',key,identifier)
                needs_dram=self.capacity['dram'] and key not in self.caches['dram']
                needs_ssd=self.capacity['ssd'] and key not in self.caches['ssd']
                if needs_dram or needs_ssd:yield from self._io('pcie','write',self.block_bytes,identifier,'publish')
                self._put('dram',key,identifier)
                if needs_ssd:yield from self._io('ssd','write',self.block_bytes,identifier,'publish')
                self._put('ssd',key,identifier)
            finally:
                done.succeed();self.pending.pop(key,None)

    def summary(self,rows):
        st=dict(self.stats);st['hit_fraction']=st.get('hit_blocks',0)/max(1,st.get('eligible_blocks',0));st['reused_fraction']=st.get('reused_tokens',0)/sum(r['input_length'] for r in rows)
        dataset=fit_dataset();source=self.s['source_id'];unique=len({key for row in rows for key in self.keys(row)});references=sum(len(self.keys(row)) for row in rows);potential=max(0,(references-unique)/max(1,references))
        forecast=None
        effective_capacity=max(self.capacity.values()) if self.layout['kind']=='inclusive' else sum(self.capacity.values())
        if dataset and source in dataset['hit']:
            f=dataset['hit'][source];forecast=dict(fit=f,working_set_blocks=unique,effective_capacity_blocks=effective_capacity,prediction=hit_prediction(f,effective_capacity,unique,potential),max_hit_fraction=potential,scope='Instantaneous-publication single-LRU trace-window approximation; uses summed exclusive-tier capacity or the largest inclusive-tier capacity. This does not validate the selected multi-tier replacement policy. Runtime queues, pending fills and short trace slices can differ; forecast is not used for cache decisions.',outside_request_range=not f['min_requests']<=len(rows)<=f['max_requests'])
        return dict(enabled=True,retention_enabled=self.s['enabled'],active_kv_location='hbm',active_kv_offload=False,retained_scope='prompt-prefix-reuse-between-requests',batching_policy='Fixed uniform microbatches use the minimum restored prefix of their members; excess restored tokens are recomputed.',placement=self.placement,layout=self.layout,spill_transitions=self.transitions,policy=self.layout['description']+' Cold start; active KV separately pinned; blocking publication/spill before handoff. Exclusive spill layouts keep one retained copy and do not promote hits; only ready contiguous prefixes can be reused. Last prompt token recomputed for logits.',profile=self.s['profile'],block_tokens=512,block_bytes=self.block_bytes,per_gpu_block_bytes=self.per_gpu_block_bytes,capacity_blocks=self.capacity,peak_blocks=dict(self.peaks),evictions=dict(self.evictions),admission_drops=dict(self.drops),stats=st,snapshots=self.snapshots,hit_forecast=forecast,fit=dataset if self.fit else None,workload_source=dict(id=source,label=self.s['source_label'],sha256=self.s['source_sha256'],source_file_sha256=self.s.get('source_file_sha256'),source_revision=self.s.get('source_revision'),requests=len(rows),classification='simulated-replay' if self.s['workload'] else 'generated-scenario'),limitations=['Local I/O fit measures warm buffered file reads and fsync writes on macOS, not raw SSD service, a production GPU server or Mooncake end-to-end performance.','One replica per pool. Fixed batching supports uniform input/output lengths; each prefill microbatch uses its minimum restored prefix length. Variable-length replay requires batch size one. No continuous/ragged batching.','HBM cache is explicitly reserved per GPU; retained copies and active request KV are separate allocations. Host/SSD capacities cover aggregate prefilling shard payloads.','Cache copies are not a NAND, NVMe or filesystem simulator. I/O chunks reserve a FIFO path; reads/writes share one SSD path; no packet or channel model.','GPU energy excludes SSD, CPU and transfer-device energy. Hit-fit validation is against trace-policy replay, not hardware cache telemetry.'])
