"""Active decode-KV backing: pinned residency plus serialized staged attention.

HBM-resident operator profiles remain the GPU reference. Remote KV adds explicit
serial tile-transfer stalls. Streaming-kernel overhead/overlap is uncalibrated.
Prefill working KV stays in HBM; retained-prefix policy is independent.
"""
import copy, hashlib
from pathlib import Path
from collections import Counter
import simpy
from .kv_transfer import active_paths

PLACEMENTS = {
    'hbm': dict(label='HBM only', backing=None, description='Active decode KV remains fully resident in GPU HBM.'),
    'dram': dict(label='Host DRAM backing', backing='dram', description='Keep active decode KV in host DRAM; stage its working set for every token.'),
    'ssd': dict(label='SSD backing', backing='ssd', description='Keep active decode KV on SSD; read and stage its working set for every token.'),
    'hbm-dram': dict(label='HBM + DRAM spill', backing='dram', description='Pin an HBM allocation at admission and back the remaining active KV in DRAM.'),
    'hbm-ssd': dict(label='HBM + SSD spill', backing='ssd', description='Pin an HBM allocation at admission and back the remaining active KV on SSD.'),
}
DEFAULT_ACTIVE = dict(placement='hbm', hbm_resident_GB=1.0, staging_GB=0.0625,
                      dram_total_GB=128.0, ssd_total_GB=1024.0)

def offloaded(c):
    return c.get('active_kv', {}).get('placement', 'hbm') != 'hbm'

def validate_active(c, number):
    raw = c.get('active_kv', {})
    if not isinstance(raw, dict):
        raise ValueError('active_kv must be an object.')
    a = dict(copy.deepcopy(DEFAULT_ACTIVE), **raw)
    c['active_kv'] = a
    if a['placement'] not in PLACEMENTS:
        raise ValueError('Unknown active decode KV placement.')
    for key in ['hbm_resident_GB', 'dram_total_GB', 'ssd_total_GB']:
        a[key] = number(a[key], 'active_kv.'+key, 0, 1e6)
    a['staging_GB'] = number(a['staging_GB'], 'active_kv.staging_GB', .000001, 1e4)
    if offloaded(c):
        storage = c.get('storage') or {}
        if not isinstance(storage, dict):
            raise ValueError('storage must be an object.')
        chunk = number(storage.get('io_chunk_KiB', 1024), 'I/O chunk KiB', 16, 2048, True)
        if a['staging_GB']*1e9 < chunk*1024:
            raise ValueError('Active KV GPU staging space must fit at least one configured I/O chunk per GPU.')

class ActiveKV:
    def __init__(self, env, c, model, rows, tiers, resident_capacity_bytes):
        self.env, self.c, self.a = env, c, c['active_kv']
        self.p, self.model, self.rows, self.tiers = c['decode'], model, rows, tiers
        self.mode = self.a['placement']
        self.backing = PLACEMENTS[self.mode]['backing']
        self.hybrid = self.mode.startswith('hbm-')
        self.shards = self.p['tp']*self.p['pp']*self.p['ep']
        dtype = {'bf16':2, 'fp16':2, 'fp8':1}[c['kv_precision']]
        self.bytes_per_token_gpu = 2*max(1, model['kv_heads']/self.p['tp'])*model['head_dim']*dtype*model['layers']/self.p['pp']
        self.hbm_capacity = resident_capacity_bytes if self.hybrid else 0
        prefix = c['storage']
        prefix_dram = prefix['dram_cache_GB']*1e9 if prefix['enabled'] and 'dram' in tiers.layout['tiers'] else 0
        prefix_ssd = prefix['ssd_cache_GB']*1e9 if prefix['enabled'] and 'ssd' in tiers.layout['tiers'] else 0
        relay = prefix['io_chunk_KiB']*1024 if self.backing=='ssd' and prefix['ssd_transfer_mode']=='host-staged' else 0
        self.partition = dict(prefix_dram_bytes=prefix_dram, prefix_ssd_bytes=prefix_ssd,
                              relay_dram_bytes=relay, dram_total_bytes=self.a['dram_total_GB']*1e9,
                              ssd_total_bytes=self.a['ssd_total_GB']*1e9)
        available = {'dram':self.a['dram_total_GB']*1e9-prefix_dram-relay,
                     'ssd':self.a['ssd_total_GB']*1e9-prefix_ssd}
        if available[self.backing] < 0 or available['dram'] < 0 or available['ssd'] < 0:
            raise ValueError('Active KV device capacity must cover retained-prefix reservations and the SSD relay chunk.')
        self.remote_capacity = available[self.backing]
        self.used_hbm = self.used_remote = self.peak_hbm = self.peak_remote = 0
        self.holders, self.allocations, self.ledger = {}, {}, []
        self.stats = Counter()
        self.changed = env.event()
        # Reject impossible atomic groups before running a schedule that could stall forever.
        n = c['prefill']['batch_size']
        for start in range(0, len(rows), n):
            group = list(range(start, min(len(rows), start+n)))
            required = sum(self.maximum_bytes(i) for i in group)
            if max(0, required-self.hbm_capacity)*self.shards > self.remote_capacity+1e-6:
                raise ValueError('An active decode KV microbatch cannot fit its HBM plus backing-store capacity; increase capacity or reduce batch/context.')

    def maximum_bytes(self, identifier):
        row = self.rows[identifier]
        return self.bytes_per_token_gpu*(row['input_length']+row['output_length']-1) if row['output_length']>1 else 0

    def admit(self, ids):
        ready = self.env.now
        needed = sum(self.maximum_bytes(i) for i in ids)
        while True:
            hbm = min(needed, max(0, self.hbm_capacity-self.used_hbm))
            remote = (needed-hbm)*self.shards
            if self.used_remote+remote <= self.remote_capacity+1e-6:
                break
            event = self.changed
            yield event
        fraction = hbm/needed if needed else 1
        self.used_hbm += hbm
        self.used_remote += remote
        self.peak_hbm = max(self.peak_hbm, self.used_hbm)
        self.peak_remote = max(self.peak_remote, self.used_remote)
        for i in ids:
            maximum = self.maximum_bytes(i)
            allocation = dict(request_id=i, max_kv_bytes_per_gpu=maximum,
                              resident_bytes_per_gpu=maximum*fraction,
                              remote_bytes=maximum*(1-fraction)*self.shards,
                              remote_fraction=(1-fraction) if maximum else 0,
                              admission_ready_ms=ready, admission_ms=self.env.now,
                              backing=self.backing, placement=self.mode)
            self.allocations[i] = allocation
            self.holders[i] = allocation
        self.stats['admission_wait_ms'] += self.env.now-ready
        self.ledger.append(dict(action='grant', time_ms=self.env.now, request_ids=list(ids),
                                resident_bytes_per_gpu=hbm, remote_bytes=remote,
                                used_hbm_bytes_per_gpu=self.used_hbm, used_remote_bytes=self.used_remote))

    def release(self, ids):
        hbm = remote = 0
        for i in ids:
            allocation = self.holders.pop(i)
            hbm += allocation['resident_bytes_per_gpu']
            remote += allocation['remote_bytes']
        self.used_hbm = max(0, self.used_hbm-hbm)
        self.used_remote = max(0, self.used_remote-remote)
        self.ledger.append(dict(action='release', time_ms=self.env.now, request_ids=list(ids),
                                resident_bytes_per_gpu=hbm, remote_bytes=remote,
                                used_hbm_bytes_per_gpu=self.used_hbm, used_remote_bytes=self.used_remote))
        old, self.changed = self.changed, self.env.event()
        old.succeed()

    def payload(self, ids, tokens, all_stages=False):
        # One pipeline stage stores layers/PP. The all-stage initialization uses PP copies.
        multiplier = self.shards if all_stages else self.p['tp']*self.p['ep']
        return sum(self.bytes_per_token_gpu*tokens*self.allocations[i]['remote_fraction'] for i in ids)*multiplier

    def transfer(self, ids, payload, operation, cause, stage=-1, step=-1, kv_tokens=0, all_stages=False):
        events = []
        if payload <= 0:
            return events
        paths = active_paths(self.backing,self.c['storage']['ssd_transfer_mode'],operation)
        for path in paths:
            event = yield from self.tiers._io(path, operation, payload, ids[0], cause,
                scope='active', model_phase='decode', direct_gpu=(path=='ssd'), request_ids=list(ids), batch_size=len(ids), context=kv_tokens, stage=stage, token_step=step, serial_tile_stream=True, staging_bytes_per_gpu=self.a['staging_GB']*1e9,kv_tokens=kv_tokens,all_stages=all_stages)
            events.append(event['event_id'])
        self.stats[operation+'_bytes'] += payload
        return events

    def initialize(self, ids):
        payload = sum(self.payload([i], self.rows[i]['input_length'], True) for i in ids if self.rows[i]['output_length']>1)
        events = yield from self.transfer(ids, payload, 'write', 'active_initialize',kv_tokens=self.rows[ids[0]]['input_length'],all_stages=True)
        if not self.c['separated']:
            resident=sum(self.bytes_per_token_gpu*self.rows[i]['input_length']*(1-self.allocations[i]['remote_fraction']) for i in ids if self.rows[i]['output_length']>1)*self.shards
            if resident:
                event=yield from self.tiers._io('hbm','write',resident,ids[0],'active_initialize',scope='active',model_phase='decode',request_ids=list(ids),batch_size=len(ids),context=self.rows[ids[0]]['input_length'],kv_tokens=self.rows[ids[0]]['input_length'],all_stages=True,payload_kind='resident_migration')
                events.append(event['event_id'])
        return events

    def read_stage(self, ids, context, stage, step):
        # The current query token has not yet appended its new KV.
        payload = self.payload(ids, max(0, context-1))
        self.stats['attention_read_bytes'] += payload
        return (yield from self.transfer(ids, payload, 'read', 'active_read', stage, step,max(0,context-1)))

    def append_stage(self, ids, tokens, stage, step):
        payload = self.payload(ids, tokens)
        self.stats['append_write_bytes'] += payload
        return (yield from self.transfer(ids, payload, 'write', 'active_append', stage, step,tokens))

    def summary(self):
        return dict(model_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), placement=self.mode, label=PLACEMENTS[self.mode]['label'], backing=self.backing,
                    scope='Active decode KV only; prefill working KV remains in HBM. Retained-prefix placement is independent.',
                    model='Serialized tiled staging: remote reads + HBM-reference GPU service + append writes. No overlap or calibrated streaming-kernel correction.',
                    calibration='Unvalidated active-offload composition. GPU fits cover HBM-resident kernels; local file-I/O error does not validate streamed attention or target-server SSD performance.',
                    staging_GB_per_gpu=self.a['staging_GB'], hbm_capacity_bytes_per_gpu=self.hbm_capacity,
                    remote_capacity_bytes=self.remote_capacity, peak_resident_bytes_per_gpu=self.peak_hbm,
                    peak_remote_bytes=self.peak_remote, partition=self.partition,
                    stats=dict(self.stats, **self.tiers.active_io_stats), allocations=list(self.allocations.values()), ledger=self.ledger,
                    limitations=['Fixed per-request residency fractions are assigned atomically at microbatch admission and remain pinned until completion; no dynamic promotion.',
                                 'Initialization reserves decode GPU staging buffers and, when disaggregated, holds the handoff channel until backing is ready. Wire and device phases aggregate serial tile cycles; intermediate wire-end does not imply full-context GPU residency.',
                                 'Remote payload is striped uniformly across tokens/layers. Aggregate transfer intervals represent serial chunked staging, not simultaneous full-context GPU residency.',
                                 'GPU resources are held during remote I/O stalls; stage durations include those stalls once. Reference GPU operator times retain HBM memory access costs.',
                                 'GPU energy charges activity only for reference GPU service, and idle power during I/O stalls. CPU/SSD/NIC and DMA power are excluded.',
                                 'Initial prefill and retained-prefix restoration still require full prefill active KV in HBM. This model does not offload the prefill working set.'])
