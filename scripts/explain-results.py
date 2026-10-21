"""Rebuild the default-run explanation figures from exported simulation events.

Run: .venv-sim/bin/python -B scripts/explain-results.py
No new simulation, measured data, or fitted coefficients are introduced.
"""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/results-explained'
OUT.mkdir(exist_ok=True)
source = ROOT / 'examples/default-result.json'
r = json.loads(source.read_text())
assert (r['config']['requests'], r['config']['input_tokens'], r['config']['output_tokens'], r['config']['arrival_rate'], r['config']['prefill']['batch_size'], r['metrics']['gpu_count']) == (16, 2048, 64, 8, 4, 8), 'This guide requires the documented default scenario.'
q = next(q for q in r['requests'] if q['id'] == 12)
spans = {}
for phase, begin, end in [('prefill', q['batch_ready_ms'], q['prefill_end_ms']),
                          ('transfer', q['prefill_end_ms'], q['first_token_ms']),
                          ('decode', q['first_token_ms'], q['finish_ms'])]:
    events = [e for e in r['timeline'] if e['phase'] == phase and q['id'] in e['request_ids']]
    service = sum(e['duration_ms'] for e in events)
    waiting = end - begin - service
    if abs(waiting) < 1e-9: waiting = 0.0
    assert abs(waiting - sum(e['queue_ms'] for e in events)) < 1e-8
    spans[phase] = dict(begin_ms=begin, end_ms=end, duration_ms=end-begin,
                        service_ms=service, waiting_ms=waiting, event_count=len(events))
formation = q['batch_ready_ms'] - q['arrival_ms']
assert abs(q['arrival_ms'] + formation + sum(s['duration_ms'] for s in spans.values()) - q['finish_ms']) < 1e-8
assert r['metrics']['elapsed_ms'] == max(q['finish_ms'] for q in r['requests'])
assert abs(r['metrics']['throughput_tps'] - 16*64/(r['metrics']['elapsed_ms']/1000)) < 1e-10
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none', 'figure.facecolor': 'white'})
C = dict(prefill='#277b6d', transfer='#bf6b18', decode='#376fa8', wait='#e0e6eb', text='#173443')

fig, (ax, small) = plt.subplots(2, 1, figsize=(13, 8.6), gridspec_kw={'height_ratios': [3, 1.7]}, layout='constrained')
for row, first in enumerate([0,4,8,12]):
    req = next(x for x in r['requests'] if x['id'] == first)
    y = 3-row
    if row == 3: ax.axhspan(y-.40,y+.40,color='#f1f6fa',zorder=-1)
    ax.barh(y,req['batch_ready_ms']-req['arrival_ms'],left=req['arrival_ms'],height=.45,color=C['wait'])
    for phase, begin, end in [('prefill',req['batch_ready_ms'],req['prefill_end_ms']),('transfer',req['prefill_end_ms'],req['first_token_ms']),('decode',req['first_token_ms'],req['finish_ms'])]:
        ax.barh(y,end-begin,left=begin,height=.45,color=C[phase],alpha=.14)
        for e in r['timeline']:
            if e['phase']==phase and first in e['request_ids']:
                ax.barh(y,e['duration_ms'],left=e['start_ms'],height=.45,color=C[phase])
    ax.text(req['finish_ms']+65,y,f"{req['finish_ms']:,.3f} ms",va='center',color=C['text'],fontsize=10)
ax.axvline(r['metrics']['elapsed_ms'],color=C['text'],ls='--',lw=1)
ax.set(xlim=(0,6450),ylim=(-.65,3.6),yticks=[3,2,1,0],yticklabels=['Requests #0–3','#4–7','#8–11','#12–15'],xlabel='Simulation clock (ms from the start of the run)')
ax.set_title('Four batches overlap: the latest completion sets the run duration',loc='left',fontsize=15,pad=22,color=C['text'])
ax.set_xticks([0,1000,2000,3000,4000,5000]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
ax.text(0,1.02,'Solid phase color = resource service; pale phase color = waiting within that phase',transform=ax.transAxes,fontsize=10,color=C['text'])
ax.legend(handles=[Patch(facecolor=C['wait'],label='Batch formation'),*[Patch(facecolor=C[p],label=label) for p,label in [('prefill','Prefill'),('transfer','KV handoff'),('decode','Decode')]]],loc='upper left',bbox_to_anchor=(0,-.16),ncol=4,frameon=False,fontsize=10)
for y,phase,label in [(2,'prefill','Prefill'),(1,'transfer','KV handoff'),(0,'decode','Decode')]:
    s=spans[phase]
    small.barh(y,s['service_ms'],height=.45,color=C[phase])
    small.barh(y,s['waiting_ms'],left=s['service_ms'],height=.45,color=C['wait'])
    small.text(s['duration_ms']+40,y,f"{s['duration_ms']:,.3f} ms total",va='center',fontsize=10,color=C['text'])
small.set(xlim=(0,3350),ylim=(-.65,2.65),yticks=[2,1,0],yticklabels=['Prefill','KV handoff','Decode'],xlabel='Duration (ms) — service and waiting regrouped for comparison')
small.set_title('Request #12: colored service + gray waiting = elapsed phase',loc='left',fontsize=13,pad=12,color=C['text'])
small.grid(axis='x',alpha=.15);small.set_axisbelow(True)
small.text(0,2.4,'592.931 service + 278.793 waiting',fontsize=10,color=C['text'])
small.text(0,.4,'1,102.833 service + 1,529.705 waiting',fontsize=10,color=C['text'])
for ext in ['png','svg']:fig.savefig(OUT/f'default-timeline.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(12,10));ax.set(xlim=(0,12),ylim=(0,10));ax.axis('off')
def box(x,y,w,h,title,detail):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10,rounding_size=0.08',facecolor='#f2f6f9',edgecolor='#b9c8d3',lw=1))
    ax.text(x+w/2,y+h*.67,title,ha='center',va='center',fontsize=12,color=C['text'],weight='bold')
    ax.text(x+w/2,y+h*.28,detail,ha='center',va='center',fontsize=10.5,color=C['text'])
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.2,color='#587181',connectionstyle='arc3'))
box(.25,8.35,3.45,1.05,'Workload','16 requests · 8 requests/s\n2,048 input · 64 output tokens')
box(4.2,8.35,3.45,1.05,'GPU service estimates','Model geometry + GPU specs\n+ functions fitted to profiles')
box(8.15,8.35,3.45,1.05,'Serving and I/O settings','Batch cap 4 · separate pools\nKV payload + bandwidth + latency')
box(2.0,6.25,8.0,1.15,'SimPy schedules work on shared resources','Work readiness + resource availability → start time\nStart time + modeled service duration → finish time')
for x in [1.975,5.925,9.875]:arrow((x,8.25),(6,7.5))
box(2.0,4.35,8.0,1.05,'Each request gets a timestamped path','Arrival → batch ready → prefill → KV handoff → decode → completion')
arrow((6,6.15),(6,5.5))
box(.3,2.35,5.1,1.05,'Choose the latest completion','max(2,580.707; 4,663.791;\n5,294.054; 5,486.647) ms')
box(6.45,2.35,5.1,1.05,'Count completed output tokens','16 × 64 = 1,024 tokens\nPrefill produces token 1; decode produces 63 more')
arrow((4.8,4.25),(2.85,3.5));arrow((7.2,4.25),(9,3.5))
box(2.0,.35,8.0,1.05,'Generation throughput','1,024 tokens ÷ 5.486646698 seconds = 186.634944 tokens/s')
arrow((2.85,2.25),(4.75,1.5));arrow((9,2.25),(7.25,1.5))
ax.set_title('How inputs become the reported throughput',loc='left',fontsize=17,pad=18,color=C['text'])
for ext in ['png','svg']:fig.savefig(OUT/f'calculation-process.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
summary=dict(source='examples/default-result.json',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),config_sha256=r['provenance']['config_sha256'],request_id=12,arrival_ms=q['arrival_ms'],formation_wait_ms=formation,spans=spans,finish_ms=q['finish_ms'],latency_ms=q['latency_ms'],ttft_ms=q['ttft_ms'],tpot_ms=q['tpot_ms'],throughput_tps=r['metrics']['throughput_tps'])
(OUT/'numbers.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
