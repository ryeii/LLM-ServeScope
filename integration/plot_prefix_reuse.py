"""Plot audited workload potential, not cache performance or hardware accuracy."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/'backend/prefix-reuse.json').read_text())
rows = list(data['workloads'].values())
fig, ax = plt.subplots(figsize=(9,4.8),layout='constrained')
bars = ax.bar([r['label'] for r in rows],[100*r['fraction'] for r in rows],color=['#347f71','#527eaf','#8b78ad'],width=.55)
ax.bar_label(bars,labels=[f"{100*r['fraction']:.2f}%" for r in rows],padding=6,fontsize=12)
ax.set(ylim=(0,80),ylabel='Potentially reusable input tokens (%)')
ax.set_title('Mooncake published traces: earlier-arrival prefix reuse potential',fontsize=13,pad=16)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.supxlabel('Cold history · unlimited retention · no reuse within tied arrivals\nWorkload potential only; not measured HBM hits or serving speedup',fontsize=10)
for extension in ['png','svg']:
    path=ROOT/f'docs/prefix-reuse/reuse-potential.{extension}'
    fig.savefig(path,dpi=180)
    if extension=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
plt.close(fig)
