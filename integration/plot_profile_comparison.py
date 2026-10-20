"""Render the retained comparison with matplotlib from the native virtualenv."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
data = json.loads(args.input.read_text())
rows = data["results"]
fitted = "calibrated" in data["methodology"]["comparison"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(16, 7.8), sharey=True)
y = np.arange(len(rows))
labels = [f"{'PRO 6000' if r['hardware']=='RTXPRO6000' else 'RTX 4090'} · TP{r['tp']}\n"
          f"B{r['batch']} · {r['input_tokens']} input / {r['output_tokens']} output" for r in rows]
for ax, (metric, title, xlabel) in zip(axes, [
    ("throughput_tps", "Output throughput", "tokens/s · higher is faster"),
    ("ttft_ms", "Time to first token", "ms · lower is faster"),
    ("tpot_ms", "Time per output token", "ms · lower is faster"),
]):
    for offset, prefix, color, label in [(-.19, "native", "#167d8d", "KAIST · profile backed"),
                                        (.19, "analytical", "#d08340", "ServeScope · fitted function" if fitted else "ServeScope · analytical")]:
        values = [r[f"{prefix}_{metric}"] for r in rows]
        bars = ax.barh(y+offset, values, height=.34, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9, color="#293b46")
    ax.set_title(title, fontsize=14, weight="bold", loc="left", pad=16)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e5ebef")
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#d0dae0')
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(max(r[f"{p}_{metric}"] for r in rows) for p in ["native","analytical"])*1.23)
    for position in [2.5,5.5]:
        ax.axhline(position,color="#d0dae0",linewidth=.8)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
fig.suptitle("LLM-ServeScope versus KAIST's profile-backed simulator", x=.02, y=.99,
             ha="left", fontsize=19, weight="bold", color="#173947")
fig.text(.02,.94,"Llama 3.1 8B · BF16 · colocated serving · nine local simulation comparisons",
         fontsize=12, color="#405864")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.58,.025),
           ncol=2, frameon=False, fontsize=11)
fig.text(.02,.005,"Throughput uses the request-completion window. B4 analytical arrivals span 0.3 ms; native arrivals are simultaneous.\n" +
         ("Operator coefficients fitted offline; no fitting to these serving results. 19 Oct 2026." if fitted else
          "Comparison against upstream profiles, not measured end-to-end inference. No coefficients fitted. 19 Oct 2026."),
         fontsize=9, color="#405864")
fig.tight_layout(rect=(0,.085,1,.91),w_pad=2.5)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=160,facecolor="white",bbox_inches="tight")
fig.savefig(args.output.with_suffix('.svg'),facecolor="white",bbox_inches="tight")

# Keep generated SVG text canonical for review and version control.
svg = args.output.with_suffix(".svg")
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
