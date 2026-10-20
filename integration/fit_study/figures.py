"""Publication-style static figures; formulas, diagrams and scores for every trial."""
import gzip,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from .data import ROOT,kaist_rows,arrays
from .functions import FAMILIES,predict
OUT=ROOT/'docs/fit-study-2026-10-19'
BLUE='#1466a1';ORANGE='#bc601a';INK='#152d43';GRAY='#526273'
FORMULAS={
 'F0':r'$L+aC+bM$', 'F1':r'$L+aC+bM$', 'F2':r'$L+\max(aC,bM)$',
 'F3':r'$L+\sqrt{(aC)^2+(bM)^2}$', 'F4':r'$L+aW+bM$',
 'F5':r'$L+aC+M[b_0(1-g)+b_1g]$',
 'F6':r'$L+C[a_0(1-g)+a_1g]+M[b_0(1-g)+b_1g]$',
 'F7':r'$L+aC+bM+dM g_c$', 'F8':r'$L+aC+bM_{correct}$'}

def box(ax,x,y,w,h,text):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=.012',facecolor='#edf3f7',edgecolor='#90a6b8',lw=.7))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9,color=INK)
def arrow(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,color=GRAY,lw=.8))

def oof_from_report(result,rows,family):
    models={(op,fold['fold']):fold['model'] for op,d in result['candidates'][family]['by_operator'].items() for fold in d['folds']}
    return np.array([predict(models[(r['op'],r['fold'])],np.array([r['x']]))[0] for r in rows])

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    k=json.load(open(OUT/'kaist/results.json'));rows=kaist_rows();y=np.array([r['ns'] for r in rows])/1000
    fig,axes=plt.subplots(3,3,figsize=(18,19));fig.patch.set_facecolor('white')
    lo,hi=.3,20000
    for ax,f in zip(axes.flat,FAMILIES):
        c=k['candidates'][f];ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.text(.02,.99,f'{f}  {FAMILIES[f]["name"]}',fontsize=13,fontweight='bold',va='top',color=INK)
        ax.text(.5,.90,FORMULAS[f],ha='center',va='center',fontsize=12 if f!='F6' else 10,color=INK)
        left='C, M*' if f=='F8' else 'C, M, shape' if f in ('F4','F5','F6','F7') else 'C, M'
        center={'F0':'weighted sum','F1':'weighted sum','F2':'maximum','F3':'quadratic norm','F4':'tile / wave cost','F5':'memory transition','F6':'two transitions','F7':'cache transition','F8':'correct TP bytes'}[f]
        box(ax,.04,.74,.23,.08,left);box(ax,.36,.74,.30,.08,center);box(ax,.75,.74,.21,.08,'+ L → floor')
        arrow(ax,.28,.78,.35,.78);arrow(ax,.67,.78,.74,.78)
        scatter=ax.inset_axes([.15,.18,.80,.48]);pred=oof_from_report(k,rows,f)/1000
        # Every held-out prediction is plotted; no selective removal of outliers.
        scatter.scatter(y,pred,s=3,alpha=.20,color=BLUE,rasterized=True,linewidths=0)
        scatter.plot([lo,hi],[lo,hi],c=GRAY,lw=.9)
        scatter.fill_between([lo,hi],[lo*.8,hi*.8],[lo*1.2,hi*1.2],color='#a9bac7',alpha=.13)
        scatter.set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),xlabel='Profiled latency (µs)',ylabel='Held-out prediction (µs)')
        scatter.tick_params(labelsize=8);scatter.grid(alpha=.12,which='major')
        ax.text(.50,.055,f'Train {c["training"]["mape_pct"]:.1f}%   |   Shape CV {c["oof"]["mape_pct"]:.1f}%   |   P95 {c["oof"]["p95_ape_pct"]:.1f}%',ha='center',fontsize=10,color=INK)
        ax.text(.50,.015,f'{FAMILIES[f]["parameters"]} parameters/operator incl. transition scale; {FAMILIES[f]["loss"]} loss',ha='center',fontsize=8.5,color=GRAY)
    fig.suptitle('Nine GPU latency functions against published profiles',fontsize=23,color=INK,fontweight='bold',y=.995)
    fig.text(.5,.970,'KAIST BF16 • 9,411 operator observations • five grouped shape folds • every panel uses the same targets',ha='center',fontsize=12,color=GRAY)
    fig.text(.5,.007,'Each panel: function → computation diagram → out-of-fold fit. Diagonal = exact fit; shaded band = ±20%. All functions include the arithmetic floor.',ha='center',fontsize=11,color=GRAY)
    fig.subplots_adjust(top=.944,bottom=.035,hspace=.12,wspace=.13)
    for ext in ['png','svg']:fig.savefig(OUT/('function-iterations.'+ext),dpi=170,facecolor='white',metadata={'Creator':'LLM-ServeScope reproducible fit study'})
    plt.close(fig)
    # Curves show the shape effect behind aggregate scatter/error statistics.
    fig,axes=plt.subplots(1,3,figsize=(17,5.5))
    for ax,(op,title) in zip(axes,[('act_fn','Activation: cache-transition hypothesis'),('embedding','Embedding: vocabulary sharding'),('attention:prefill','Prefill attention: residual shape effects')]):
        chosen=k['selection'][op]['family']
        subset=sorted([r for r in rows if r['op']==op and r['gpu']=='rtx6000' and r['model']=='llama8' and r['tp']==1 and (op!='attention:prefill' or r['shape']['kp']==r['n']/2)],key=lambda r:r['n'])
        if not subset and op=='attention:prefill':
            subset=sorted([r for r in rows if r['op']==op and r['gpu']=='rtx6000' and r['model']=='llama8' and r['tp']==1 and r['shape']['kp']>=r['n']/2],key=lambda r:r['n'])
            subset=[min([r for r in subset if r['n']==n],key=lambda r:r['shape']['kp']) for n in sorted(set(r['n'] for r in subset))]
        x=np.array([r['n'] for r in subset]);target=np.array([r['ns'] for r in subset])/1000
        ax.plot(x,target,'o',ms=4,c=INK,label='Published profile')
        for f,color in [('F0',ORANGE),(chosen,BLUE)]:
            ax.plot(x,oof_from_report(k,subset,f)/1000,lw=1.7,c=color,label=f'{f}: grouped held-out prediction')
        ax.set(xscale='log',yscale='log',xlabel='Tokens per operator call',ylabel='Latency (µs)',title=title);ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Selected functions: representative workload curves (RTX PRO 6000, Llama 8B, TP1)',fontsize=15,color=INK)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','svg']:fig.savefig(OUT/('selected-shapes.'+ext),dpi=170,facecolor='white')
    plt.close(fig)
    audits=[]
    for source in ['kaist','vidur']:
        for name,a in json.load(open(OUT/source/'audits.json')).items():
            labels={('kaist','gpu_rtx4090'):'KAIST\nRTX 4090 GPU',('kaist','gpu_rtx6000'):'KAIST\nRTX PRO 6000 GPU',('kaist','model_qwen32'):'KAIST\nQwen32 model',('vidur','gpu_h100_llama'):'Vidur\nH100 GPU',('vidur','model_codellama_seen_gpus'):'Vidur\nCodeLlama model',('vidur','gpu_and_model_h100_codellama'):'Vidur\nH100 + CodeLlama'}
            audits.append((labels[(source,name)],a))
    fig,axes=plt.subplots(1,2,figsize=(17,6.5));pos=np.arange(len(audits));width=.35
    for ax,metric,title in [(axes[0],'mape_pct','Mean absolute percentage error'),(axes[1],'latency_weighted_ape_pct','Latency-weighted absolute error')]:
        for name,offset,color in [('baseline',-width/2,ORANGE),('selected',width/2,BLUE)]:
            vals=[a[name]['metrics'][metric] for _,a in audits]
            bars=ax.bar(pos+offset,vals,width,color=color,label=name.title())
            ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
        ax.set_xticks(pos);ax.set_xticklabels([label for label,_ in audits],fontsize=8,rotation=15,ha='right')
        ax.set(ylabel='Error (%)',title=title);ax.legend();ax.grid(axis='y',alpha=.15)
    fig.suptitle('Transfer audits: lower average error does not guarantee lower latency-weighted error',fontsize=16,color=INK)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','svg']:fig.savefig(OUT/('transfer-audits.'+ext),dpi=170,facecolor='white')
    plt.close(fig)
    # Normalize generated SVG text for clean version-control diffs.
    for path in OUT.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
if __name__=='__main__':main()
