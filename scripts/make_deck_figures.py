import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Polygon
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.edgecolor":"#3A4A63",
                     "axes.labelcolor":"#12203C","text.color":"#12203C","xtick.color":"#3A4A63",
                     "ytick.color":"#3A4A63","axes.grid":True,"grid.alpha":0.25,"grid.linestyle":":"})
OUT="/Users/abhishekr/Documents/EEG/enova/gedai_enova_benchmark/results/figures/deck/"
NAVY="#12203C"; TEAL="#2A9D8F"; CORAL="#E76F51"; AMBER="#E9A13B"; GREY="#8A97A8"

# ─────────────────────────────────────────────────────────── 1. PIPELINE
fig,ax=plt.subplots(figsize=(15.5,5.0)); ax.set_xlim(0,100); ax.set_ylim(0,34); ax.axis("off")
def box(x,y,w,h,t,sub,fc,tc="white",fs=10.2,sfs=8.4):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.35,rounding_size=1.1",
                fc=fc,ec="none"))
    ax.text(x+w/2,y+h*0.63,t,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold")
    ax.text(x+w/2,y+h*0.26,sub,ha="center",va="center",color=tc,fontsize=sfs,alpha=0.92)
def arrow(x1,y1,x2,y2,c=NAVY,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,
                lw=1.6,color=c,linestyle=ls,shrinkA=0,shrinkB=0))
Y=20.5; H=7.2
box(1.0,Y,13.0,H,"RAW EEG","X :  C × T  matrix",NAVY)
box(16.0,Y,13.0,H,"AVG REFERENCE","x − mean(x)·N/(N+1)","#274060")
box(31.0,Y,14.5,H,"MODWT","5–9 bands, shift-invariant","#274060")
box(47.5,Y,17.0,H,"GEVD  per band","C·w = λ·R·w",TEAL)
box(66.5,Y,15.0,H,"SENSAI","argmax_τ  S_sig − 3·S_noi",TEAL)
box(83.5,Y,15.5,H,"BACK-PROJECT + iMODWT","X_clean = X − R·V_art·V_artᵀ·X",TEAL,fs=9.4,sfs=8.0)
for a,b in [(14.0,16.0),(29.0,31.0),(45.5,47.5),(64.5,66.5),(81.5,83.5)]: arrow(a,Y+H/2,b,Y+H/2)
# reference cov feeding GEVD
box(47.5,30.0,17.0,3.4,"R = L·Lᵀ   (BEM leadfield, 343 electrodes)","",AMBER,tc=NAVY,fs=9.0)
arrow(56.0,30.0,56.0,Y+H,c=AMBER)
Y2=8.5
box(83.5,Y2,15.5,H,"ENOVA","var(noise) / var(original)","#274060")
arrow(91.2,Y,91.2,Y2+H,)
box(60.0,Y2,20.0,H,"REJECTION DECISIONS","channel 0.95   |   epoch 1.00",CORAL)
arrow(83.5,Y2+H/2,80.0,Y2+H/2)
box(41.0,Y2,16.0,H,"EPOCH → COV","C_i = X_i X_i' / (T_e − 1)","#274060")
arrow(60.0,Y2+H/2,57.0,Y2+H/2)
box(21.0,Y2,17.0,H,"CSP  /  Riemannian","V :  C × K  filters","#274060")
arrow(41.0,Y2+H/2,38.0,Y2+H/2)
box(1.0,Y2,17.0,H,"shrinkage-LDA / MDM","F : N × K   →   y",NAVY)
arrow(21.0,Y2+H/2,18.0,Y2+H/2)
ax.text(50,2.2,"Everything left of the rejection block is a fixed linear operator applied to the continuous record — the decoder never learns that anything changed.",
        ha="center",fontsize=9.6,style="italic",color="#3A4A63")
plt.tight_layout(); plt.savefig(OUT+"fig_pipeline.png",dpi=190,bbox_inches="tight",facecolor="white"); plt.close()

# ─────────────────────────────────────────────────────────── 2. GEVD CONCEPT
fig,(a1,a2)=plt.subplots(1,2,figsize=(13.6,4.9),gridspec_kw={"width_ratios":[1.15,1]})
rng=np.random.default_rng(4)
lam=np.sort(np.concatenate([rng.gamma(2.0,0.28,size=18), rng.gamma(6,1.9,size=4)]))[::-1]
tau=2.6
cols=[CORAL if v>=tau else TEAL for v in lam]
a1.bar(np.arange(1,len(lam)+1),lam,color=cols,width=0.72)
a1.axhline(tau,color=NAVY,ls="--",lw=1.6)
a1.text(0.72,0.585,"τ  (SENSAI-optimal)",transform=a1.transAxes,fontsize=10.2,color=NAVY,fontweight="bold")
a1.set_yscale("log"); a1.set_xlabel("generalized eigenvalue index  k")
a1.set_ylabel("$\\lambda_k$ = ($w_k^T C w_k$) / ($w_k^T R w_k$)")
a1.set_title("λ is a dimensionless physical plausibility ratio\nobserved power ÷ power a cortex could produce",
             fontsize=11.5,fontweight="bold",loc="left")
a1.text(0.30,0.93,"λ  >  τ   →   ARTIFACT   (projected out)",transform=a1.transAxes,color=CORAL,fontsize=10.5,
        fontweight="bold",va="top",bbox=dict(fc="white",ec=CORAL,lw=1.2,boxstyle="round,pad=0.35"))
a1.text(0.30,0.20,"λ  <  τ   →   physically legal cortical source   (kept)",transform=a1.transAxes,color=TEAL,fontsize=10.5,
        fontweight="bold",va="top",bbox=dict(fc="white",ec=TEAL,lw=1.2,boxstyle="round,pad=0.35"))
a1.grid(axis="y",alpha=0.25)

a2.set_xlim(-1.35,1.35); a2.set_ylim(-1.35,1.35); a2.axis("off")
a2.add_patch(Ellipse((0,0),2.0,1.35,fc=TEAL,alpha=0.20,ec=TEAL,lw=2.0))
a2.text(0,-0.62,"range( L )",ha="center",fontsize=13,color="#1B6F66",fontweight="bold")
a2.text(0,-0.86,"scalp topographies a brain can physically produce",ha="center",fontsize=9.4,color="#1B6F66")
for (x,y,lab) in [(-1.02,0.92,"blink\n(corneo-retinal)"),(1.02,0.86,"temporalis EMG\n(outside skull)"),(0.62,-1.00,"contact noise\n(rank-1, 1 electrode)")]:
    a2.plot([x],[y],marker="X",ms=13,color=CORAL)
    a2.text(x,y+(0.20 if y>0 else -0.32),lab,ha="center",fontsize=8.8,color=CORAL,fontweight="bold")
for (x,y) in [(-0.42,0.10),(0.30,-0.16),(0.05,0.20),(-0.10,-0.09),(0.52,0.16),(-0.66,-0.10),(0.66,-0.05)]:
    a2.plot([x],[y],marker="o",ms=9,color="#1B6F66")
a2.text(0,0.46,"mu / beta cortical sources",ha="center",fontsize=9.6,color="#1B6F66",fontweight="bold")
a2.set_title("The constraint is geometric, not statistical:\n“could a head have generated this topography?”",fontsize=11.5,fontweight="bold",loc="left")
plt.tight_layout(); plt.savefig(OUT+"fig_gevd.png",dpi=190,bbox_inches="tight",facecolor="white"); plt.close()

# ─────────────────────────────────────────────────────────── 3. REJECTION VALLEY (preprint Table 3.1)
th=np.array([.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,1.00])
ba=np.array([.682,.677,.673,.673,.673,.673,.673,.674,.674,.675,.677,.678,.675,.675,.675,.675,.675,.684,.688])
ret=np.array([59.0,61.4,63.1,64.0,64.7,65.0,65.2,65.4,65.8,65.9,66.2,66.7,67.4,68.4,70.9,74.9,77.7,80.8,100.0])
itr=np.array([1.125,1.108,1.108,1.115,1.123,1.129,1.132,1.152,1.175,1.201,1.218,1.251,1.209,1.224,1.305,1.388,1.461,1.620,1.942])
apl=np.array([3.128,2.980,2.926,2.877,2.850,2.845,2.839,2.825,2.810,2.801,2.784,2.765,2.752,2.724,2.673,2.591,2.533,2.397,1.956])
fig,axs=plt.subplots(1,3,figsize=(15.4,4.5))
for ax,y,lab,ttl,c in [(axs[0],ba*100,"Balanced accuracy (%)","Accuracy is FLAT across the sweep",TEAL),
                       (axs[1],ret,"Trials retained (%)","Rejection destroys throughput",AMBER),
                       (axs[2],itr,"Effective ITR (bits/min)","ITR — the only objective that\nrespects the user's clock",CORAL)]:
    ax.plot(th,y,"-o",color=c,lw=2.3,ms=5.2,mec="white",mew=0.8)
    ax.axvline(0.90,color=GREY,ls=":",lw=1.6); ax.axvline(1.00,color=NAVY,ls="--",lw=1.6)
    ax.set_xlabel("ENOVA$_{epoch}$ threshold"); ax.set_ylabel(lab)
    ax.set_title(ttl,fontsize=11.2,fontweight="bold",loc="left")
axs[0].set_ylim(65,72); axs[0].text(.30,70.6,"74.3% = bandpass-only baseline",fontsize=9,color=NAVY)
axs[0].axhline(74.3,color=NAVY,ls="-",lw=1.3,alpha=0.55)
axs[0].set_ylim(65,76)
axs[1].annotate("0.90 discards\n22.3% of trials",(0.90,77.7),xytext=(0.42,86),fontsize=9.2,color=NAVY,
                arrowprops=dict(arrowstyle="->",color=NAVY,lw=1.3))
axs[2].annotate("1.00 → 1.942",(1.00,1.942),xytext=(0.52,1.80),fontsize=10,fontweight="bold",color=NAVY,
                arrowprops=dict(arrowstyle="->",color=NAVY,lw=1.3))
axs[2].annotate("0.90 → 1.461",(0.90,1.461),xytext=(0.30,1.32),fontsize=10,color=GREY,
                arrowprops=dict(arrowstyle="->",color=GREY,lw=1.3))
axs[2].text(0.10,1.88,"+33%  p=0.0060  d=0.70",fontsize=10.5,fontweight="bold",color=CORAL)
plt.tight_layout(); plt.savefig(OUT+"fig_valley.png",dpi=190,bbox_inches="tight",facecolor="white"); plt.close()

# ─────────────────────────────────────────────────────────── 4. BNCI HONEST COMPARISON
fig,(a1,a2)=plt.subplots(1,2,figsize=(13.4,4.6),gridspec_kw={"width_ratios":[1,1]})
names=["Bandpass\nbaseline","GEDAI\nreconstruct-only","ASR\n(2-min calibration)"]
vals=[74.3,68.8,67.05]; sds=[14.3,9.2,13.1]; cs=[NAVY,TEAL,GREY]
b=a1.bar(names,vals,yerr=sds,capsize=6,color=cs,width=0.6,error_kw=dict(ecolor="#5A6A80",lw=1.4))
a1.set_ylim(45,95); a1.axhline(50,color=GREY,ls=":",lw=1.2)
a1.set_ylabel("Balanced accuracy (%)  ± SD")
a1.set_title("BNCI2014_001 (N=9) — accuracy: nobody wins",fontsize=11.5,fontweight="bold",loc="left")
for r,v in zip(b,vals): a1.text(r.get_x()+r.get_width()/2,v+1.2,f"{v:.1f}%",ha="center",fontweight="bold",fontsize=11)
a1.text(0.5,88.5,"p = 0.074   d = −0.74   (n.s.)",ha="center",fontsize=10,color=NAVY)
a1.text(1.5,84.0,"p = 0.119   d = −0.64   (n.s.)",ha="center",fontsize=10,color=NAVY)
a1.text(0.02,46.5,"chance = 50%",fontsize=8.5,color=GREY)

a2.bar(["Reject @ 0.90","Reconstruct @ 1.00"],[1.461,1.942],yerr=[0.512,0.687],capsize=6,
       color=[GREY,CORAL],width=0.5,error_kw=dict(ecolor="#5A6A80",lw=1.4))
a2.set_ylabel("Effective ITR (bits/min) ± SD"); a2.set_ylim(0,3.0)
a2.set_title("…but throughput does: +0.48 bits/min",fontsize=11.5,fontweight="bold",loc="left")
a2.text(0,1.60,"1.461",ha="center",fontweight="bold",fontsize=12)
a2.text(1,2.08,"1.942",ha="center",fontweight="bold",fontsize=12,color=CORAL)
a2.text(0.5,2.72,"p$_{perm}$ = 0.0060    d = 0.70",ha="center",fontsize=11,fontweight="bold",color=NAVY)
plt.tight_layout(); plt.savefig(OUT+"fig_bnci.png",dpi=190,bbox_inches="tight",facecolor="white"); plt.close()

# ─────────────────────────────────────────────────────────── 5. DRY ELECTRODES
fig,ax=plt.subplots(figsize=(9.4,4.9))
nm=["Raw\n(no preprocessing)","GEDAI\ncontinuous-first","Notch-only\n(frequency domain)"]
v=[59.36,55.61,49.42]; s=[10.5,10.6,10.4]; c=[NAVY,TEAL,CORAL]
b=ax.bar(nm,v,yerr=s,capsize=6,color=c,width=0.58,error_kw=dict(ecolor="#5A6A80",lw=1.4))
ax.axhline(50,color=GREY,ls=":",lw=1.4); ax.set_ylim(35,78)
ax.set_ylabel("Balanced accuracy (%) ± SD")
ax.set_title("Dataset 27 — dry electrodes, 11 channels, N=15\nreach-and-grasp (palmar vs lateral)",fontsize=12,fontweight="bold",loc="left")
for r,val in zip(b,v): ax.text(r.get_x()+r.get_width()/2,val+1.4,f"{val:.2f}%",ha="center",fontweight="bold",fontsize=12)
ax.text(0.5,71.5,"−3.75 pp   p = 0.133   d = −0.42   (n.s.)",ha="center",fontsize=10.2,color=NAVY)
ax.text(1.0,66.0,"−9.94 pp   p = 0.0196   d = −0.67   ✱",ha="center",fontsize=10.6,color=CORAL,fontweight="bold")
ax.text(2.02,50.7,"chance",fontsize=9,color=GREY)
ax.annotate("",xy=(2,63.5),xytext=(0,63.5),arrowprops=dict(arrowstyle="<->",color=CORAL,lw=1.5))
ax.annotate("",xy=(1,69.0),xytext=(0,69.0),arrowprops=dict(arrowstyle="<->",color=NAVY,lw=1.2))
plt.tight_layout(); plt.savefig(OUT+"fig_dry.png",dpi=190,bbox_inches="tight",facecolor="white"); plt.close()

# ─────────────────────────────────────────────────────────── 6. WHAT THE SWEEP SAYS (2-knob)
fig,ax=plt.subplots(figsize=(11.0,4.4)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,40)
def card(x,y,w,h,head,body,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6,rounding_size=1.6",fc=fc,ec=ec,lw=2.0))
    ax.text(x+2.5,y+h-4.0,head,fontsize=13.5,fontweight="bold",color=ec,va="top")
    ax.text(x+2.5,y+h-10.5,body,fontsize=10.4,color="#25344B",va="top",linespacing=1.55)
card(1,2,46,36,"CHANNELS  →  GO   (η_ch = 0.95)",
     "A bad channel is a GLOBAL corruption:\nit poisons every trial in the session.\n\n• Removal cost is bounded — volume conduction\n  makes neighbours highly redundant\n• Benefit applies to every epoch at once\n• Protects the GED filter from collapse",
     "#FDF1EC",CORAL)
card(53,2,46,36,"EPOCHS  →  STAY   (η_ep = 1.00)",
     "A bad epoch is a LOCAL corruption that\nGEDAI has ALREADY repaired.\n\n• You would discard a trial P already fixed\n• You pay twice: training N ↓ and throughput ↓\n• Smaller N degrades the CSP covariance itself\n• ITR penalty: −0.48 bits/min at 0.90",
     "#EAF6F4",TEAL)
plt.tight_layout(); plt.savefig(OUT+"fig_twoknob.png",dpi=190,bbox_inches="tight",facecolor="white"); plt.close()
print("figures written")
