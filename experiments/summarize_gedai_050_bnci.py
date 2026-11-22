"""Summarize completed, paired subject results without imputing failures."""
from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]/'results'/'gedai_050_bnci'
NAMES={'bandpass_baseline':'Bandpass baseline','bandpass_average_reference':'Average-reference control',
'legacy_current_checkout':'Legacy current checkout','gedai_050_broadband_epochs':'GEDAI 0.5.0 broadband epochs',
'gedai_050_continuous':'GEDAI 0.5.0 continuous'}
rows=[]
for s in range(1,10):
 r=json.loads((OUT/f'subject_{s}.json').read_text())
 assert set(r['methods'])==set(NAMES),f'Incomplete subject {s}'
 for k,v in r['methods'].items():rows.append({'subject':s,'method':k,'ba':v['ba']})
d=pd.DataFrame(rows);d.to_csv(OUT/'scores.csv',index=False)
p=d.pivot(index='subject',columns='method',values='ba')[list(NAMES)]*100
assert p.shape==(9,5) and np.isfinite(p).all().all()
summ=[]
for k in NAMES:
 a=p[k].to_numpy();diff=a-p.bandpass_baseline.to_numpy()
 se=stats.sem(diff);ci=(diff.mean()-stats.t.ppf(.975,8)*se,diff.mean()+stats.t.ppf(.975,8)*se)
 perms=np.array(list(itertools.product([-1,1],repeat=9)))@diff/9
 pv=float(np.mean(np.abs(perms)>=abs(diff.mean())-1e-12))
 summ.append({'method':k,'mean_ba_percent':a.mean(),'sd_percent':a.std(ddof=1),
 'paired_delta_pp':diff.mean(),'delta_ci_low':ci[0],'delta_ci_high':ci[1],'exact_two_sided_p':pv})
s=pd.DataFrame(summ);s.to_csv(OUT/'summary.csv',index=False)
text=['# GEDAI 0.5.0: BNCI2014_001 results','',
'Nine subjects; both sessions; binary left/right CSP + shrinkage LDA; paired seed-42 80/20 holdout; no synthetic artifacts or epoch rejection. See [protocol](PROTOCOL.md) for full settings and limitations.','',
'| Pipeline | BA, mean ± SD | Change vs baseline | Paired 95% CI |','|---|---:|---:|---:|']
for r in summ:
 text.append(f"| {NAMES[r['method']]} | {r['mean_ba_percent']:.2f} ± {r['sd_percent']:.2f}% | {r['paired_delta_pp']:+.2f} pp | [{r['delta_ci_low']:+.2f}, {r['delta_ci_high']:+.2f}] pp |")
text+=['','## Subject results','', '| Subject | '+ ' | '.join(NAMES.values())+' |','|---|'+'---:|'*5]
for i,row in p.iterrows():text.append('| '+str(i)+' | '+' | '.join(f'{v:.2f}%' for v in row)+' |')
text+=['','## Interpretation','',
'The saved historical CSVs report 74.33% baseline and 75.29% GEDAI; the manuscript reports 68.8% GEDAI. The historical 68.8% result was not reproduced or identified in these saved CSVs. The legacy row above is a fresh run of the current working tree.','',
'Continuous cleaning is offline and transductive: a full run can contain training and test trials, and cleaning uses unlabeled EEG from both. Classifier training remains confined to training labels. These results do not establish prospective online performance, and cannot isolate wavelet boundary effects from other pipeline changes.','',
'Confidence intervals treat subjects as the independent units. Exact sign-flip p values are saved in summary.csv (two-sided, unadjusted). A non-significant difference does not establish equivalence.','',
'![Subject accuracies](comparison.png)']
(OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
fig,ax=plt.subplots(figsize=(10,5.2))
x=np.arange(5)
for i,row in p.iterrows():ax.plot(x,row,marker='o',alpha=.45,lw=1,label=f'S{i}')
ax.plot(x,p.mean(),color='black',marker='D',lw=3,label='Subject mean')
ax.axhline(74.3,color='gray',ls=':',lw=1,label='Historical baseline 74.3%')
ax.set_xticks(x,['Bandpass','Average reference','Legacy checkout','0.5.0 broadband\nepochs','0.5.0 adaptive\ncontinuous'])
ax.set_ylabel('Balanced accuracy (%)');ax.set_title('BNCI2014_001 · paired offline holdout · 9 subjects')
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',fontsize=8)
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180)
print(s.to_string(index=False))
