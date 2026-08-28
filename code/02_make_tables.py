# -*- coding: utf-8 -*-
"""Compute Tables 5-9 (means, paired t-tests, 95% CIs) from ../results/raw/ and write results/tables/tables_5_9.md."""
import numpy as np, pandas as pd
from scipy import stats
import os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'raw') + os.sep
TOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'tables')
os.makedirs(TOUT, exist_ok=True)
def st(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
def ci_mean(v):
    m=v.mean(); lo,hi=stats.t.interval(0.95,len(v)-1,loc=m,scale=stats.sem(v)); return m,lo,hi
def paired(s,m):
    diff=m-s; mn=diff.mean()
    lo,hi=stats.t.interval(0.95,len(diff)-1,loc=mn,scale=stats.sem(diff))
    t,p=stats.ttest_rel(m,s); return mn,lo,hi,p
MET={'ROC':'ROC-AUC','PR':'PR-AUC','Rec':'Recall','F1':'F1'}
L=[]
def w(x=''): L.append(x)

# Table 5
ov=pd.read_csv(R+'optA_t5_raw.csv')
w('## Table 5. Overall Performance Comparison (Structured-only vs Text-merged, CAT)\n')
w('| Metric | Structured-only | 95% CI | Text-merged | 95% CI | Text Effect (Δ) | 95% CI | p-value |')
w('|---|---|---|---|---|---|---|---|')
piv=ov.pivot_table(index='seed',columns='Model',values=list(MET))
for met in MET:
    s=piv[(met,'S')].values; m=piv[(met,'M')].values
    ms,ls,hs=ci_mean(s); mm,lm,hm=ci_mean(m); d,lo,hi,p=paired(s,m)
    w(f'| {MET[met]} | {ms:.4f} | [{ls:.4f}, {hs:.4f}] | {mm:.4f} | [{lm:.4f}, {hm:.4f}] | {d:+.5f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f}{st(p)} |')
w()

# Table 6 / Table 9
for fname,label,groups,gdesc in [
  ('optA_t6_raw.csv','Table 6. Conditional Effect 1: Performance by Prediction Uncertainty',['Marginal','Clear'],
   'Marginal: p̂∈[0.3,0.7] from the structured-only model; Clear: otherwise. Same subset for both models.'),
  ('optA_t9_raw.csv','Table 9. Conditional Effect 4: Performance by Narrative Informativeness',['Rich','Sparse'],
   'Rich: narrative contains ≥1 of the 100 selected TF–IDF terms (≈29%); Sparse: none.')]:
    df=pd.read_csv(fname if fname.startswith('/') else R+fname)
    w(f'## {label}\n')
    w(f'_{gdesc}_\n')
    w('| Group | Metric | Structured-only | Text-merged | Text Effect (Δ) | 95% CI | p-value |')
    w('|---|---|---|---|---|---|---|')
    for g in groups:
        sub=df[df.Group==g]; piv=sub.pivot_table(index='seed',columns='Model',values=list(MET))
        for met in MET:
            s=piv[(met,'S')].values; m=piv[(met,'M')].values
            d,lo,hi,p=paired(s,m)
            w(f'| {g} | {MET[met]} | {s.mean():.4f} | {m.mean():.4f} | {d:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {p:.3f}{st(p)} |')
    w()

# Table 7
d3=pd.read_csv(R+'optA_t7_raw.csv')
w('## Table 7. False Negative Reduction by Risk Group\n')
w('_Risk groups: credit score bottom 30% (High-Risk) / top 30% (Low-Risk) of each test set._\n')
w('| Risk Group | Structured FN | Text-merged FN | FN Reduction | Reduction Rate | 95% CI (rate) | p-value |')
w('|---|---|---|---|---|---|---|')
for g in ['Overall','High-Risk','Low-Risk']:
    gg=d3[d3.Group==g].sort_values('seed')
    fs=gg.fs.values.astype(float); fm=gg.fm.values.astype(float); r=gg.rate.values
    mr,lo,hi=ci_mean(r); t,p=stats.ttest_rel(fs,fm)
    w(f'| {g} | {fs.mean():.2f} ({fs.std(ddof=1):.2f}) | {fm.mean():.2f} ({fm.std(ddof=1):.2f}) | {np.mean(fs-fm):+.2f} | {mr:+.2f}% | [{lo:.2f}, {hi:.2f}] | {p:.3f}{st(p)} |')
w()

# Table 8
d4=pd.read_csv(R+'optA_t8_raw.csv')
w('## Table 8. Text Effect (Δ) on Recall by Classification Threshold\n')
w('| Threshold (τ) | Group | Struct Recall | Merged Recall | Improvement (Δ) | 95% CI | p-value | Recall Gap (HR−LR, merged) |')
w('|---|---|---|---|---|---|---|---|')
for tau in [0.3,0.4,0.5,0.6,0.7]:
    hr=d4[(d4.tau==tau)&(d4.Group=='High-Risk')].sort_values('seed').rm.values
    lr=d4[(d4.tau==tau)&(d4.Group=='Low-Risk')].sort_values('seed').rm.values
    gap=np.mean(hr-lr)
    for g in ['Overall','High-Risk','Low-Risk']:
        gg=d4[(d4.tau==tau)&(d4.Group==g)].sort_values('seed')
        s=gg.rs.values; m=gg.rm.values; d,lo,hi,p=paired(s,m)
        gcol=f'{gap:.4f}' if g=='High-Risk' else ''
        w(f'| {tau if g=="Overall" else ""} | {g} | {s.mean():.4f} | {m.mean():.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f}{st(p)} | {gcol} |')
w()
open(os.path.join(TOUT,'tables_5_9.md'),'w',encoding='utf-8').write('\n'.join(L))
print('tables written')
