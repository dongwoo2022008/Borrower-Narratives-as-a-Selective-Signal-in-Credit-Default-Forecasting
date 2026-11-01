# -*- coding: utf-8 -*-
"""Cohort-restricted FULL sensitivity re-run (review request).

Drops the two most recent application-year cohorts (181 loans, all recorded
as non-default) and repeats, on the remaining 5,876 loans, every conditional
analysis of the manuscript with the same 50 stratified 80/20 splits and the
fixed CAT specification: Tables 6-9 subgroup effects, calibration (ECE,
Brier, marginal-band shares), and the gating-candidate reconstruction.

The main sample is unchanged; this is a sensitivity analysis. Note that the
released inputs were preprocessed once on the full sample, so dropping rows
does not refit the preprocessing; results are a robustness check on fixed,
prespecified inputs.

Outputs (../results/raw/):
  restricted_t6.csv  seed,Group(Marginal/Clear),Model,ROC,PR,Rec,F1
  restricted_t7.csv  seed,Group(Overall/High/Low),fs,fm,rate
  restricted_t8.csv  seed,tau,Group,rs,rm
  restricted_t9.csv  seed,Group(Rich/Sparse),Model,n,ROC,PR,Rec,F1
  restricted_calibration.csv  seed,model,brier,ece,band_share,fn_total,fn_in_band
  restricted_gating.csv  seed,n_band,dFN,dFP,rec_s,rec_g
"""
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, f1_score,
                             precision_recall_curve, auc)

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, '01_analysis_unified.py'), encoding='utf-8').read()
ns = {'__file__': os.path.join(HERE, '01_analysis_unified.py')}
exec(compile(src.split("def main(")[0], '01_analysis_unified.py', 'exec'), ns)
load_data = ns['load_data']

OUT = os.path.join(HERE, '..', 'results', 'raw')
X, y, struct_cols = load_data()
ly = X[:, struct_cols.index('loan_year')]
top2 = np.sort(np.unique(ly))[-2:]
keep = ~np.isin(ly, top2)
X, y = X[keep], y[keep]
ns_ = len(struct_cols)
cs = struct_cols.index('credit_score')
rich_all = (X[:, ns_:ns_ + 100] != 0).any(axis=1)
print(f'restricted n = {len(y)}, default rate = {y.mean():.4f}, '
      f'rich share = {rich_all.mean():.4f}')


def pr_auc(yt, p):
    pr, rc, _ = precision_recall_curve(yt, p)
    return auc(rc, pr)


def metrics(yt, p, yhat):
    return dict(ROC=roc_auc_score(yt, p), PR=pr_auc(yt, p),
                Rec=recall_score(yt, yhat), F1=f1_score(yt, yhat))


def ece10(yt, p):
    bins = np.clip((p * 10).astype(int), 0, 9)
    e, n = 0.0, len(yt)
    for b in range(10):
        m = bins == b
        if m.sum():
            e += m.sum() / n * abs(yt[m].mean() - p[m].mean())
    return e


t6, t7, t8, t9, cal, gat = [], [], [], [], [], []
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]
for seed in range(1, 51):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              stratify=y, random_state=seed)
    probs = {}
    for name, sl in (('S', slice(0, ns_)), ('M', slice(0, ns_ + 100))):
        m = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                               auto_class_weights='Balanced', random_state=42,
                               verbose=False)
        m.fit(X[tr, sl], y[tr])
        probs[name] = m.predict_proba(X[te, sl])[:, 1]
    yt = y[te]
    ps, pm = probs['S'], probs['M']
    band = (ps >= 0.3) & (ps <= 0.7)

    # Table 6: marginal vs clear (defined by structured-only p-hat)
    for gname, mask in (('Marginal', band), ('Clear', ~band)):
        for name, p in (('S', ps), ('M', pm)):
            yh = (p >= 0.5).astype(int)
            t6.append(dict(seed=seed, Group=gname, Model=name,
                           **metrics(yt[mask], p[mask], yh[mask])))

    # risk tiers by credit score quantiles within the test set
    score = X[te, cs]
    lo_cut, hi_cut = np.quantile(score, 0.30), np.quantile(score, 0.70)
    tiers = {'Overall': np.ones(len(te), bool),
             'High-Risk': score <= lo_cut, 'Low-Risk': score >= hi_cut}

    # Table 7: FN counts at tau=0.5
    ys, ym = (ps >= 0.5).astype(int), (pm >= 0.5).astype(int)
    for gname, mask in tiers.items():
        fs = int(((ys == 0) & (yt == 1) & mask).sum())
        fm = int(((ym == 0) & (yt == 1) & mask).sum())
        rate = (fs - fm) / fs * 100 if fs > 0 else 0.0
        t7.append(dict(seed=seed, Group=gname, fs=fs, fm=fm, rate=rate))

    # Table 8: threshold-specific recall
    for tau in TAUS:
        ys_t, ym_t = (ps >= tau).astype(int), (pm >= tau).astype(int)
        for gname, mask in tiers.items():
            t8.append(dict(seed=seed, tau=tau, Group=gname,
                           rs=recall_score(yt[mask], ys_t[mask]),
                           rm=recall_score(yt[mask], ym_t[mask])))

    # Table 9: rich vs sparse
    rich_te = rich_all[te]
    for gname, mask in (('Rich', rich_te), ('Sparse', ~rich_te)):
        for name, p in (('S', ps), ('M', pm)):
            yh = (p >= 0.5).astype(int)
            t9.append(dict(seed=seed, Group=gname, Model=name,
                           n=int(mask.sum()),
                           **metrics(yt[mask], p[mask], yh[mask])))

    # calibration + band stats
    for name, p in (('S', ps), ('M', pm)):
        cal.append(dict(seed=seed, model=name,
                        brier=float(np.mean((p - yt) ** 2)),
                        ece=ece10(yt, p),
                        band_share=float(band.mean()),
                        fn_total=int(((ys == 0) & (yt == 1)).sum()),
                        fn_in_band=int(((ys == 0) & (yt == 1) & band).sum())))

    # gating candidate
    pg = np.where(band, pm, ps)
    yg = (pg >= 0.5).astype(int)
    gat.append(dict(seed=seed, n_band=int(band.sum()),
                    dFN=int(((ys == 0) & (yt == 1)).sum()
                            - ((yg == 0) & (yt == 1)).sum()),
                    dFP=int(((yg == 1) & (yt == 0)).sum()
                            - ((ys == 1) & (yt == 0)).sum()),
                    rec_s=recall_score(yt, ys), rec_g=recall_score(yt, yg)))
    print(f'seed {seed} done')

pd.DataFrame(t6).to_csv(os.path.join(OUT, 'restricted_t6.csv'),
                        index=False, float_format='%.6f')
pd.DataFrame(t7).to_csv(os.path.join(OUT, 'restricted_t7.csv'),
                        index=False, float_format='%.8f')
pd.DataFrame(t8).to_csv(os.path.join(OUT, 'restricted_t8.csv'),
                        index=False, float_format='%.6f')
pd.DataFrame(t9).to_csv(os.path.join(OUT, 'restricted_t9.csv'),
                        index=False, float_format='%.6f')
pd.DataFrame(cal).to_csv(os.path.join(OUT, 'restricted_calibration.csv'),
                         index=False, float_format='%.6f')
pd.DataFrame(gat).to_csv(os.path.join(OUT, 'restricted_gating.csv'),
                         index=False, float_format='%.6f')
print('saved restricted_t6/t7/t8/t9, restricted_calibration, restricted_gating')
