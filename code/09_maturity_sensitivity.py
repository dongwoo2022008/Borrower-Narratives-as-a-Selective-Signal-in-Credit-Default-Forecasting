# -*- coding: utf-8 -*-
"""Cohort-exclusion sensitivity (review request).

Drops the two most recent application-year cohorts (181 loans, all recorded
as non-default; see 12_cohort_outcome_diagnostic.py — the cohort appears
selectively composed of quickly resolved loans) and repeats the Table 5
comparison over the same 50 stratified 80/20 splits and fixed CAT
specification. Output: ../results/raw/maturity_sensitivity_t5.csv
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
Xk, yk = X[keep], y[keep]
print(f'kept n = {keep.sum()}, default rate = {yk.mean():.4f}')


def pr_auc(yt, p):
    pr, rc, _ = precision_recall_curve(yt, p)
    return auc(rc, pr)


ns_ = len(struct_cols)
rows = []
for seed in range(1, 51):
    tr, te = train_test_split(np.arange(len(yk)), test_size=0.2,
                              stratify=yk, random_state=seed)
    for name, sl in (('S', slice(0, ns_)), ('M', slice(0, ns_ + 100))):
        m = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                               auto_class_weights='Balanced', random_state=42,
                               verbose=False)
        m.fit(Xk[tr, sl], yk[tr])
        p = m.predict_proba(Xk[te, sl])[:, 1]
        yhat = (p >= 0.5).astype(int)
        rows.append({'seed': seed, 'Model': name,
                     'ROC': roc_auc_score(yk[te], p), 'PR': pr_auc(yk[te], p),
                     'Rec': recall_score(yk[te], yhat),
                     'F1': f1_score(yk[te], yhat)})
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'maturity_sensitivity_t5.csv'),
                          index=False, float_format='%.6f')
print('saved maturity_sensitivity_t5.csv')
