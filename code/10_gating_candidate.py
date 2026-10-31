# -*- coding: utf-8 -*-
"""Gating-candidate reconstruction (review request).

Candidate policy: use the text-merged model's probability only for
applications whose structured-only probability lies in the marginal band
[0.3, 0.7]; use the structured-only probability elsewhere (tau = 0.5).
Reports per-seed band size, FN removed, FP added, and overall Recall of the
gated policy versus structured-only. Output:
../results/raw/gating_candidate.csv
"""
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, '01_analysis_unified.py'), encoding='utf-8').read()
ns = {'__file__': os.path.join(HERE, '01_analysis_unified.py')}
exec(compile(src.split("def main(")[0], '01_analysis_unified.py', 'exec'), ns)
load_data = ns['load_data']

OUT = os.path.join(HERE, '..', 'results', 'raw')
X, y, struct_cols = load_data()
ns_ = len(struct_cols)
rows = []
for seed in range(1, 51):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              stratify=y, random_state=seed)
    ms = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                            auto_class_weights='Balanced', random_state=42,
                            verbose=False)
    ms.fit(X[tr, :ns_], y[tr])
    ps = ms.predict_proba(X[te, :ns_])[:, 1]
    mm = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                            auto_class_weights='Balanced', random_state=42,
                            verbose=False)
    mm.fit(X[tr, :ns_ + 100], y[tr])
    pm = mm.predict_proba(X[te, :ns_ + 100])[:, 1]
    band = (ps >= 0.3) & (ps <= 0.7)
    pg = np.where(band, pm, ps)
    ys, yg = (ps >= .5).astype(int), (pg >= .5).astype(int)
    yt = y[te]
    rows.append({'seed': seed, 'n_band': int(band.sum()),
                 'dFN': int(((ys == 0) & (yt == 1)).sum()
                            - ((yg == 0) & (yt == 1)).sum()),
                 'dFP': int(((yg == 1) & (yt == 0)).sum()
                            - ((ys == 1) & (yt == 0)).sum()),
                 'rec_s': recall_score(yt, ys), 'rec_g': recall_score(yt, yg)})
    print(f'seed {seed} done')
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'gating_candidate.csv'),
                          index=False, float_format='%.6f')
print('saved gating_candidate.csv')
