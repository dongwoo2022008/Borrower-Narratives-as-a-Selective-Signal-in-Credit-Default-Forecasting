# -*- coding: utf-8 -*-
"""Appendix B robustness: repeat the unified pipeline with dense text embeddings
(MiniLM 384-d, KoSimCSE 768-d) replacing TF-IDF.

Design is identical to code/01_analysis_unified.py: 80/20 stratified splits
(seeds 1-50), CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
auto_class_weights='Balanced', random_state=42) for both configurations;
struct-band uncertainty; credit-score 30/70 tiers; thresholds 0.3-0.7.

NOTE ON DATA: the dense embedding matrices (6,057 x 384 and 6,057 x 768) are too
large to distribute in this repository and are derived from the confidential
narrative texts; they are available from the corresponding author on reasonable
request. The per-seed outputs of this script (results/raw/rob_*.csv) are
included, so the Appendix B summary (results/tables/appendix_b_table.md) is
fully reproducible from the shipped raw records.

Expected pickle structure per representation: dict with X_train, X_test
(struct 13 columns first, then embedding dims), y_train, y_test.
"""
import pickle, sys, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, f1_score,
                             precision_recall_curve, auc)

BASE = ''  # directory containing the embedding pickles (available on request)
REPS = {
    'MiniLM': 'preprocessed_merged_struct_minilm_binary.pkl',
    'KoSimCSE': 'preprocessed_merged_struct_kosimcse_binary.pkl',
}
NS = 13


def prauc(yt, pp):
    p, r, _ = precision_recall_curve(yt, pp)
    return auc(r, p)


def mk():
    return CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                              auto_class_weights='Balanced', random_state=42,
                              verbose=False)


def run(rep):
    d = pickle.load(open(BASE + REPS[rep], 'rb'))
    X = np.vstack([d['X_train'], d['X_test']])
    y = np.concatenate([np.array(d['y_train']), np.array(d['y_test'])])
    cs_idx = 1  # credit_score column position within the structured block
    ov, d2, d3, d4 = [], [], [], []
    for seed in range(1, 51):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                              random_state=seed, stratify=y)
        m1, m2 = mk(), mk()
        m1.fit(Xtr[:, :NS], ytr)
        m2.fit(Xtr, ytr)
        ps = m1.predict_proba(Xte[:, :NS])[:, 1]
        pm = m2.predict_proba(Xte)[:, 1]
        yps = (ps >= 0.5).astype(int)
        ypm = (pm >= 0.5).astype(int)
        for model, pp, yp in [('S', ps, yps), ('M', pm, ypm)]:
            ov.append(dict(seed=seed, Model=model, ROC=roc_auc_score(yte, pp),
                           PR=prauc(yte, pp), Rec=recall_score(yte, yp),
                           F1=f1_score(yte, yp)))
        mb = (ps >= 0.3) & (ps <= 0.7)
        for g, mask in [('Marginal', mb), ('Clear', ~mb)]:
            for model, pp, yp in [('S', ps, yps), ('M', pm, ypm)]:
                d2.append(dict(seed=seed, Group=g, Model=model,
                               ROC=roc_auc_score(yte[mask], pp[mask]),
                               PR=prauc(yte[mask], pp[mask]),
                               Rec=recall_score(yte[mask], yp[mask]),
                               F1=f1_score(yte[mask], yp[mask])))
        cs = Xte[:, cs_idx]
        tiers = [('Overall', np.ones(len(yte), bool)),
                 ('High-Risk', cs <= np.percentile(cs, 30)),
                 ('Low-Risk', cs >= np.percentile(cs, 70))]
        for g, mask in tiers:
            fs = int(np.sum((yte[mask] == 1) & (yps[mask] == 0)))
            fm = int(np.sum((yte[mask] == 1) & (ypm[mask] == 0)))
            d3.append(dict(seed=seed, Group=g, fs=fs, fm=fm,
                           rate=100 * (fs - fm) / fs if fs else 0))
        for tau in [0.3, 0.4, 0.5, 0.6, 0.7]:
            yst = (ps >= tau).astype(int)
            ymt = (pm >= tau).astype(int)
            for g, mask in tiers:
                d4.append(dict(seed=seed, tau=tau, Group=g,
                               rs=recall_score(yte[mask], yst[mask]),
                               rm=recall_score(yte[mask], ymt[mask])))
        if seed % 10 == 0:
            print(rep, 'seed', seed, flush=True)
    out = '../results/raw/'
    pd.DataFrame(ov).to_csv(out + f'rob_{rep}_t5.csv', index=False)
    pd.DataFrame(d2).to_csv(out + f'rob_{rep}_t6.csv', index=False)
    pd.DataFrame(d3).to_csv(out + f'rob_{rep}_t7.csv', index=False)
    pd.DataFrame(d4).to_csv(out + f'rob_{rep}_t8.csv', index=False)
    print(rep, 'DONE', flush=True)


if __name__ == '__main__':
    for rep in (sys.argv[1:] or list(REPS)):
        run(rep)
