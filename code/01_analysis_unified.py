# -*- coding: utf-8 -*-
"""Unified conditional-effects analysis.

Reproduces the per-seed raw results behind Tables 5-9 of
"Borrower Narratives as a Selective Signal in Credit Default Forecasting".

Pipeline
--------
Data      : N = 6,057 loan applications; 13 standardized structured features
            + 100 TF-IDF narrative features (see ../data/README.md).
Splits    : stratified 80/20 train/test re-splits of the full sample,
            random_state = seed for seed in 1..50.
Models    : CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
            auto_class_weights='Balanced', random_state=42) for BOTH the
            structured-only (13 features) and text-merged (113 features)
            configurations. Class predictions use threshold 0.5 unless stated.
D1        : uncertainty - marginal = structured-only p-hat in [0.3, 0.7],
            the SAME subset is used to evaluate both models.
D2        : FN dynamics - credit-score bottom/top 30% tiers per test set.
D3        : thresholds tau in {0.3, 0.4, 0.5, 0.6, 0.7}.
D4        : narrative informativeness - Rich = at least one nonzero TF-IDF
            feature, Sparse = none.

Outputs (../results/raw/)
-------------------------
optA_t5_raw.csv .. optA_t9_raw.csv : per-seed records for Tables 5-9
fig2_hist.csv                      : mean per-bin histogram data for Figure 2

Run time: a few minutes on a laptop CPU.
"""
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, f1_score,
                             precision_recall_curve, auc)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(HERE, '..', 'results', 'raw')
os.makedirs(OUT, exist_ok=True)

N_TFIDF = 100
CREDIT_COL = 'credit_score'


def load_data():
    parts = [pd.read_csv(os.path.join(DATA, f'structured_features_part{i}.csv'))
             for i in (1, 2, 3)]
    s = pd.concat(parts, ignore_index=True).sort_values('row_id').reset_index(drop=True)
    t = pd.read_csv(os.path.join(DATA, 'tfidf_features_sparse.csv'))
    l = pd.read_csv(os.path.join(DATA, 'labels.csv'))
    struct_cols = [c for c in s.columns if c != 'row_id']
    X = np.zeros((len(s), len(struct_cols) + N_TFIDF))
    X[:, :len(struct_cols)] = s[struct_cols].values
    X[t['row_id'].values, len(struct_cols) + t['term_id'].values] = t['value'].values
    y = l['default'].values.astype(int)
    return X, y, struct_cols


def prauc(y_true, proba):
    p, r, _ = precision_recall_curve(y_true, proba)
    return auc(r, p)


def make_model():
    return CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                              auto_class_weights='Balanced', random_state=42,
                              verbose=False)


def main():
    X, y, struct_cols = load_data()
    ns = len(struct_cols)
    cs_idx = struct_cols.index(CREDIT_COL)

    ov, d2, d3, d4, d9 = [], [], [], [], []
    bins = np.arange(0.0, 1.025, 0.025)
    h_norm = np.zeros(len(bins) - 1)
    h_def = np.zeros(len(bins) - 1)
    shares = []

    for seed in range(1, 51):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        m1, m2 = make_model(), make_model()
        m1.fit(Xtr[:, :ns], ytr)
        m2.fit(Xtr, ytr)
        ps = m1.predict_proba(Xte[:, :ns])[:, 1]
        pm = m2.predict_proba(Xte)[:, 1]
        yps = (ps >= 0.5).astype(int)
        ypm = (pm >= 0.5).astype(int)

        # Table 5: overall
        for model, pp, yp in [('S', ps, yps), ('M', pm, ypm)]:
            ov.append(dict(seed=seed, Model=model, ROC=roc_auc_score(yte, pp),
                           PR=prauc(yte, pp), Rec=recall_score(yte, yp),
                           F1=f1_score(yte, yp)))

        # Table 6: uncertainty (struct-band, same subset for both models)
        mb = (ps >= 0.3) & (ps <= 0.7)
        shares.append(mb.mean())
        h_norm += np.histogram(ps[yte == 0], bins=bins)[0]
        h_def += np.histogram(ps[yte == 1], bins=bins)[0]
        for g, mask in [('Marginal', mb), ('Clear', ~mb)]:
            for model, pp, yp in [('S', ps, yps), ('M', pm, ypm)]:
                d2.append(dict(seed=seed, Group=g, Model=model,
                               ROC=roc_auc_score(yte[mask], pp[mask]),
                               PR=prauc(yte[mask], pp[mask]),
                               Rec=recall_score(yte[mask], yp[mask]),
                               F1=f1_score(yte[mask], yp[mask])))

        # Table 7: FN dynamics by credit-score tiers
        cs = Xte[:, cs_idx]
        tiers = [('Overall', np.ones(len(yte), bool)),
                 ('High-Risk', cs <= np.percentile(cs, 30)),
                 ('Low-Risk', cs >= np.percentile(cs, 70))]
        for g, mask in tiers:
            fs = int(np.sum((yte[mask] == 1) & (yps[mask] == 0)))
            fm = int(np.sum((yte[mask] == 1) & (ypm[mask] == 0)))
            d3.append(dict(seed=seed, Group=g, fs=fs, fm=fm,
                           rate=100 * (fs - fm) / fs if fs else 0))

        # Table 8: threshold sensitivity
        for tau in [0.3, 0.4, 0.5, 0.6, 0.7]:
            yst = (ps >= tau).astype(int)
            ymt = (pm >= tau).astype(int)
            for g, mask in tiers:
                d4.append(dict(seed=seed, tau=tau, Group=g,
                               rs=recall_score(yte[mask], yst[mask]),
                               rm=recall_score(yte[mask], ymt[mask])))

        # Table 9: narrative informativeness
        nz = np.count_nonzero(Xte[:, ns:], axis=1)
        for g, mask in [('Rich', nz > 0), ('Sparse', nz == 0)]:
            for model, pp, yp in [('S', ps, yps), ('M', pm, ypm)]:
                d9.append(dict(seed=seed, Group=g, Model=model, n=int(mask.sum()),
                               ROC=roc_auc_score(yte[mask], pp[mask]),
                               PR=prauc(yte[mask], pp[mask]),
                               Rec=recall_score(yte[mask], yp[mask]),
                               F1=f1_score(yte[mask], yp[mask])))
        if seed % 10 == 0:
            print(f'seed {seed}/50 done')

    pd.DataFrame(ov).to_csv(os.path.join(OUT, 'optA_t5_raw.csv'), index=False)
    pd.DataFrame(d2).to_csv(os.path.join(OUT, 'optA_t6_raw.csv'), index=False)
    pd.DataFrame(d3).to_csv(os.path.join(OUT, 'optA_t7_raw.csv'), index=False)
    pd.DataFrame(d4).to_csv(os.path.join(OUT, 'optA_t8_raw.csv'), index=False)
    pd.DataFrame(d9).to_csv(os.path.join(OUT, 'optA_t9_raw.csv'), index=False)
    centers = (bins[:-1] + bins[1:]) / 2
    pd.DataFrame({'bin_center': centers, 'mean_repaid': h_norm / 50,
                  'mean_default': h_def / 50}).to_csv(
        os.path.join(OUT, 'fig2_hist.csv'), index=False)
    with open(os.path.join(OUT, 'marginal_share.txt'), 'w') as f:
        f.write(f'mean marginal share = {np.mean(shares):.4f}\n')
    print(f'done; mean marginal share = {np.mean(shares):.4f}')


if __name__ == '__main__':
    main()
