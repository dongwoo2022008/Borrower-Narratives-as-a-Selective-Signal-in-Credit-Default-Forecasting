# -*- coding: utf-8 -*-
"""Calibration analysis of the structured-only CAT model (reviewer request).

For each of the 50 stratified 80/20 splits (seed = 1..50), trains the
structured-only and text-merged CatBoost models with the fixed specification
and collects test-set predicted probabilities. Reports:
  - reliability curve (10 equal-width bins): mean predicted vs observed default
  - Brier score per seed, both configurations
  - expected calibration error (ECE, 10 bins)
  - properties of the marginal band [0.3, 0.7]: share of test obs, observed
    default rate inside/outside, share of structured-only FNs (tau=0.5) inside
Outputs: ../results/raw/calibration_bins.csv, calibration_seed.csv
"""
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, '01_analysis_unified.py'), encoding='utf-8').read()
ns = {'__file__': os.path.join(HERE, '01_analysis_unified.py')}
exec(compile(src.split("def main(")[0], '01_analysis_unified.py', 'exec'), ns)
load_data = ns['load_data']

OUT = os.path.join(HERE, '..', 'results', 'raw')
X, y, struct_cols = load_data()
n_struct = len(struct_cols)
BINS = np.linspace(0, 1, 11)

bin_rows, seed_rows = [], []
for seed in range(1, 51):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              stratify=y, random_state=seed)
    out = {}
    for name, cols in (('S', slice(0, n_struct)), ('M', slice(0, n_struct + 100))):
        m = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                               auto_class_weights='Balanced', random_state=42,
                               verbose=False)
        m.fit(X[tr, cols], y[tr])
        p = m.predict_proba(X[te, cols])[:, 1]
        out[name] = p
        brier = np.mean((p - y[te]) ** 2)
        idx = np.digitize(p, BINS[1:-1])
        ece = 0.0
        for b in range(10):
            mask = idx == b
            if mask.sum() == 0:
                continue
            ece += mask.mean() * abs(p[mask].mean() - y[te][mask].mean())
        seed_rows.append({'seed': seed, 'model': name, 'brier': brier, 'ece': ece})
        if name == 'S':
            for b in range(10):
                mask = idx == b
                if mask.sum() == 0:
                    continue
                bin_rows.append({'seed': seed, 'bin': b,
                                 'p_mean': p[mask].mean(),
                                 'obs_rate': y[te][mask].mean(),
                                 'n': int(mask.sum())})
    # marginal-band properties (structured-only)
    p = out['S']
    band = (p >= 0.3) & (p <= 0.7)
    fn = (p < 0.5) & (y[te] == 1)
    seed_rows.append({'seed': seed, 'model': 'band',
                      'brier': np.nan, 'ece': np.nan,
                      'band_share': band.mean(),
                      'dr_in': y[te][band].mean(),
                      'dr_out': y[te][~band].mean(),
                      'fn_total': int(fn.sum()),
                      'fn_in_band': int((fn & band).sum())})
    print(f'seed {seed} done')

pd.DataFrame(bin_rows).to_csv(os.path.join(OUT, 'calibration_bins.csv'), index=False, float_format='%.6f')
pd.DataFrame(seed_rows).to_csv(os.path.join(OUT, 'calibration_seed.csv'), index=False, float_format='%.6f')
print('saved')
