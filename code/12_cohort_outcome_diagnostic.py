# -*- coding: utf-8 -*-
"""Outcome-observability diagnostic for the two most recent cohorts.

The 181 loans from the two most recent application years carry zero
recorded defaults. This script quantifies how implausible that is if
their recorded status were a completed outcome:
  1. cohort-by-year default rates (the discontinuity),
  2. the default rate of earlier-cohort loans with a matching low-rate /
     high-score profile, and the binomial probability of 0/181 at that rate,
  3. the expected number of defaults among the 181 implied by a CatBoost
     model fitted on the earlier cohorts.
Supports the statement in Section 3.1 that the recorded status of these
cohorts most plausibly reflects incomplete outcome observation.
Prints results; writes no files.
"""
import os
import numpy as np
from scipy import stats as st
from catboost import CatBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, '01_analysis_unified.py'), encoding='utf-8').read()
ns = {'__file__': os.path.join(HERE, '01_analysis_unified.py')}
exec(compile(src.split("def main(")[0], '01_analysis_unified.py', 'exec'), ns)
X, y, cols = ns['load_data']()

ly = X[:, cols.index('loan_year')]
years = np.sort(np.unique(ly))
print('--- cohort default rates (ascending application years) ---')
for i, v in enumerate(years):
    m = ly == v
    print(f'year {2007 + i}: n={m.sum():4d}, defaults={int(y[m].sum()):4d}, '
          f'rate={y[m].mean() * 100:5.1f}%')

last2 = np.isin(ly, years[-2:])
others = ~last2
n2 = int(last2.sum())

ir = st.rankdata(X[:, cols.index('interest_rate')]) / len(y)
cs = st.rankdata(X[:, cols.index('credit_score')]) / len(y)
print('\n--- composition of the last two cohorts (percentile ranks) ---')
print(f'interest rate: median rank {np.median(ir[last2]):.2f} '
      f'(others {np.median(ir[others]):.2f}); '
      f'credit score: {np.median(cs[last2]):.2f} '
      f'(others {np.median(cs[others]):.2f})')

m = others & (ir <= 0.15) & (cs >= 0.60)
p = y[m].mean()
print('\n--- profile-matched earlier loans (interest rank<=0.15, '
      'credit rank>=0.60) ---')
print(f'n={m.sum()}, default rate={p * 100:.1f}%; '
      f'P(0/{n2} | p={p:.3f}) = {(1 - p) ** n2:.2e}')

ns_ = len(cols)
mdl = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                         auto_class_weights='Balanced', random_state=42,
                         verbose=False)
mdl.fit(X[others][:, :ns_], y[others])
p2 = mdl.predict_proba(X[last2][:, :ns_])[:, 1]
print('\n--- model-implied risk of the 181 loans (CatBoost fitted on '
      'earlier cohorts) ---')
print(f'mean prob {p2.mean():.3f}, median {np.median(p2):.3f}, '
      f'min {p2.min():.3f}; expected defaults = {p2.sum():.1f}; '
      f'P(0 defaults) = {np.prod(1 - p2):.2e}')
