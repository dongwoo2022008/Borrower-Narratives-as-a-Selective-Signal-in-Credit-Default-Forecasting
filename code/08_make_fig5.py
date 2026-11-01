# -*- coding: utf-8 -*-
"""Figure 3: held-out calibration of the structured-only CAT model.

Reliability curve averaged over 50 held-out test sets; the shaded band shows
across-split variability (±1 SD of the per-split observed default rate per
bin); point size is proportional to the mean number of observations per bin.
Input : ../results/raw/calibration_bins.csv, calibration_seed.csv
Output: ../figures/Fig3_calibration.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, '..', 'results', 'raw')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 9,
                     'axes.labelsize': 9, 'figure.dpi': 110, 'savefig.dpi': 600,
                     'axes.spines.top': False, 'axes.spines.right': False})

b = pd.read_csv(os.path.join(RAW, 'calibration_bins.csv'))
rel = b.groupby('bin').apply(lambda d: pd.Series({
    'p': np.average(d['p_mean'], weights=d['n']),
    'o': np.average(d['obs_rate'], weights=d['n']),
    'n': d['n'].mean()}), include_groups=False)
spread = (b.groupby(['bin', 'seed'])
            .apply(lambda d: np.average(d['obs_rate'], weights=d['n']),
                   include_groups=False)
            .groupby('bin').std())

seed = pd.read_csv(os.path.join(RAW, 'calibration_seed.csv'))
s_rows = seed[seed.model == 'S']
ece, brier = s_rows.ece.mean(), s_rows.brier.mean()

fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color='0.6', label='Perfect calibration')
ax.fill_between(rel['p'], rel['o'] - spread.values, rel['o'] + spread.values,
                color='0.75', alpha=0.45, lw=0,
                label='±1 SD across 50 splits')
ax.plot(rel['p'], rel['o'], '-', color='0.15', lw=1.1, zorder=3)
ax.scatter(rel['p'], rel['o'], s=rel['n'] / rel['n'].max() * 130 + 8,
           color='0.15', zorder=4,
           label='Structured-only CAT (mean over 50 splits)')
ax.axvspan(0.3, 0.7, color='0.92', zorder=0)
ax.text(0.5, 0.045, 'Marginal band\n[0.3, 0.7]', ha='center', fontsize=7.5,
        color='0.35')
ax.text(0.97, 0.16, f'ECE = {ece:.3f}\nBrier = {brier:.3f}', ha='right',
        va='bottom', fontsize=8, color='0.15',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', lw=0.6))
ax.set_xlabel('Mean predicted default probability (bin)')
ax.set_ylabel('Observed default rate')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=7.2, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'Fig3_calibration.png'), bbox_inches='tight')
print(f'saved Fig3_calibration.png (point size ~ bin n; ECE {ece:.3f}, Brier {brier:.3f})')
