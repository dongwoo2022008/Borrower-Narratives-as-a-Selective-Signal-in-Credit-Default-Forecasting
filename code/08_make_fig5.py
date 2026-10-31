# -*- coding: utf-8 -*-
"""Figure 5: reliability (calibration) curve of the structured-only CAT model.

Input : ../results/raw/calibration_bins.csv (from 07_calibration.py)
Output: ../figures/Fig5_calibration.png (600 dpi)
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
    'o': np.average(d['obs_rate'], weights=d['n'])}), include_groups=False)
spread = (b.groupby(['bin', 'seed'])
            .apply(lambda d: np.average(d['obs_rate'], weights=d['n']),
                   include_groups=False)
            .groupby('bin').std())

fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color='0.6', label='Perfect calibration')
ax.errorbar(rel['p'], rel['o'], yerr=1.96 * spread / np.sqrt(50), fmt='o-',
            color='0.15', lw=1.2, ms=4, capsize=2,
            label='Structured-only CAT (mean over 50 splits)')
ax.axvspan(0.3, 0.7, color='0.85', zorder=0)
ax.text(0.5, 0.06, 'Marginal band\n[0.3, 0.7]', ha='center', fontsize=8, color='0.3')
ax.set_xlabel('Mean predicted default probability (bin)')
ax.set_ylabel('Observed default rate')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'Fig5_calibration.png'), bbox_inches='tight')
print('saved Fig5_calibration.png')
