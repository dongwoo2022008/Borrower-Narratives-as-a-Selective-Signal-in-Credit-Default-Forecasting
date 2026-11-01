# -*- coding: utf-8 -*-
"""Final manuscript figures from the unified re-run (Option A pipeline).
Design: 80/20 stratified splits, seeds 1-50, CatBoost(it=100, depth=5, lr=0.1,
auto_class_weights='Balanced', random_state=42).
Inference displayed in all effect figures uses the Nadeau-Bengio variance
correction (Section 3.6): corrected SE = sd * sqrt(1/J + n_test/n_train)
with J = 50 and n_test/n_train = 0.25.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10.5,
    'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'axes.edgecolor': '#333333', 'axes.linewidth': 0.9,
    'grid.color': '#d9d9d9', 'grid.linewidth': 0.6,
    'legend.frameon': True, 'legend.framealpha': 1.0, 'legend.edgecolor': '#999999',
    'savefig.dpi': 300})
G = {'dark': '#3b3b3b', 'mid': '#7f7f7f', 'light': '#c9c9c9'}
import os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'results', 'raw') + os.sep
FOUT = os.path.join(HERE, '..', 'figures')
os.makedirs(FOUT, exist_ok=True)

J = 50
RHO = 0.25          # n_test / n_train for 80/20 splits
TCRIT = stats.t.ppf(0.975, J - 1)


def nb(diff):
    """Mean difference with Nadeau-Bengio-corrected 95% CI and p-value."""
    diff = np.asarray(diff, dtype=float)
    mn = diff.mean()
    se = diff.std(ddof=1) * np.sqrt(1.0 / J + RHO)
    t = mn / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), J - 1)
    return mn, mn - TCRIT * se, mn + TCRIT * se, p


def paired_diff(df, met, grp_col=None, grp=None):
    src = df if grp is None else df[df[grp_col] == grp]
    piv = src.pivot_table(index='seed', columns='Model', values=met)
    return (piv['M'] - piv['S']).values

# ============ Fig 2: probability distribution ============
hist = pd.read_csv(R + 'fig2_hist.csv')
h_norm = hist['mean_repaid'].values; h_def = hist['mean_default'].values
bins = np.arange(0.0, 1.025, 0.025)
share_line = open(R + 'marginal_share.txt').read()
share = float(share_line.split('=')[1]) * 100; n_marg = share / 100 * 1212
cal = pd.read_csv(R + 'calibration_seed.csv')
bandrows = cal[cal.model == 'band']
fn_share = bandrows.fn_in_band.sum() / bandrows.fn_total.sum() * 100
centers = (bins[:-1] + bins[1:]) / 2
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.axvspan(0.3, 0.7, color='#efefef', zorder=0)
ax.bar(centers, h_norm, width=0.023, color=G['light'], edgecolor='white', linewidth=0.3,
       label='Non-default ($y=0$)', zorder=2)
ax.bar(centers, h_def, width=0.023, bottom=h_norm, color=G['dark'], edgecolor='white',
       linewidth=0.3, label='Default ($y=1$)', zorder=2)
ax.axvline(0.3, color='#333333', ls='--', lw=1.1, zorder=3)
ax.axvline(0.7, color='#333333', ls='--', lw=1.1, zorder=3)
ax.axvline(0.5, color='#666666', ls=':', lw=1.3, zorder=3)
ymax = (h_norm + h_def).max()
ax.text(0.502, ymax * 0.99, '$\\tau=0.5$', fontsize=8.5, color='#555555',
        ha='left', va='top', rotation=90)
ax.annotate(f'Marginal band  $\\hat{{p}}\\in[0.3,\\,0.7]$\n{share:.1f}% of test observations'
            f' (≈{n_marg:.0f} of 1,212 on average);\n{fn_share:.1f}% of structured-only'
            f' false negatives fall here',
            xy=(0.56, ymax * 1.02), ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#999999', lw=0.8))
ax.set_xlim(0, 1); ax.set_ylim(0, ymax * 1.33)
ax.set_xlabel('Predicted default probability (structured-only CAT model)')
ax.set_ylabel('Mean frequency per split')
ax.legend(loc='upper left', fontsize=9)
ax.yaxis.grid(True); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FOUT, 'Fig2_probability_distribution_CAT.png'), bbox_inches='tight')
plt.close(fig); print(f'Fig2 done (share {share:.1f}%, FN share {fn_share:.1f}%)')

# ============ Fig 4: threshold sensitivity (corrected CIs, 3 panels) ============
d4 = pd.read_csv(R + 'optA_t8_raw.csv')
taus = [0.3, 0.4, 0.5, 0.6, 0.7]
panels = ['Overall', 'High-Risk', 'Low-Risk']
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.5), sharey=True)
for ax, g in zip(axes, panels):
    ax.axhline(0, color='#666666', lw=0.9)
    ds, los, his = [], [], []
    for tau in taus:
        gg = d4[(d4.tau == tau) & (d4.Group == g)].sort_values('seed')
        mn, lo, hi, p = nb(gg.rm.values - gg.rs.values)
        ds.append(mn * 100); los.append((mn - lo) * 100); his.append((hi - mn) * 100)
    ax.errorbar(taus, ds, yerr=[los, his], capsize=3, lw=1.5, elinewidth=1.0,
                markersize=6, marker='o', ls='-', color='#2b2b2b',
                markerfacecolor='white', markeredgecolor='#2b2b2b',
                markeredgewidth=1.1, zorder=3)
    ax.set_title(g, fontsize=12)
    ax.set_xticks(taus)
    ax.set_xlabel('Threshold  $\\tau$', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axes[0].set_ylabel('$\\Delta$ Recall (pp)\nText-merged $-$ Structured-only', fontsize=11.5)
fig.suptitle('95% CIs, Nadeau–Bengio corrected', x=0.99, y=0.99,
             ha='right', fontsize=10, fontweight='normal', style='italic',
             color='#555555')
fig.tight_layout()
fig.savefig(os.path.join(FOUT, 'Fig4_threshold_sensitivity.png'), bbox_inches='tight')
plt.close(fig); print('Fig4 done')

# ============ Fig 5: detection effects summary (corrected CIs) ============
d1 = pd.read_csv(R + 'optA_t5_raw.csv')
d2 = pd.read_csv(R + 'optA_t6_raw.csv')
d9 = pd.read_csv(R + 'optA_t9_raw.csv')
d3 = pd.read_csv(R + 'optA_t7_raw.csv')

rows_a = [
    ('Overall', paired_diff(d1, 'Rec')),
    ('Marginal  ($\\hat{p}\\in[0.3,0.7]$)', paired_diff(d2, 'Rec', 'Group', 'Marginal')),
    ('Clear  ($\\hat{p}<0.3$ or $>0.7$)', paired_diff(d2, 'Rec', 'Group', 'Clear')),
    ('Contains selected term', paired_diff(d9, 'Rec', 'Group', 'Rich')),
    ('No selected term', paired_diff(d9, 'Rec', 'Group', 'Sparse')),
]
rows_b = []
for g in ['Overall', 'High-Risk', 'Low-Risk']:
    gg = d3[d3.Group == g].sort_values('seed')
    rows_b.append((g, (gg.fs - gg.fm).values.astype(float)))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.9),
                              gridspec_kw={'width_ratios': [1.45, 1.0],
                                           'wspace': 0.52})
for axx, rows, xlab, title in (
        (ax, rows_a, '$\\Delta$ Recall (pp)',
         '(a) Change in Recall, Text-merged $-$ Structured-only'),
        (ax2, rows_b, 'False negatives removed (count per test set)',
         '(b) Reduction in false-negative counts')):
    ypos = np.arange(len(rows))[::-1]
    axx.axvline(0, color='#666666', lw=0.9)
    for (label, diff), yv in zip(rows, ypos):
        scale = 100.0 if axx is ax else 1.0
        mn, lo, hi, p = nb(diff)
        mn, lo, hi = mn * scale, lo * scale, hi * scale
        axx.plot([lo, hi], [yv, yv], color='#3b3b3b', lw=1.4,
                 solid_capstyle='butt')
        for xend in (lo, hi):
            axx.plot([xend, xend], [yv - 0.1, yv + 0.1], color='#3b3b3b', lw=1.2)
        axx.plot(mn, yv, marker='o', ms=6, mfc='white', mec='#1a1a1a', mew=1.2)
    axx.set_yticks(ypos)
    axx.set_yticklabels([r[0] for r in rows], fontsize=11)
    axx.set_ylim(-0.6, len(rows) - 0.4)
    axx.set_xlabel(xlab, fontsize=11)
    axx.set_title(title, loc='left', fontsize=11)
    axx.tick_params(axis='x', labelsize=10.5)
    axx.xaxis.grid(True); axx.set_axisbelow(True)
    for s in ('top', 'right', 'left'): axx.spines[s].set_visible(False)
    axx.tick_params(left=False)
fig.suptitle('Dots: means over 50 splits; lines: 95% CIs, Nadeau–Bengio corrected',
             x=0.99, y=1.005, ha='right', fontsize=10, fontweight='normal',
             style='italic', color='#555555')
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(FOUT, 'Fig5_conditional_effects_forest.png'), bbox_inches='tight')
plt.close(fig); print('Fig5 done')
