# -*- coding: utf-8 -*-
"""Final manuscript figures from the unified re-run (Option A pipeline).
Design: 80/20 stratified splits, seeds 1-50, CatBoost(it=100, depth=5, lr=0.1,
auto_class_weights='Balanced', random_state=42); struct-band uncertainty;
credit-score 30/70 risk tiers; rich/sparse narrative split.
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

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def paired(df, mets, grp_col=None, grp=None):
    src = df if grp is None else df[df[grp_col] == grp]
    piv = src.pivot_table(index='seed', columns='Model', values=mets)
    out = {}
    for met in mets:
        s = piv[(met, 'S')].values; m = piv[(met, 'M')].values
        diff = m - s; mn = diff.mean()
        lo, hi = stats.t.interval(0.95, len(diff)-1, loc=mn, scale=stats.sem(diff))
        t, p = stats.ttest_rel(m, s)
        out[met] = (mn, lo, hi, p)
    return out

# ============ Fig 2: probability distribution ============
hist = pd.read_csv(R + 'fig2_hist.csv')
centers_csv = hist['bin_center'].values
h_norm = hist['mean_repaid'].values; h_def = hist['mean_default'].values
bins = np.arange(0.0, 1.025, 0.025)
share_line = open(R + 'marginal_share.txt').read()
share = float(share_line.split('=')[1]) * 100; n_marg = share / 100 * 1212
centers = (bins[:-1] + bins[1:]) / 2
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.axvspan(0.3, 0.7, color='#f5f0d8', zorder=0)
ax.bar(centers, h_norm, width=0.023, color=G['light'], edgecolor='white', linewidth=0.3,
       label='Actual repaid', zorder=2)
ax.bar(centers, h_def, width=0.023, bottom=h_norm, color=G['dark'], edgecolor='white',
       linewidth=0.3, label='Actual default', zorder=2)
ax.axvline(0.3, color='#333333', ls='--', lw=1.1, zorder=3)
ax.axvline(0.7, color='#333333', ls='-.', lw=1.1, zorder=3)
ymax = (h_norm + h_def).max()
ax.annotate(f'Marginal region  $\\hat{{p}}\\in[0.3,\\,0.7]$\n{share:.1f}% of test observations'
            f' (≈{n_marg:.0f} of 1,212 on average)',
            xy=(0.56, ymax * 1.02), ha='center', va='bottom', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#999999', lw=0.8))
ax.set_xlim(0, 1); ax.set_ylim(0, ymax * 1.24)
ax.set_xlabel('Predicted default probability (structured-only CAT model)')
ax.set_ylabel('Mean frequency per split')
ax.legend(loc='upper left', fontsize=9)
ax.yaxis.grid(True); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FOUT, 'Fig2_probability_distribution_CAT.png'), bbox_inches='tight')
plt.close(fig); print(f'Fig2 done (share {share:.1f}%)')

# ============ Fig 3: threshold sensitivity ============
d4 = pd.read_csv(R + 'optA_t8_raw.csv')
taus = [0.3, 0.4, 0.5, 0.6, 0.7]
styles = {'Overall':   dict(color='#1a1a1a', marker='o', ls='-'),
          'High-Risk': dict(color='#5a5a5a', marker='s', ls='--'),
          'Low-Risk':  dict(color='#9a9a9a', marker='^', ls=':')}
offs = {'Overall': -0.012, 'High-Risk': 0.0, 'Low-Risk': 0.012}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.axhline(0, color='#666666', lw=0.9)
ax.axvspan(0.375, 0.525, color='#f2f2f2', zorder=0)
for g in ['Overall', 'High-Risk', 'Low-Risk']:
    ds, los, his, ps_ = [], [], [], []
    for tau in taus:
        gg = d4[(d4.tau == tau) & (d4.Group == g)].sort_values('seed')
        diff = gg.rm.values - gg.rs.values
        mn = diff.mean()
        lo, hi = stats.t.interval(0.95, len(diff)-1, loc=mn, scale=stats.sem(diff))
        t, p = stats.ttest_rel(gg.rm.values, gg.rs.values)
        ds.append(mn); los.append(mn-lo); his.append(hi-mn); ps_.append(p)
    x = np.array(taus) + offs[g]
    ax.errorbar(x, ds, yerr=[los, his], capsize=3, lw=1.6, elinewidth=1.0, markersize=6,
                markeredgecolor='white', markeredgewidth=0.6, label=g, **styles[g], zorder=3)
    for xi, mn, hi_amt, p in zip(x, ds, his, ps_):
        if stars(p):
            ax.annotate(stars(p), (xi, mn + hi_amt + 0.0005), ha='center', fontsize=8.5,
                        color=styles[g]['color'])
ax.text(0.45, ax.get_ylim()[0] + 0.0008, 'stable positive zone', ha='center',
        fontsize=8.5, color='#555555', style='italic')
ax.set_xticks(taus)
ax.set_xlabel('Classification threshold  $\\tau$')
ax.set_ylabel('$\\Delta$ Recall  (Text-merged $-$ Structured-only)')
ax.legend(loc='lower left', fontsize=9)
ax.yaxis.grid(True); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FOUT, 'Fig3_threshold_sensitivity.png'), bbox_inches='tight')
plt.close(fig); print('Fig3 done')

# ============ Fig 4: forest + FN panel ============
d2 = pd.read_csv(R + 'optA_t6_raw.csv'); d2 = d2.rename(columns={'Rec': 'Recall'})
d9 = pd.read_csv(R + 'optA_t9_raw.csv'); d9 = d9.rename(columns={'Rec': 'Recall'})
mets = ['ROC', 'PR', 'Recall', 'F1']
met_label = {'ROC': 'ROC-AUC', 'PR': 'PR-AUC', 'Recall': 'Recall', 'F1': 'F1'}
groups = [('Marginal cases  ($\\hat{p}\\in[0.3,0.7]$)', d2, 'Marginal'),
          ('Clear cases  ($\\hat{p}<0.3$ or $>0.7$)', d2, 'Clear'),
          ('Rich narratives  (≥1 selected term)', d9, 'Rich'),
          ('Sparse narratives  (no selected term)', d9, 'Sparse')]
rows = []
for title, df, grp in groups:
    rows.append((title, None, None, None, ''))
    eff = paired(df, mets, 'Group', grp)
    for met in mets:
        mn, lo, hi, p = eff[met]
        rows.append(('   ' + met_label[met], mn, lo, hi, stars(p)))

fig = plt.figure(figsize=(8.6, 6.0))
gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.32)
ax = fig.add_subplot(gs[0])
ypos = np.arange(len(rows))[::-1]
ax.axvline(0, color='#666666', lw=0.9)
XL, XR = -0.0135, 0.014
for (label, mn, lo, hi, sig), yv in zip(rows, ypos):
    if mn is None:
        ax.text(XL, yv, label.strip(), fontsize=9.5, fontweight='bold', va='center')
        continue
    ax.plot([lo, hi], [yv, yv], color='#3b3b3b', lw=1.3)
    ax.plot(mn, yv, marker='o', ms=6.5, mfc=('#1a1a1a' if sig else 'white'),
            mec='#1a1a1a', mew=1.1)
    ax.text(XL, yv, label.strip(), fontsize=9, va='center')
    if sig:
        ax.text(hi + 0.0006, yv, sig, fontsize=8.5, va='center')
ax.set_yticks([]); ax.set_xlim(XL - 0.001, XR)
ax.set_ylim(-0.8, len(rows) - 0.2)
ax.set_xlabel('Text effect  ($\\Delta$ = Text-merged $-$ Structured-only)')
ax.set_title('(a) Conditional text effects with 95% CIs', loc='left')
ax.xaxis.grid(True); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)

d3 = pd.read_csv(R + 'optA_t7_raw.csv')
ax2 = fig.add_subplot(gs[1])
tier_names = ['Overall', 'High-Risk', 'Low-Risk']
red, ci_l, ci_h, sig2 = [], [], [], []
for g in tier_names:
    gg = d3[d3.Group == g].sort_values('seed')
    r = gg.rate.values
    m = r.mean(); lo, hi = stats.t.interval(0.95, len(r)-1, loc=m, scale=stats.sem(r))
    t, p = stats.ttest_rel(gg.fs.values, gg.fm.values)
    red.append(m); ci_l.append(lo); ci_h.append(hi); sig2.append(stars(p))
cols = [G['mid'], G['light'], G['dark']]
xb = np.arange(3)
ax2.axhline(0, color='#666666', lw=0.9)
ax2.bar(xb, red, width=0.58, color=cols, edgecolor='white', lw=0.5, zorder=3)
err = np.array([[r - l for r, l in zip(red, ci_l)], [h - r for r, h in zip(red, ci_h)]])
ax2.errorbar(xb, red, yerr=err, fmt='none', ecolor='#1a1a1a', elinewidth=1.0, capsize=3, zorder=4)
for xi, r, s, h in zip(xb, red, sig2, ci_h):
    ax2.text(xi, h + 0.12, f'{r:+.2f}%{s}', ha='center', fontsize=8, fontweight='bold')
ax2.set_ylim(min(ci_l) - 0.5, max(ci_h) + 0.7)
ax2.set_xticks(xb); ax2.set_xticklabels(tier_names, fontsize=8.5, rotation=15)
ax2.set_ylabel('FN reduction rate (%)')
ax2.set_title('(b) False-negative reduction', loc='left')
ax2.yaxis.grid(True); ax2.set_axisbelow(True)
for s in ('top', 'right'): ax2.spines[s].set_visible(False)
fig.savefig(os.path.join(FOUT, 'Fig4_conditional_effects_forest.png'), bbox_inches='tight')
plt.close(fig); print('Fig4 done')
