# -*- coding: utf-8 -*-
"""Figure 1: paired benchmark design and sensitivity analyses.

Depicts the actual order of operations (full-sample preprocessing before the
repeated splits), the paired model configurations, the Nadeau-Bengio-corrected
inference standard, and the separate sensitivity analyses.
Output: ../figures/Fig1_research_pipeline.png (300 dpi)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FOUT = os.path.join(HERE, '..', 'figures')
os.makedirs(FOUT, exist_ok=True)

plt.rcParams.update({'font.size': 9.5, 'savefig.dpi': 300})

fig, ax = plt.subplots(figsize=(11.6, 7.1))
ax.set_xlim(0, 11.6); ax.set_ylim(0, 7.05); ax.axis('off')


def box(x, y, w, h, title, lines, fc='#f4f4f4', ec='#3b3b3b', title_fs=10,
        body_fs=8.8, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                fc=fc, ec=ec, lw=1.1, linestyle=ls))
    ax.text(x + w / 2, y + h - 0.28, title, ha='center', va='center',
            fontsize=title_fs, fontweight='bold', color='#1a1a1a')
    body = '\n'.join(lines)
    ax.text(x + w / 2, y + (h - 0.52) / 2, body, ha='center', va='center',
            fontsize=body_fs, color='#222222', linespacing=1.45)


def arrow(x1, y1, x2, y2, ls='-', ec='#3b3b3b'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=14,
                                 lw=1.3, color=ec, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


MX, MW = 0.45, 6.7          # main column
Y1, Y2, Y3, Y4, Y5 = 5.65, 4.30, 2.85, 1.40, 0.12
H1, H2, H3, H4, H5 = 1.25, 1.05, 1.15, 1.15, 0.98

box(MX, Y1, MW, H1, '1.  Data and full-sample preprocessing',
    ['6,057 online lending applications, 2007–2015 (default rate 55.3%)',
     'Structured features (13): rank-based quantile-normal transform;',
     'borrower narratives: TF–IDF, 100 selected terms — both fitted on the full sample'],
    fc='#ececec')
box(MX, Y2, MW, H2, '2.  Repeated evaluation splits',
    ['50 stratified 80/20 train/test splits (seeds 1–50);',
     'test sets overlap across repetitions'],
    fc='#f4f4f4')
box(MX, Y3, MW, H3, '3.  Paired configurations within each split',
    ['Structured-only: CatBoost on 13 features   vs.   Text-merged: identical',
     'CatBoost on 13 + 100 features — same training rows, same hyperparameters'],
    fc='#f4f4f4')
box(MX, Y4, MW, H4, '4.  Paired evaluation on each test set',
    ['ROC-AUC, PR-AUC, Recall, F1;  thresholds $\\tau \\in \\{0.3,\\ldots,0.7\\}$;',
     'subgroups: marginal band, risk tiers, narrative terms;  calibration'],
    fc='#f4f4f4')
box(MX, Y5, MW, H5, '5.  Inference standard',
    ['Nadeau–Bengio-corrected paired tests (primary);',
     'nominal paired $t$-tests reported for reference only'],
    fc='#dedede')

cx = MX + MW / 2
arrow(cx, Y1, cx, Y2 + H2 + 0.10)
arrow(cx, Y2, cx, Y3 + H3 + 0.10)
arrow(cx, Y3, cx, Y4 + H4 + 0.10)
arrow(cx, Y4, cx, Y5 + H5 + 0.10)

# ---- sensitivity analyses (separate block) ----
SX, SW = 7.75, 3.55
SY, SH = 1.75, 3.4
box(SX, SY, SW, SH, 'Sensitivity analyses',
    ['Embedding-based text representations',
     '(Appendix I)',
     '',
     'Fold-internal preprocessing',
     '(Appendix II)',
     '',
     'Recent-cohort exclusion',
     '(Section 3.1)',
     '',
     'Gating-candidate diagnostic',
     '(Section 5.3)'],
    fc='#fbfbfb', ls='--', body_fs=8.6)
arrow(MX + MW + 0.12, Y3 + H3 / 2, SX - 0.06, SY + SH / 2, ls='--', ec='#7f7f7f')
ax.text(SX + SW / 2, SY - 0.22,
        'Each repeats the paired comparison\nunder an alternative design choice',
        ha='center', va='top', fontsize=8.2, style='italic', color='#555555')

fig.savefig(os.path.join(FOUT, 'Fig1_research_pipeline.png'), bbox_inches='tight')
plt.close(fig)
print('saved Fig1_research_pipeline.png')
