# -*- coding: utf-8 -*-
"""Figure 1: research pipeline diagram (matplotlib, grayscale) for docx embedding."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13.2, 4.4))
ax.set_xlim(0, 13.2); ax.set_ylim(0, 4.4); ax.axis('off')

def box(x, y, w, h, title, lines, fc='#ffffff', title_fs=9.8, fs=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.06',
                       fc=fc, ec='#333333', lw=1.2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h - 0.3, title, ha='center', va='center',
            fontsize=title_fs, fontweight='bold')
    body = '\n'.join(lines)
    ax.text(x + w/2, y + (h - 0.52)/2, body, ha='center', va='center', fontsize=fs)

def arrow(x1, y1, x2, y2, label=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16,
                        color='#333333', lw=1.4, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.14, label, ha='center', fontsize=8.5,
                style='italic', color='#333333')

# Stage 1: data
box(0.15, 1.35, 2.1, 1.7, 'Loan applications',
    ['N = 6,057', 'default 55.3%', '', '80/20 stratified splits', 'seeds 1–50'], fc='#f2f2f2')
# Stage 2: features (two boxes)
box(2.62, 2.45, 2.62, 1.45, 'Structured features (13)',
    ['borrower attributes,', 'borrowing history,', 'loan characteristics'])
box(2.62, 0.5, 2.62, 1.45, 'Text features (100)',
    ['narratives → tokenization,', 'stop-word removal,', 'TF–IDF (top 100)'])
# Stage 3: configurations
box(5.6, 1.2, 2.3, 2.0, 'Two configurations',
    ['① Structured-only (13)', '② Text-merged (113)', '', 'identical splits, model,', 'and seeds'], fc='#e8e8e8')
# Stage 4: model
box(8.4, 1.2, 2.15, 2.0, 'Base learner: CAT',
    ['selected from 10', 'algorithms (Table 4);', 'fixed specification', 'for both configurations'])
# Stage 5: conditional framework
box(11.0, 0.28, 2.05, 3.85, 'Conditional effects',
    ['D1 uncertainty', '(p̂ ∈ [0.3, 0.7])', 'D2 FN dynamics', '(credit-score tiers)',
     'D3 thresholds', '(τ = 0.3–0.7)', 'D4 informativeness', '(rich vs. sparse text)',
     '', 'ΔROC-AUC, ΔPR-AUC,', 'ΔRecall, ΔF1;', 'paired t-tests, 95% CIs'], fc='#e8e8e8', fs=8.6)

arrow(2.25, 2.5, 2.62, 3.1)
arrow(2.25, 1.9, 2.62, 1.3)
arrow(5.24, 3.15, 5.6, 2.6)
arrow(5.24, 1.2, 5.6, 1.75)
arrow(7.9, 2.2, 8.4, 2.2)
arrow(10.55, 2.2, 11.0, 2.2)

import os
FOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FOUT, exist_ok=True)
fig.savefig(os.path.join(FOUT, 'Fig1_research_pipeline.png'), dpi=300, bbox_inches='tight',
            facecolor='white')
print('fig1 saved')
