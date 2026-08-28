## Table B1. Text Effects by Text Representation (Δ = Text-merged − Structured-only)

Design identical to the main analysis (50 stratified 80/20 splits; fixed CAT specification;
struct-band uncertainty; credit-score 30/70 risk tiers). Dense embeddings: MiniLM (384-d,
mean pooling) and KoSimCSE (768-d), concatenated with the same 13 structured features.

| Condition / Metric | TF–IDF (main) | MiniLM | KoSimCSE |
|---|---|---|---|
| Overall ΔROC-AUC | +0.00001 | −0.01287*** | −0.01601*** |
| Overall ΔPR-AUC | −0.00055 | −0.01191*** | −0.01571*** |
| Overall ΔRecall | +0.00277** | −0.00155 | −0.00125 |
| Overall ΔF1 | +0.00042 | −0.00669*** | −0.00902*** |
| Marginal ΔROC-AUC | +0.00114 | −0.03125*** | −0.04007*** |
| Marginal ΔRecall | +0.00639* | −0.00121 | −0.00081 |
| FN reduction rate — Overall | +0.99%** | −0.68% | −0.64% |
| FN reduction rate — High-Risk | −0.12% | +0.24% | +0.47% |
| FN reduction rate — Low-Risk | +1.35%* | −0.37% | −0.01% |

Significance from two-sided paired t-tests across seeds: *** p < 0.001, ** p < 0.01, * p < 0.05.
Per-seed records: `results/raw/rob_{MiniLM,KoSimCSE}_t{5,6,7,8}.csv`; script:
`code/05_robustness_embeddings.py`.
