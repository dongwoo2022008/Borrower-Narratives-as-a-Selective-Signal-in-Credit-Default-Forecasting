# Data

De-identified, model-ready inputs for the analyses in this repository.
N = 6,057 loan applications from a Korean lending platform (2007–2015 application years).

The raw application records and the borrower-written narrative texts are covered by a
confidentiality agreement with the platform and are **not** distributed. What is released
here is exactly what the models consume: standardized numeric features and the sparse
TF–IDF representation of the narratives with anonymized term indices. No personal
information and no text content are included. Reasonable requests for further data access
can be directed to the corresponding author and are subject to the platform's agreement.

## Files

### `structured_features_part1.csv` – `structured_features_part3.csv` (6,057 rows × 14 columns in total)

The 13 structured features plus `row_id`, split into three consecutive row blocks
(rows 0–2018, 2019–4037, 4038–6056) solely to keep individual files small; concatenating
the three parts and sorting by `row_id` restores the full table (see
`code/01_analysis_unified.py::load_data()`). Continuous variables are standardized (zero
mean, unit variance); binary variables are as coded. Column names map to the manuscript's
Table 3:

| Column | Manuscript variable |
|---|---|
| `age` | Age |
| `credit_score` | Credit Score |
| `loan_amount` | Loan Amount |
| `interest_rate` | Loan Interest Rate |
| `loan_year` | Loan Year |
| `loan_purpose` | Loan Purpose (1 = debt repayment, 0 = other) |
| `bank_loan` | Bank Loan |
| `total_count` | Total Count |
| `success_count` | Success Count |
| `success_rate` | Success Rate |
| `num_investors` | Number of Investors |
| `months_of_service` | Months of Service |
| `social_insurance` | Social Insurance (1 = enrolled, 0 = not) |

### `tfidf_features_sparse.csv` (2,689 nonzero entries)

Sparse triplet representation of the 6,057 × 100 TF–IDF block:
`row_id` (0–6056), `term_id` (0–99, anonymized index of the selected TF–IDF vocabulary),
`value` (TF–IDF weight). Rows not listed for a given `term_id` are zero. The vocabulary
itself (Korean terms extracted from the narratives) is not distributed; term identity is
not needed to reproduce any result in the paper.

### `labels.csv` (6,057 rows)

`row_id`, `default` (1 = default, 0 = repaid; default rate 55.34%).

## Notes

- Row order: rows 0–4,844 correspond to the original fixed training partition and rows
  4,845–6,056 to the original fixed test partition used in earlier preprocessing. The
  analyses in this repository do not rely on that ordering — they re-split the full sample
  with stratified 80/20 splits (seeds 1–50).
- Reconstruction: `code/01_analysis_unified.py::load_data()` shows how the dense
  6,057 × 113 design matrix is assembled from these files.
