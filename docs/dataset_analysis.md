# Dataset analysis and EDA plan

## Constraint analysis

The recommended datasets do not all satisfy the assignment's combined minimums without adaptation:

- 20 Newsgroups: 18,846 documents and 20 categories, but English-only.
- MLDoc: multilingual, but only four labels, below the required five categories.
- Reuters-21578: many categories, but the recommended source is English-only.

The project therefore uses 20 Newsgroups as the canonical labeled source and adds Spanish by offline translation while preserving the original labels. This keeps category semantics aligned across languages and avoids combining incompatible taxonomies.

## EDA performed by the notebook

`notebooks/EDA_and_Training.ipynb` computes and visualizes from the real generated CSV files:

1. dataset size and audit minimums;
2. train/validation/test sizes;
3. class distribution by split;
4. language distribution by split;
5. category × language balance;
6. document character and word-length distributions;
7. duplicate rate;
8. empty/missing text checks;
9. sample documents per language/category;
10. baseline metrics and transfer-learning history;
11. validation-loss and accuracy curves;
12. final performance report and threshold checks.

No fabricated metric values are committed. EDA outputs become real only after `scripts/prepare_data.py` has downloaded/translated the dataset and the notebook is executed.
