# Architecture

This document is architectural context, not normal agent boot context. Production code and runtime reports are the implementation/evidence source of truth.

## Data flow

```text
20 Newsgroups (English)
        |
        +--> normalization + stratified train/validation/test split
        |
        +--> offline MarianMT English -> Spanish augmentation (cached/resumable)
                         |
                         v
                multilingual CSV splits
                         |
             +-----------+-----------+
             |                       |
             v                       v
   TF-IDF + Logistic          multilingual DistilBERT
   Regression baseline        TensorFlow/Keras fine-tuning
             |                       |
             +-----------+-----------+
                         |
                         v
                  batch classifier
                         +----> spaCy language-specific NER/tagging
                         |
                         v
                  inference pipeline
                    /           \
                   v             v
             evaluation       Streamlit dashboard
```

## Dataset decision

The assignment requires one recommended dataset, at least 10,000 documents, at least five categories, and at least two languages. MLDoc is multilingual but has only four classification labels, so the project uses the recommended 20 Newsgroups corpus and creates a Spanish mirror with an offline open-source translation model. Eight source categories are used; mirroring the English data produces more than 10,000 labeled documents while preserving identical class semantics in both languages.

The original English source text is retained conceptually through `source_dataset`, `source_language`, and `is_translation` columns. Generated CSVs are reproducible and intentionally not committed.

## Model

The transfer-learning model is `distilbert/distilbert-base-multilingual-cased`. It is fine-tuned for exactly five epochs by default, with a learning rate of `3e-5`. Every epoch writes a checkpoint. The checkpoint with the minimum validation loss becomes `text_classifier_best.h5`.

The classifier is compared against a word-level TF-IDF + Logistic Regression baseline on the same multilingual split. Evaluation fails the project thresholds if the transformer does not beat the baseline by at least five percentage points.

## Tagging

The tagging layer uses language detection and routes English to `en_core_web_sm` and Spanish to `es_core_news_sm`. Named entities become high-priority context tags; frequent non-stopword lemmas contribute lexical tags. Batch tagging uses `nlp.pipe` for throughput.

## Runtime evidence

Training and evaluation generate:

- `models/checkpoints/training_history.csv`
- `models/checkpoints/config.json`
- `models/checkpoints/best_epoch.json`
- `models/checkpoints/baseline_metrics.json`
- `reports/performance_metrics.json`
- `reports/example_predictions.csv`

Large weights remain local and are ignored by Git.
