# Document Categorization

Multilingual document intelligence pipeline for English and Spanish text. The system combines TensorFlow/Keras + `microsoft/mdeberta-v3-base` classification, TF-IDF + Logistic Regression benchmarking, spaCy NER and context-aware tagging, MarianMT augmentation, calibrated confidence, XLA-accelerated inference, and an interactive Streamlit dashboard.

· [Русская версия](README_RU.md)

## 📋 TOC

- [📊 Performance](#-performance)
- [🧩 What it does](#-what-it-does)
- [🚀 Quick start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [📚 Dataset](#-dataset)
- [🧹 Preprocessing](#-preprocessing)
- [🧠 Classifier](#-classifier)
- [🏷️ Context-aware tagging](#️-context-aware-tagging)
- [⚡ Runtime](#-runtime)
- [🧪 Evaluation and reproducibility](#-evaluation-and-reproducibility)
- [🖥️ Dashboard](#️-dashboard)
- [📁 Project structure](#-project-structure)
- [⚠️ Notes](#️-notes)
- [🧑‍💻 Author](#-author)

## 📊 Performance

Current model: **Revision 2 / mDeBERTa-v3-base**.

| Metric | Revision 1 | Revision 2 |
| --- | ---: | ---: |
| Test accuracy | 81.95% | **87.79%** |
| Macro F1 | 81.92% | **87.80%** |
| End-to-end throughput | 134.08 docs/s | **130.84 docs/s** |
| English accuracy | 82.79% | **88.85%** |
| Spanish accuracy | 81.12% | **86.73%** |
| TF-IDF + Logistic Regression baseline | 77.81% | 83.14% |
| Relative improvement over baseline | +5.32% | **+5.59%** |
| Mean calibrated confidence | 88.02% | **88.14%** |

Revision 2 is the current production candidate. It reaches **87.79% test accuracy** while sustaining **130.84 documents/sec** through the complete classification + language detection + spaCy tagging path.

The two revisions are preserved as development history, not as a controlled A/B test: Revision 2 changed the document representation, category selection, validation construction, and preprocessing policy before retraining. The important controlled comparison inside each revision is the transformer against the baseline trained on the same data.

Final evidence is stored in:

```text
reports/revision1_performance_metrics.json
reports/revision2/performance_metrics.json
reports/performance_metrics.json
reports/revision2/example_predictions.csv
```

## 🧩 What it does

For an English or Spanish document, the pipeline can:

- detect the document language;
- classify it into one of 12 topical categories;
- return calibrated confidence;
- extract named entities;
- generate context-aware tags;
- process documents individually or in batches;
- expose results through a Streamlit dashboard;
- report measured accuracy, F1, language-level quality, confidence calibration, and throughput.

The runtime separates GPU-heavy transformer classification from CPU-heavy spaCy processing and overlaps both stages for higher throughput.

## 🚀 Quick start

### Requirements

- Python 3.11 or 3.12 recommended;
- Linux / WSL2 recommended for NVIDIA TensorFlow training;
- NVIDIA GPU strongly recommended for mDeBERTa fine-tuning;
- internet access for the initial dataset and model downloads.

### Clone

```bash
git clone https://github.com/legion2440/document-categorization.git
cd document-categorization
```

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "tensorflow[and-cuda]==2.21.0"
```

Under WSL2, TensorFlow may need the NVIDIA library path in each shell:

```bash
export LD_LIBRARY_PATH=$(python - <<'PY'
import site, glob
paths = []
for root in site.getsitepackages():
    paths += glob.glob(root + "/nvidia/*/lib")
print(":".join(paths))
PY
):/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
```

### Download model dependencies

```bash
python scripts/download_models.py
```

This caches:

- `microsoft/mdeberta-v3-base` — classifier base model;
- `Helsinki-NLP/opus-mt-en-es` — EN→ES augmentation;
- `en_core_web_sm` and `es_core_news_sm` — spaCy tagging and NER.

### Prepare data

```bash
python scripts/prepare_data.py
```

Translation is resumable through `data/processed_data/translation_cache_revision2/`.

### Preflight

```bash
python scripts/preflight_revision2.py
```

The preflight validates train/validation invariants, translation sanity, tokenizer budget, language detection, and the classical baseline without using held-out test metrics.

### Train

```bash
python scripts/train_revision2.py
```

### Calibrate and freeze

```bash
python scripts/calibrate_validation.py
python scripts/freeze_production.py
python scripts/verify_production_validation.py
```

### Inspect preserved final metrics

```bash
python scripts/evaluate.py
```

After the final held-out evaluation this command is evidence-only; it prints the preserved metrics instead of re-running the test.

### Validate

```bash
pytest
python scripts/validate_agent_contracts.py
python scripts/validate_project.py
```

### Dashboard

```bash
streamlit run app/real_time_dashboard.py
```

## 🏗️ Architecture

```text
20 Newsgroups
      │
      ├── header-aware cleanup ── Subject + body
      │
      ├── structural / token-density cleanup
      │
      ├── deterministic EN source documents
      │          │
      │          └── MarianMT EN→ES augmentation
      │
      └── paired EN/ES train · validation · test
                 │
                 ├── mDeBERTa-v3-base classifier ──┐
                 │                                  ├── calibrated result
                 └── spaCy NER + context tags ─────┘
                                                    │
                                      batch API / Streamlit dashboard
```

Core runtime modules:

```text
models/text_classifier.py
models/tagger.py
utils/inference.py
utils/production_inference.py
```

## 📚 Dataset

The project uses **20 Newsgroups** as the labeled source and mirrors each retained English document into Spanish with MarianMT.

Current processed corpus:

- `11,861` source pairs;
- `23,722` total EN/ES rows;
- `12` categories;
- `2` languages;
- train: `6,022` source pairs;
- validation: `1,093` source pairs;
- test: `4,746` source pairs.

Categories:

```text
rec.sport.hockey
soc.religion.christian
rec.motorcycles
rec.sport.baseball
sci.crypt
rec.autos
sci.med
comp.windows.x
sci.space
comp.os.ms-windows.misc
sci.electronics
comp.sys.ibm.pc.hardware
```

Category selection is deterministic and based on cleaned source counts, not model scores. Every English source document and its Spanish translation share one `pair_id` and remain in the same split.

## 🧹 Preprocessing

Revision 2 keeps `Subject` as document-title content while dropping routing and sender metadata. `sklearn` removes footers and quotes; the project parses the remaining RFC-style header block itself.

Classifier representation:

```text
Subject without repeated leading Re: markers

Body
```

The pipeline then applies:

- Unicode normalization;
- structural-noise removal;
- token-density cleanup;
- whitespace normalization;
- deterministic `150`-word classification window;
- post-translation normalization for Spanish.

Repeated leading `Re:` markers are removed because train-only diagnostics showed that their presence was strongly category-associated (`Cramér's V = 0.379`).

Validation is deterministic and thread-grouped: documents sharing the same normalized subject within a category are kept together instead of being split across train and validation.

The original temporal-validation design could not be implemented because the official corpus provided no usable `Date:` coverage. The fallback decision and its rationale are preserved in:

```text
config/revision2_protocol.json
config/revision2_amendment_01.json
docs/revision2_preregistration.md
docs/revision2_amendment_01.md
```

## 🧠 Classifier

### Baseline

```text
TF-IDF word unigrams → Logistic Regression
```

Implementation: `models/baseline.py`.

### Transfer learning

Production model:

```text
microsoft/mdeberta-v3-base
```

Frozen training configuration:

| Setting | Value |
| --- | ---: |
| Epochs | `5` |
| Learning rate | `2e-5` |
| Batch size | `2` |
| Max model tokens | `512` |
| Source text window | `150` words |
| Optimizer | AdamW |
| Weight decay | `0.01` |
| Warmup | `10%` |
| Gradient clip | `1.0` |
| Random seed | `42` |

Checkpoint selection is deterministic: highest validation correct-document count, then lower validation loss, then earlier epoch. Revision 2 selected epoch 5 with `1886/2186` validation documents correct (`86.28%`).

Training artifacts include epoch checkpoints, runtime config, history, baseline metrics, optimizer plan, and token-budget diagnostics. Large weights are intentionally excluded from Git.

### Confidence calibration

Scalar temperature scaling is fitted on validation only. Revision 2 uses `T = 2.9108`.

Calibration preserved class argmax while improving validation calibration:

| Metric | Before | After |
| --- | ---: | ---: |
| Mean confidence | 98.97% | 87.83% |
| NLL | 1.3163 | 0.5928 |
| ECE | 0.1292 | 0.0454 |
| Brier score | 0.2648 | 0.2304 |

## 🏷️ Context-aware tagging

`models/tagger.py` provides language-aware spaCy tagging:

1. detect English or Spanish;
2. route to the matching spaCy model;
3. extract named entities;
4. prioritize entities as context tags;
5. add frequent meaningful lemmas;
6. deduplicate tags;
7. process batches with `nlp.pipe`.

| Language | spaCy model |
| --- | --- |
| English | `en_core_web_sm` |
| Spanish | `es_core_news_sm` |

The production tagger uses a `75`-word window while classification uses `150` words.

## ⚡ Runtime

The frozen inference path uses:

- float32 TensorFlow inference;
- XLA compilation;
- fixed token buckets `64 / 128 / 192 / 256 / 384 / 512`;
- attention-balanced classifier batch sizes `32 / 32 / 16 / 16 / 4 / 4`;
- parallel CPU spaCy tagging and GPU transformer classification;
- validation-fitted temperature scaling.

Measured Revision 2 end-to-end throughput: **130.84 docs/s**.

Frozen validation verification reproduced the selected checkpoint exactly (`1886/2186` correct) at **112.36 docs/s** before the final held-out evaluation.

## 🧪 Evaluation and reproducibility

The repository preserves two model-development revisions.

**Revision 1** exposed a generalization gap on the first held-out evaluation: `81.95%` test accuracy despite stronger validation performance. Instead of tuning repeatedly against the test set, the result was preserved and a new data/model protocol was written before retraining.

**Revision 2** introduced the current title-aware representation, deterministic grouped validation, a fresh translation cache, a new validation-only calibration fit, and a new mDeBERTa training run. Its held-out result is `87.79%` accuracy / `87.80%` macro F1.

The original Revision 1 result remains published alongside Revision 2. Revision 2 also stores an immutable evaluation marker so the held-out result is not silently regenerated after seeing the outcome.

Statistical comparison against the same-split classical baseline:

- McNemar EN: `434` transformer-only vs `213` baseline-only, `p = 2.44e-18`;
- McNemar ES: `480` transformer-only vs `260` baseline-only, `p = 5.05e-16`;
- pair-cluster bootstrap absolute improvement 95% CI: `+3.76` to `+5.62` percentage points;
- pair-cluster bootstrap relative improvement 95% CI: `+4.50%` to `+6.79%`.

Reproducibility evidence:

```text
reports/revision2_preflight.json
models/checkpoints/production_validation_verification.json
reports/revision1_performance_metrics.json
reports/revision2/performance_metrics.json
reports/revision2/final_test_consumed.json
```

## 🖥️ Dashboard

`app/real_time_dashboard.py` displays:

- predicted category;
- calibrated confidence;
- detected language;
- context tags;
- named entities;
- final accuracy / F1 / throughput;
- per-language metrics;
- category and tag distributions;
- example predictions.

Run it with:

```bash
streamlit run app/real_time_dashboard.py
```

## 📁 Project structure

```text
document-categorization/
├── agent/                         # repository navigation metadata
├── app/                           # Streamlit dashboard
├── config/                        # reproducibility protocol
├── data/                          # generated datasets / caches
├── docs/                          # architecture and protocol notes
├── models/                        # classifier, baseline, tagger, checkpoints
├── notebooks/                     # EDA and training notebook
├── reports/                       # measured runtime/evaluation evidence
│   └── revision2/
├── scripts/                       # data, train, calibration, freeze, evaluation
├── tests/
├── utils/                         # preprocessing, inference, translation
├── AGENTS.md
├── Makefile
├── README.md
├── README_RU.md
├── pyproject.toml
└── requirements.txt
```

## ⚠️ Notes

- Spanish documents are machine-translated augmentation, not an independently authored Spanish corpus.
- Raw/processed datasets and large model weights are reproducible and intentionally excluded from Git.
- Revision 1 and Revision 2 use different data policies, so cross-revision metric changes should be read as development history rather than a strict same-sample experiment.
- `scripts/evaluate.py` is evidence-only after the preserved final evaluation.

## 🧑‍💻 Author

- Nazar Yestayev (@nyestaye)
