# Document Categorization

Multilingual document classification and context-aware tagging for the 01-edu assignment. The production path uses TensorFlow/Keras + `microsoft/mdeberta-v3-base`, a TF-IDF + Logistic Regression baseline, spaCy NER/tagging, English→Spanish MarianMT augmentation, calibrated confidence, XLA inference, and a Streamlit dashboard.

· [Русская версия](README_RU.md)

## 📋 TOC

- [📊 Final results](#-final-results)
- [🚀 Quick start](#-quick-start)
- [📚 Dataset](#-dataset)
- [🧹 Preprocessing](#-preprocessing)
- [🧠 Classifier](#-classifier)
- [🏷️ Context-aware tagging](#️-context-aware-tagging)
- [⚡ Runtime](#-runtime)
- [🧪 Evaluation protocol](#-evaluation-protocol)
- [🖥️ Dashboard](#️-dashboard)
- [✅ Audit checks](#-audit-checks)
- [📁 Project structure](#-project-structure)
- [⚠️ Notes](#️-notes)
- [🧑‍💻 Author](#-author)

## 📊 Final results

The first held-out evaluation did not meet the assignment's `85%` accuracy gate. That result is preserved as Revision 1. A pre-registered post-first-test Revision 2 changed the document representation and validation protocol before retraining, then used one second and final held-out evaluation. Both results remain published.

| Metric | Revision 1 | Revision 2 |
| --- | ---: | ---: |
| Test accuracy | 81.95% | **87.79%** |
| Macro F1 | 81.92% | **87.80%** |
| Throughput | 134.08 docs/s | **130.84 docs/s** |
| English accuracy | 82.79% | **88.85%** |
| Spanish accuracy | 81.12% | **86.73%** |
| Baseline accuracy | 77.81% | 83.14% |
| Relative improvement over baseline | +5.32% | **+5.59%** |
| Assignment gates | FAIL | **PASS** |

Revision 2 satisfies all mandatory assignment gates: accuracy `>=85%`, macro F1 `>=80%`, speed `>=100 docs/s`, per-language accuracy `>=80%`, and relative improvement over baseline `>=5%`.

Evidence:

```text
reports/revision1_performance_metrics.json
reports/revision2/performance_metrics.json
reports/revision2/final_test_consumed.json
reports/performance_metrics.json
```

The second test run is final. `reports/revision2/final_test_consumed.json` records that a third final-test evaluation is forbidden.

## 🚀 Quick start

### Requirements

- Python 3.11 or 3.12 recommended;
- Linux / WSL2 recommended for NVIDIA TensorFlow training;
- NVIDIA GPU strongly recommended for mDeBERTa fine-tuning;
- internet access for the initial dataset/model downloads.

### Clone

```bash
git clone https://01.tomorrow-school.ai/git/nyestaye/document-categorization
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

- `microsoft/mdeberta-v3-base` — production classifier base;
- `Helsinki-NLP/opus-mt-en-es` — EN→ES augmentation;
- `en_core_web_sm` and `es_core_news_sm` — spaCy tagging/NER.

### Prepare Revision 2 data

```bash
python scripts/prepare_data.py
```

Translation is resumable through the separate Revision 2 cache under `data/processed_data/translation_cache_revision2/`.

### Preflight

```bash
python scripts/preflight_revision2.py
```

The preflight reads **train + validation only** and checks dataset invariants, translation sanity, token budget, language detection, and the baseline. It does not read `test.csv`.

### Train

```bash
python scripts/train_revision2.py
```

The production run is frozen to:

| Setting | Value |
| --- | ---: |
| Model | `microsoft/mdeberta-v3-base` |
| Epochs | `5` |
| Learning rate | `2e-5` |
| Batch size | `2` |
| Max model tokens | `512` |
| Classification source window | `150` words |
| Optimizer | AdamW |
| Weight decay | `0.01` |
| Warmup | `10%` |
| Gradient clip | `1.0` |

Checkpoint selection is deterministic: highest validation correct-document count, then lower validation loss, then earlier epoch. Revision 2 selected epoch 5 (`1886/2186`, `86.28%`).

### Calibrate, freeze, verify

```bash
python scripts/calibrate_validation.py
python scripts/freeze_production.py
python scripts/verify_production_validation.py
```

Temperature scaling is fitted on validation only. Revision 2 uses `T=2.9108`; calibration preserves argmax while reducing validation ECE from `0.1292` to `0.0454`.

The frozen production verification remains test-free and reproduced the selected `1886/2186` validation correct count exactly.

### Show final evidence

```bash
python scripts/evaluate.py
```

This command now **only prints the preserved final metrics**. It does not re-read the held-out test because the one allowed Revision 2 final evaluation has already been consumed.

### Validate

```bash
pytest
python scripts/validate_agent_contracts.py
python scripts/validate_project.py
```

## 📚 Dataset

The project uses **20 Newsgroups** as the labeled source and mirrors each retained English document into Spanish with MarianMT.

After Revision 2 cleaning and cross-split deduplication:

- `11,861` independent source pairs;
- `23,722` total EN/ES rows;
- `12` categories;
- `2` languages;
- train: `6,022` source pairs;
- validation: `1,093` source pairs;
- test: `4,746` source pairs.

Revision 2 categories are selected mechanically from cleaned source counts, not model performance:

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

The rule ranks categories by cleaned official-train count with a lexicographic tie-break and takes the smallest prefix whose cleaned train+test source count reaches at least `11,000`.

Every English document and its Spanish translation share one `pair_id` and remain in the same split.

## 🧹 Preprocessing

Revision 2 keeps the document's `Subject` as a legitimate title signal while dropping routing/sender metadata. `sklearn` removes footers and quotes; the project then parses the remaining RFC-style header block itself.

Representation:

```text
Subject without repeated leading Re: markers

Body
```

Other headers are dropped. `Re:` is removed because train-only diagnostics showed strong category association (`Cramér's V = 0.379`, registered threshold `0.10`).

The content pipeline then applies structural cleanup, token-density cleanup, Unicode normalization, and a deterministic `150`-word classification window. Spanish is translated from the cleaned English representation and normalized again after translation.

Revision 2 validation uses a deterministic thread-grouped stratified fallback. The planned temporal split could not be constructed because parseable `Date:` coverage in official train was `0/11,314`; that stop condition was triggered before retraining and documented in Amendment 01. Whole normalized-Subject thread groups are never split between train and validation.

Relevant protocol files:

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

Implementation:

```text
models/text_classifier.py
utils/transfer_learning.py
scripts/train_revision2.py
```

The training loop saves every epoch and writes:

```text
models/checkpoints_revision2/
├── epoch_01.h5 ... epoch_05.h5
├── text_classifier_best.h5
├── text_classifier_best_accuracy.h5
├── config.json
├── training_history.csv
├── best_accuracy_epoch.json
├── baseline_metrics.json
├── optimizer_plan.json
└── token_budget.json
```

Large weight artifacts are intentionally ignored by Git.

### Calibration

Confidence is calibrated with validation-only scalar temperature scaling. It changes confidence values, not class argmax.

## 🏷️ Context-aware tagging

`models/tagger.py` provides language-aware spaCy tagging:

1. detect English or Spanish;
2. route to the matching spaCy pipeline;
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

The frozen production runtime uses:

- float32 inference;
- XLA compilation;
- fixed sequence buckets `64/128/192/256/384/512`;
- attention-balanced classifier batch sizes `32/32/16/16/4/4`;
- parallel CPU spaCy tagging and GPU classification;
- validation-fitted temperature scaling.

Revision 2 final end-to-end throughput was **130.84 docs/s**, including classification and tagging.

Main APIs:

```text
utils/inference.py
utils/production_inference.py
```

## 🧪 Evaluation protocol

Revision 1 remains the original first held-out result and failed only the mandatory `85%` accuracy gate.

Revision 2 was explicitly registered as a **post-first-test protocol revision** before its new preprocessing/training run. The second final evaluation was guarded by an irreversible marker created before reading `test.csv`. The marker is now completed, so a third final-test run is disallowed regardless of outcome.

Statistical evidence on Revision 2 also favors the transformer over the baseline:

- McNemar EN: `434` transformer-only vs `213` baseline-only, `p = 2.44e-18`;
- McNemar ES: `480` transformer-only vs `260` baseline-only, `p = 5.05e-16`;
- pair-cluster bootstrap absolute improvement 95% CI: `+3.76` to `+5.62 pp`;
- pair-cluster bootstrap relative improvement 95% CI: `+4.50%` to `+6.79%`.

The point estimate passes the assignment's relative `+5%` gate. The bootstrap interval is reported as uncertainty and is not substituted for the assignment's point-estimate rule.

## 🖥️ Dashboard

Run:

```bash
streamlit run app/real_time_dashboard.py
```

The dashboard displays category, calibrated confidence, detected language, tags, named entities, final accuracy/F1/throughput, language breakdown, and example predictions.

## ✅ Audit checks

| Audit area | Implementation / evidence |
| --- | --- |
| Recommended dataset | 20 Newsgroups |
| ≥10k documents | 23,722 EN/ES rows / 11,861 source pairs |
| ≥5 categories | 12 categories |
| ≥2 languages | English + Spanish |
| TensorFlow/Keras | `models/text_classifier.py` |
| Transfer learning | mDeBERTa-v3-base |
| ≥5 epochs | 5 epochs |
| LR 2e-5…5e-5 | 2e-5 |
| Validation-loss monitoring | `training_history.csv` |
| Epoch checkpoints | `utils/transfer_learning.py` |
| Baseline | TF-IDF + Logistic Regression |
| spaCy tagging + NER | `models/tagger.py` |
| Real-time batching | production inference pipeline |
| Accuracy ≥85% | 87.79% |
| Macro F1 ≥80% | 87.80% |
| Speed ≥100 docs/s | 130.84 docs/s |
| Per-language accuracy ≥80% | EN 88.85%, ES 86.73% |
| Relative baseline improvement ≥5% | +5.59% |
| Quantization path | `utils/model_optimization.py` |
| Final evidence | `reports/` |

## 📁 Project structure

```text
document-categorization/
├── agent/
├── app/
├── config/
├── data/
├── docs/
├── models/
├── notebooks/
├── reports/
│   └── revision2/
├── scripts/
├── tests/
├── utils/
├── AGENTS.md
├── Makefile
├── README.md
├── README_RU.md
├── pyproject.toml
└── requirements.txt
```

## ⚠️ Notes

- Spanish data is machine-translated augmentation, not independently authored Spanish news.
- Raw/processed datasets and large model weights are reproducible and intentionally excluded from Git.
- Revision 1 and Revision 2 must both remain visible; Revision 2 is not represented as an untouched first test.
- The final held-out test is consumed. Do not delete the consumption marker to run it again.
- `scripts/evaluate.py` is intentionally evidence-only after final evaluation.

## 🧑‍💻 Author

- Nazar Yestayev (@nyestaye)
