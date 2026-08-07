# Document Categorization

An intelligent multilingual document categorization and context-aware tagging system built for the 01-edu assignment. The project combines a TensorFlow/Keras multilingual DistilBERT classifier, a TF-IDF + Logistic Regression baseline, spaCy NER/tagging, resumable English→Spanish dataset augmentation, end-to-end performance evaluation, and a Streamlit dashboard.

The repository contains the complete training/evaluation pipeline but does **not** commit fabricated metrics or large model weights. Real checkpoints, training history and reports are produced when the pipeline is executed.

· [Русская версия](README_RU.md)

## 📋 TOC

- [🚀 Quick start](#-quick-start)
- [📝 About](#-about)
- [📚 Dataset](#-dataset)
- [🧹 EDA and preprocessing](#-eda-and-preprocessing)
- [🧠 Models](#-models)
- [🏷️ Context-aware tagging](#️-context-aware-tagging)
- [⚡ Real-time pipeline](#-real-time-pipeline)
- [📊 Evaluation and thresholds](#-evaluation-and-thresholds)
- [🖥️ Dashboard](#️-dashboard)
- [🧪 Tests and audit verification](#-tests-and-audit-verification)
- [📁 Project structure](#-project-structure)
- [⚠️ Notes](#️-notes)
- [🧑‍💻 Author](#-author)

## 🚀 Quick start

### Requirements

- Python 3.11 or 3.12 recommended;
- Linux / WSL2 recommended for NVIDIA GPU training;
- enough disk space for TensorFlow, Hugging Face caches, translated data and local checkpoints;
- internet access for the first dataset/model download.

### Clone

```bash
git clone https://01.tomorrow-school.ai/git/nyestaye/document-categorization
cd document-categorization
```

### Create the environment

Linux / WSL2:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPU use under WSL2, install TensorFlow's CUDA extra after the base requirements:

```bash
pip install "tensorflow[and-cuda]==2.21.0"
```

### Download model dependencies

```bash
python scripts/download_models.py
```

This prefetches:

- `distilbert/distilbert-base-multilingual-cased` for classification;
- `Helsinki-NLP/opus-mt-en-es` for offline Spanish augmentation;
- `en_core_web_sm` and `es_core_news_sm` for spaCy NER/tagging.

Nothing needs to be downloaded manually into the repository. Hugging Face models use the normal local cache; trained project weights are created by fine-tuning.

### Prepare the dataset

```bash
python scripts/prepare_data.py
```

The translation cache is resumable. If preparation is interrupted, rerunning the command continues from already translated source document IDs.

### Train

```bash
python scripts/train.py
```

The command:

1. trains the TF-IDF + Logistic Regression baseline;
2. fine-tunes multilingual DistilBERT for 5 epochs by default;
3. saves one `.h5` checkpoint after every epoch;
4. selects the epoch with minimum validation loss as `text_classifier_best.h5`;
5. writes `training_history.csv` and model configuration.

### Evaluate

```bash
python scripts/evaluate.py
```

This generates the real audit artifacts:

```text
reports/performance_metrics.json
reports/example_predictions.csv
```

### Validate

```bash
python scripts/validate_agent_contracts.py
python scripts/validate_project.py
pytest
```

### Run everything

```bash
python scripts/run_pipeline.py
```

Optional quantization after training/evaluation:

```bash
python scripts/optimize_model.py
```

### Dashboard

```bash
streamlit run app/real_time_dashboard.py
```

## 📝 About

The runtime pipeline accepts English or Spanish text, detects the language, classifies the document, extracts named entities, generates context-aware tags, and returns category/confidence/tagging results. Batch classification and spaCy `nlp.pipe` are used for throughput-oriented evaluation.

The project deliberately separates reproducible source code from runtime evidence. Large weights and generated datasets are ignored by Git; small training/evaluation evidence files can be committed after a real run.

## 📚 Dataset

The assignment requires all of the following at once:

- one of the recommended datasets;
- at least 10,000 documents;
- at least 5 categories;
- at least 2 languages.

The project uses **20 Newsgroups** as the canonical labeled source. Eight categories are selected:

```text
comp.graphics
comp.os.ms-windows.misc
comp.sys.ibm.pc.hardware
comp.sys.mac.hardware
comp.windows.x
misc.forsale
rec.autos
rec.sport.baseball
```

Every split is mirrored into Spanish with the offline MarianMT model `Helsinki-NLP/opus-mt-en-es`. Translation happens **inside the original train/validation/test boundary**, so an English source document and its Spanish counterpart never cross into another split.

Why not MLDoc? It is multilingual, but its standard classification setup has only four labels, which conflicts with this assignment's explicit minimum of five categories. Using translated 20 Newsgroups keeps one taxonomy across both supported languages instead of combining unrelated label spaces.

Generated CSV columns include:

```text
document_id
text
label
label_id
language
source_language
is_translation
source_dataset
```

## 🧹 EDA and preprocessing

The required notebook is:

```text
notebooks/EDA_and_Training.ipynb
```

It performs real EDA after dataset preparation:

- dataset-size and audit-minimum assertions;
- train/validation/test sizes;
- category balance;
- language balance;
- category × language coverage;
- word/character-length distributions;
- missing/empty text checks;
- exact duplicate counts;
- representative document inspection;
- baseline training;
- DistilBERT fine-tuning;
- loss/validation-loss curves;
- final metric and threshold validation.

Normalization in `utils/text_preprocessing.py` performs Unicode NFKC normalization, HTML entity decoding, whitespace normalization and stable URL/e-mail placeholders without deleting punctuation needed by NLP models.

## 🧠 Models

### Baseline

`models/baseline.py` implements:

```text
TF-IDF (word unigrams) → Logistic Regression
```

It is trained and evaluated on the same multilingual split used by the transformer.

### Transfer learning

`models/text_classifier.py` and `utils/transfer_learning.py` use:

```text
distilbert/distilbert-base-multilingual-cased
```

Default assignment settings:

| Setting | Value |
| --- | ---: |
| Epochs | `5` |
| Learning rate | `3e-5` |
| Allowed LR range | `2e-5`–`5e-5` |
| Max sequence length | `256` |
| Batch size | `16` |
| Best-model criterion | minimum validation loss |

Checkpoint layout after training:

```text
models/checkpoints/
├── epoch_01.h5
├── epoch_02.h5
├── epoch_03.h5
├── epoch_04.h5
├── epoch_05.h5
├── text_classifier_best.h5
├── config.json
├── best_epoch.json
├── training_history.csv
├── baseline.joblib
└── baseline_metrics.json
```

The large `.h5`, `.tflite` and `.joblib` artifacts are intentionally ignored by Git.

### Optimization

`utils/model_optimization.py` implements TensorFlow Lite post-training dynamic-range quantization. It is optional and should be run only after the normal model has passed accuracy validation.

## 🏷️ Context-aware tagging

`models/tagger.py` provides a separate tagging layer:

1. detect English or Spanish;
2. route to the matching spaCy pipeline;
3. extract NER entities;
4. prioritize normalized entity text as context tags;
5. add frequent non-stopword lemmas as lexical tags;
6. deduplicate tags;
7. use `nlp.pipe` for batch processing.

Language models:

| Language | spaCy model |
| --- | --- |
| English | `en_core_web_sm` |
| Spanish | `es_core_news_sm` |

## ⚡ Real-time pipeline

`utils/inference.py` combines classification and tagging behind `DocumentCategorizationPipeline`.

Single document:

```python
from utils.inference import DocumentCategorizationPipeline

pipeline = DocumentCategorizationPipeline()
result = pipeline.process("NASA announced a new orbital mission...")
print(result)
```

Batch processing:

```python
results = pipeline.process_batch(documents)
```

Batch mode is also the path used for throughput measurement.

## 📊 Evaluation and thresholds

`python scripts/evaluate.py` measures the **real** test set and writes:

```json
{
  "classification_accuracy": "<measured>",
  "f1_score_macro": "<measured>",
  "processing_speed_docs_per_sec": "<measured>",
  "languages_supported": ["en", "es"],
  "per_language_accuracy": {
    "en": "<measured>",
    "es": "<measured>"
  },
  "baseline_accuracy": "<measured>",
  "accuracy_improvement_over_baseline": "<measured>"
}
```

Mandatory gates enforced by `scripts/validate_project.py`:

| Metric | Required |
| --- | ---: |
| Classification accuracy | ≥ 0.85 |
| Macro F1 | ≥ 0.80 |
| Processing speed | ≥ 100 docs/s |
| Per-language accuracy | ≥ 0.80 |
| Transfer-learning improvement over baseline | ≥ 0.05 |

The validator fails if real results do not satisfy them; it never replaces failed results with expected values.

## 🖥️ Dashboard

`app/real_time_dashboard.py` displays:

- real-time category and confidence;
- detected language;
- generated tags;
- named entities;
- overall accuracy/F1/throughput;
- category distribution;
- tag counts;
- language distribution;
- language-specific accuracy;
- example predictions.

The dashboard reads generated evaluation reports. Before training/evaluation it shows an explicit missing-artifact message rather than fake data.

## 🧪 Tests and audit verification

Run lightweight tests:

```bash
pytest
```

Validate repository-local agent metadata:

```bash
python scripts/validate_agent_contracts.py
```

Validate subject structure and available runtime evidence:

```bash
python scripts/validate_project.py
```

Before training, missing generated data/checkpoints/reports are warnings. Once those artifacts exist, threshold violations become validation failures.

### Audit coverage

| Audit area | Implementation / evidence |
| --- | --- |
| Required project structure | repository tree + `scripts/validate_project.py` |
| README / dependencies | `README.md`, `README_RU.md`, `requirements.txt` |
| Recommended dataset | 20 Newsgroups via `sklearn.datasets.fetch_20newsgroups` |
| ≥10k / ≥5 categories / ≥2 languages | prepared split validator + notebook assertions |
| Multilingual preprocessing | `utils/text_preprocessing.py`, Spanish augmentation |
| EDA | `notebooks/EDA_and_Training.ipynb` |
| TensorFlow/Keras classifier | `models/text_classifier.py` |
| Transfer learning | multilingual DistilBERT fine-tuning |
| 5 epochs / LR range | hard validation in `ClassifierConfig` |
| Epoch checkpoints | `utils/transfer_learning.py` |
| Training history | generated `training_history.csv` |
| spaCy tagging | `models/tagger.py` |
| NER | spaCy `doc.ents` extraction |
| Language detection | `langdetect` + model routing |
| Real-time batch pipeline | `utils/inference.py` |
| Metrics JSON | `scripts/evaluate.py` |
| Example predictions | `scripts/evaluate.py` |
| Dashboard | `app/real_time_dashboard.py` |
| Baseline comparison | `models/baseline.py` + evaluation report |
| Quantization | `utils/model_optimization.py` |
| Error handling | explicit artifact/language/input validation |
| Agent navigation wrapper | `AGENTS.md`, `agent/`, contract validator |

## 📁 Project structure

```text
document-categorization/
├── agent/
│   ├── methodology.json
│   ├── module-index.json
│   ├── dependency-graph.json
│   ├── modules/
│   └── schemas/
├── app/
│   └── real_time_dashboard.py
├── data/
│   ├── raw_documents/
│   └── processed_data/
├── docs/
│   ├── architecture.md
│   └── dataset_analysis.md
├── models/
│   ├── checkpoints/
│   ├── baseline.py
│   ├── tagger.py
│   └── text_classifier.py
├── notebooks/
│   └── EDA_and_Training.ipynb
├── reports/
├── scripts/
│   ├── download_models.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── optimize_model.py
│   ├── run_pipeline.py
│   ├── validate_agent_contracts.py
│   └── validate_project.py
├── tests/
├── utils/
│   ├── data_loader.py
│   ├── inference.py
│   ├── model_optimization.py
│   ├── text_preprocessing.py
│   ├── transfer_learning.py
│   └── translation.py
├── .gitignore
├── AGENTS.md
├── Makefile
├── README.md
├── README_RU.md
├── pyproject.toml
└── requirements.txt
```

## ⚠️ Notes

- Do not commit generated raw/processed datasets or large model weights; they are reproducible and ignored by Git.
- Hugging Face pretrained models are cached outside the repository. There is no manual model file to place into this repository.
- The TensorFlow Hugging Face stack is pinned to Transformers 4.x and `tf-keras` compatibility because the project explicitly uses TensorFlow/Keras and MarianMT translation.
- NVIDIA GPU training is best run under WSL2/Linux. CPU training is possible but much slower.
- The Spanish corpus is machine-translated augmentation, not independently authored Spanish news. This is explicitly represented by `is_translation` and must be disclosed when interpreting per-language metrics.
- End-to-end throughput includes classification and spaCy tagging. If the measured speed is below the 100 docs/s requirement, the validator fails and optimization/batching must be revisited rather than changing the report.
- `scripts/optimize_model.py` uses TFLite conversion with Select TF Ops because Transformer graphs may contain operations outside the builtin TFLite set.

## 🧑‍💻 Author

- Nazar Yestayev (@nyestaye)
