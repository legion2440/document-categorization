# Document Categorization

Мультиязычный pipeline для классификации и контекстного анализа документов на английском и испанском языках. Система объединяет TensorFlow/Keras + `microsoft/mdeberta-v3-base`, baseline TF-IDF + Logistic Regression, spaCy NER и контекстные теги, EN→ES augmentation через MarianMT, calibration confidence, XLA-ускоренный inference и интерактивный Streamlit dashboard.

· [English version](README.md)

## 📋 Содержание

- [📊 Производительность](#-производительность)
- [🧩 Возможности](#-возможности)
- [🚀 Быстрый старт](#-быстрый-старт)
- [🏗️ Архитектура](#️-архитектура)
- [📚 Датасет](#-датасет)
- [🧹 Preprocessing](#-preprocessing)
- [🧠 Классификатор](#-классификатор)
- [🏷️ Контекстное тегирование](#️-контекстное-тегирование)
- [⚡ Runtime](#-runtime)
- [🧪 Оценка и воспроизводимость](#-оценка-и-воспроизводимость)
- [🖥️ Dashboard](#️-dashboard)
- [📁 Структура проекта](#-структура-проекта)
- [⚠️ Примечания](#️-примечания)
- [🧑‍💻 Автор](#-автор)

## 📊 Производительность

Текущая модель: **Revision 2 / mDeBERTa-v3-base**.

| Метрика | Revision 1 | Revision 2 |
| --- | ---: | ---: |
| Test accuracy | 81.95% | **87.79%** |
| Macro F1 | 81.92% | **87.80%** |
| End-to-end throughput | 134.08 docs/s | **130.84 docs/s** |
| English accuracy | 82.79% | **88.85%** |
| Spanish accuracy | 81.12% | **86.73%** |
| Baseline TF-IDF + Logistic Regression | 77.81% | 83.14% |
| Relative improvement над baseline | +5.32% | **+5.59%** |
| Mean calibrated confidence | 88.02% | **88.14%** |

Revision 2 — текущий production candidate. Модель показывает **87.79% test accuracy** при **130.84 документах/сек** для полного пути classification + language detection + spaCy tagging.

Обе ревизии сохранены как история разработки, а не как строгий A/B-тест: в Revision 2 изменились представление документа, набор категорий, построение validation и preprocessing policy. Контролируемое сравнение внутри каждой ревизии — transformer против baseline, обученного на тех же данных.

Фактические результаты хранятся в:

```text
reports/revision1_performance_metrics.json
reports/revision2/performance_metrics.json
reports/performance_metrics.json
reports/revision2/example_predictions.csv
```

## 🧩 Возможности

Для английского или испанского документа pipeline умеет:

- определять язык;
- классифицировать документ в одну из 12 тематических категорий;
- возвращать calibrated confidence;
- извлекать именованные сущности;
- генерировать контекстные теги;
- обрабатывать один документ или batch;
- показывать результаты в Streamlit dashboard;
- измерять accuracy, F1, качество по языкам, calibration и throughput.

Runtime разделяет GPU-нагрузку transformer classifier и CPU-нагрузку spaCy, а затем перекрывает эти стадии для увеличения throughput.

## 🚀 Быстрый старт

### Требования

- Python 3.11 или 3.12;
- Linux / WSL2 рекомендуется для NVIDIA TensorFlow training;
- NVIDIA GPU крайне желателен для fine-tuning mDeBERTa;
- интернет при первой загрузке dataset и моделей.

### Клонирование

```bash
git clone https://github.com/legion2440/document-categorization.git
cd document-categorization
```

### Окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "tensorflow[and-cuda]==2.21.0"
```

Под WSL2 TensorFlow может потребовать NVIDIA library path в каждом shell:

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

### Загрузка моделей

```bash
python scripts/download_models.py
```

Кэшируются:

- `microsoft/mdeberta-v3-base` — base model классификатора;
- `Helsinki-NLP/opus-mt-en-es` — EN→ES augmentation;
- `en_core_web_sm` и `es_core_news_sm` — spaCy tagging и NER.

### Подготовка данных

```bash
python scripts/prepare_data.py
```

Перевод возобновляемый через `data/processed_data/translation_cache_revision2/`.

### Preflight

```bash
python scripts/preflight_revision2.py
```

Preflight проверяет train/validation invariants, качество перевода, tokenizer budget, language detection и classical baseline без использования held-out test metrics.

### Обучение

```bash
python scripts/train_revision2.py
```

### Calibration и freeze

```bash
python scripts/calibrate_validation.py
python scripts/freeze_production.py
python scripts/verify_production_validation.py
```

### Просмотр сохранённых финальных метрик

```bash
python scripts/evaluate.py
```

После финальной held-out оценки эта команда работает только как evidence viewer и не запускает test повторно.

### Проверки

```bash
pytest
python scripts/validate_agent_contracts.py
python scripts/validate_project.py
```

### Dashboard

```bash
streamlit run app/real_time_dashboard.py
```

## 🏗️ Архитектура

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

Основные runtime-модули:

```text
models/text_classifier.py
models/tagger.py
utils/inference.py
utils/production_inference.py
```

## 📚 Датасет

В качестве labeled source используется **20 Newsgroups**. Каждый retained English документ зеркалируется в Spanish через MarianMT.

Текущий processed corpus:

- `11,861` source pairs;
- `23,722` строк EN/ES;
- `12` категорий;
- `2` языка;
- train: `6,022` source pairs;
- validation: `1,093` source pairs;
- test: `4,746` source pairs.

Категории:

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

Выбор категорий детерминирован и основан на cleaned source counts, а не на model scores. English source и его Spanish translation имеют общий `pair_id` и всегда остаются в одном split.

## 🧹 Preprocessing

Revision 2 сохраняет `Subject` как содержательный заголовок документа, но отбрасывает routing и sender metadata. `sklearn` удаляет footers и quotes, после чего проект самостоятельно разбирает оставшийся RFC-style header block.

Представление для classifier:

```text
Subject без повторяющихся ведущих Re:

Body
```

Далее выполняются:

- Unicode normalization;
- удаление structural noise;
- token-density cleanup;
- normalization whitespace;
- детерминированное classification window `150` слов;
- post-translation normalization для Spanish.

Повторяющиеся ведущие `Re:` удаляются, потому что train-only diagnostics показали сильную связь их наличия с категорией (`Cramér's V = 0.379`).

Validation строится детерминированно и thread-grouped: документы с одинаковым normalized subject внутри категории не разделяются между train и validation.

Изначальный temporal-validation design оказался технически недоступен из-за отсутствия пригодного `Date:` coverage в official corpus. Решение о fallback и его обоснование сохранены в:

```text
config/revision2_protocol.json
config/revision2_amendment_01.json
docs/revision2_preregistration.md
docs/revision2_amendment_01.md
```

## 🧠 Классификатор

### Baseline

```text
TF-IDF word unigrams → Logistic Regression
```

Реализация: `models/baseline.py`.

### Transfer learning

Production model:

```text
microsoft/mdeberta-v3-base
```

Зафиксированная training configuration:

| Параметр | Значение |
| --- | ---: |
| Epochs | `5` |
| Learning rate | `2e-5` |
| Batch size | `2` |
| Max model tokens | `512` |
| Source text window | `150` слов |
| Optimizer | AdamW |
| Weight decay | `0.01` |
| Warmup | `10%` |
| Gradient clip | `1.0` |
| Random seed | `42` |

Checkpoint выбирается детерминированно: максимум правильных validation documents, затем меньший validation loss, затем более ранняя epoch. Revision 2 выбрал epoch 5 с `1886/2186` правильных validation documents (`86.28%`).

Training artifacts включают checkpoints всех эпох, runtime config, history, baseline metrics, optimizer plan и token-budget diagnostics. Большие веса намеренно исключены из Git.

### Calibration confidence

Scalar temperature scaling обучается только на validation. Revision 2 использует `T = 2.9108`.

Calibration не изменила class argmax и улучшила validation calibration:

| Метрика | До | После |
| --- | ---: | ---: |
| Mean confidence | 98.97% | 87.83% |
| NLL | 1.3163 | 0.5928 |
| ECE | 0.1292 | 0.0454 |
| Brier score | 0.2648 | 0.2304 |

## 🏷️ Контекстное тегирование

`models/tagger.py` реализует language-aware spaCy tagging:

1. определяет English или Spanish;
2. выбирает соответствующую spaCy model;
3. извлекает named entities;
4. приоритизирует entities как context tags;
5. добавляет частотные meaningful lemmas;
6. удаляет дубли;
7. обрабатывает batch через `nlp.pipe`.

| Язык | spaCy model |
| --- | --- |
| English | `en_core_web_sm` |
| Spanish | `es_core_news_sm` |

Production tagger использует окно `75` слов, classifier — `150` слов.

## ⚡ Runtime

Frozen inference path использует:

- float32 TensorFlow inference;
- XLA compilation;
- fixed token buckets `64 / 128 / 192 / 256 / 384 / 512`;
- attention-balanced batch sizes `32 / 32 / 16 / 16 / 4 / 4`;
- parallel CPU spaCy tagging и GPU transformer classification;
- validation-fitted temperature scaling.

Измеренный Revision 2 end-to-end throughput: **130.84 docs/s**.

Перед финальной held-out оценкой frozen validation verification точно воспроизвёл выбранный checkpoint (`1886/2186` правильных) со скоростью **112.36 docs/s**.

## 🧪 Оценка и воспроизводимость

В репозитории сохранены две model-development ревизии.

**Revision 1** показал generalization gap на первой held-out evaluation: `81.95%` test accuracy при более сильных validation metrics. Вместо повторного подбора по test результат был сохранён, а новый data/model protocol зафиксирован до retraining.

**Revision 2** добавил текущее title-aware представление, deterministic grouped validation, новый translation cache, новую validation-only calibration и новый mDeBERTa training run. Его held-out результат — `87.79%` accuracy / `87.80%` macro F1.

Исходный Revision 1 опубликован рядом с Revision 2. Для Revision 2 также хранится immutable evaluation marker, чтобы held-out результат нельзя было незаметно пересчитать после просмотра результата.

Статистическое сравнение с classical baseline на том же split:

- McNemar EN: `434` transformer-only против `213` baseline-only, `p = 2.44e-18`;
- McNemar ES: `480` против `260`, `p = 5.05e-16`;
- pair-cluster bootstrap absolute improvement 95% CI: `+3.76`…`+5.62` процентного пункта;
- pair-cluster bootstrap relative improvement 95% CI: `+4.50%`…`+6.79%`.

Reproducibility evidence:

```text
reports/revision2_preflight.json
models/checkpoints/production_validation_verification.json
reports/revision1_performance_metrics.json
reports/revision2/performance_metrics.json
reports/revision2/final_test_consumed.json
```

## 🖥️ Dashboard

`app/real_time_dashboard.py` показывает:

- predicted category;
- calibrated confidence;
- detected language;
- context tags;
- named entities;
- final accuracy / F1 / throughput;
- per-language metrics;
- category и tag distributions;
- example predictions.

Запуск:

```bash
streamlit run app/real_time_dashboard.py
```

## 📁 Структура проекта

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

## ⚠️ Примечания

- Spanish документы — machine-translated augmentation, а не независимо написанный Spanish corpus.
- Raw/processed datasets и большие model weights воспроизводимы и намеренно исключены из Git.
- Revision 1 и Revision 2 используют разные data policies, поэтому изменение метрик между ними следует читать как историю разработки, а не как строгий same-sample experiment.
- `scripts/evaluate.py` после сохранённой финальной оценки работает только как evidence viewer.

## 🧑‍💻 Автор

- Nazar Yestayev (@nyestaye)
