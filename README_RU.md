# Document Categorization

Мультиязычная категоризация документов и контекстное тегирование для задания 01-edu. Production-путь использует TensorFlow/Keras + `microsoft/mdeberta-v3-base`, baseline TF-IDF + Logistic Regression, spaCy NER/tagging, EN→ES augmentation через MarianMT, calibration confidence, XLA inference и Streamlit dashboard.

· [English version](README.md)

## 📋 Содержание

- [📊 Финальные результаты](#-финальные-результаты)
- [🚀 Быстрый старт](#-быстрый-старт)
- [📚 Датасет](#-датасет)
- [🧹 Preprocessing](#-preprocessing)
- [🧠 Классификатор](#-классификатор)
- [🏷️ Контекстное тегирование](#️-контекстное-тегирование)
- [⚡ Runtime](#-runtime)
- [🧪 Протокол оценки](#-протокол-оценки)
- [🖥️ Dashboard](#️-dashboard)
- [✅ Проверка аудита](#-проверка-аудита)
- [📁 Структура проекта](#-структура-проекта)
- [⚠️ Примечания](#️-примечания)
- [🧑‍💻 Автор](#-автор)

## 📊 Финальные результаты

Первый held-out test не прошёл обязательный порог задания `85%` accuracy. Этот результат сохранён как Revision 1. После него был заранее зафиксирован Revision 2 — post-first-test protocol revision с новым представлением документа и новым validation protocol. Затем был выполнен один второй и последний held-out test. Оба результата опубликованы.

| Метрика | Revision 1 | Revision 2 |
| --- | ---: | ---: |
| Test accuracy | 81.95% | **87.79%** |
| Macro F1 | 81.92% | **87.80%** |
| Throughput | 134.08 docs/s | **130.84 docs/s** |
| English accuracy | 82.79% | **88.85%** |
| Spanish accuracy | 81.12% | **86.73%** |
| Baseline accuracy | 77.81% | 83.14% |
| Relative improvement над baseline | +5.32% | **+5.59%** |
| Absolute improvement | +4.14 п.п. | +4.65 п.п. |
| Пороги задания | FAIL | **PASS** |

Revision 2 проходит все обязательные gates задания: accuracy `>=85%`, macro F1 `>=80%`, speed `>=100 docs/s`, accuracy каждого языка `>=80%` и relative improvement над baseline `>=5%`.

Более строгая, но не являющаяся gate интерпретация `+5 percentage points` тоже публикуется отдельно и **не выполнена**: Revision 2 даёт `+4.65 п.п.`. В задании используется буквальное относительное улучшение `>=5%`, при этом обе формы результата раскрыты.

Evidence:

```text
reports/revision1_performance_metrics.json
reports/revision2/performance_metrics.json
reports/revision2/final_test_consumed.json
reports/performance_metrics.json
```

Второй test-run является последним. `reports/revision2/final_test_consumed.json` фиксирует запрет третьего final-test evaluation.

## 🚀 Быстрый старт

### Требования

- Python 3.11 или 3.12;
- Linux / WSL2 рекомендуется для NVIDIA TensorFlow training;
- NVIDIA GPU крайне желателен для fine-tuning mDeBERTa;
- интернет при первой загрузке dataset/models.

### Клонирование

```bash
git clone https://01.tomorrow-school.ai/git/nyestaye/document-categorization
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

### Загрузка зависимостей моделей

```bash
python scripts/download_models.py
```

Кэшируются:

- `microsoft/mdeberta-v3-base` — production classifier base;
- `Helsinki-NLP/opus-mt-en-es` — EN→ES augmentation;
- `en_core_web_sm` и `es_core_news_sm` — spaCy tagging/NER.

### Подготовка Revision 2 dataset

```bash
python scripts/prepare_data.py
```

Перевод возобновляемый: Revision 2 использует отдельный cache в `data/processed_data/translation_cache_revision2/`.

### Preflight

```bash
python scripts/preflight_revision2.py
```

Preflight физически читает только **train + validation** и проверяет dataset invariants, sanity перевода, token budget, language detection и baseline. `test.csv` не читается.

### Обучение

```bash
python scripts/train_revision2.py
```

Зафиксированный production run:

| Параметр | Значение |
| --- | ---: |
| Model | `microsoft/mdeberta-v3-base` |
| Epochs | `5` |
| Learning rate | `2e-5` |
| Batch size | `2` |
| Max model tokens | `512` |
| Classification source window | `150` слов |
| Optimizer | AdamW |
| Weight decay | `0.01` |
| Warmup | `10%` |
| Gradient clip | `1.0` |

Checkpoint выбирается детерминированно: максимум правильных validation документов, затем меньший validation loss, затем более ранняя epoch. Revision 2 выбрал epoch 5: `1886/2186`, или `86.28%`.

### Calibration, freeze, validation verification

```bash
python scripts/calibrate_validation.py
python scripts/freeze_production.py
python scripts/verify_production_validation.py
```

Temperature scaling обучается только на validation. Для Revision 2 получено `T=2.9108`; argmax не меняется, а validation ECE снижается с `0.1292` до `0.0454`.

Frozen validation verification без чтения test воспроизвёл выбранные `1886/2186` правильных документов точно.

### Показать финальный evidence

```bash
python scripts/evaluate.py
```

Теперь эта команда **только выводит уже сохранённые final metrics**. Held-out test повторно не читается, потому что единственный разрешённый Revision 2 final evaluation уже израсходован.

### Проверки

```bash
pytest
python scripts/validate_agent_contracts.py
python scripts/validate_project.py
```

## 📚 Датасет

Используется **20 Newsgroups**. Каждый retained English документ зеркалируется в Spanish через MarianMT.

После Revision 2 cleaning и cross-split deduplication:

- `11,861` независимая source pair;
- `23,722` строк EN/ES;
- `12` категорий;
- `2` языка;
- train: `6,022` source pairs;
- validation: `1,093` source pairs;
- test: `4,746` source pairs.

Категории Revision 2 выбираются механически по cleaned source counts, без model performance:

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

Правило сортирует категории по cleaned official-train count с lexicographic tie-break и берёт минимальный prefix, пока cleaned train+test source count не достигнет `11,000`.

English source и его Spanish translation имеют общий `pair_id` и всегда остаются в одном split.

## 🧹 Preprocessing

Revision 2 сохраняет `Subject` как содержательный заголовок документа, но отбрасывает routing/sender metadata. `sklearn` удаляет footers и quotes, после чего проект самостоятельно разбирает оставшийся RFC-style header block.

Представление документа:

```text
Subject без повторяющихся ведущих Re:

Body
```

Остальные headers удаляются. `Re:` удаляется, потому что train-only diagnostic показал сильную связь с категорией: `Cramér's V = 0.379` при заранее установленном пороге `0.10`.

Далее выполняются structural cleanup, token-density cleanup, Unicode normalization и детерминированное окно `150` слов. Spanish переводится уже из очищенного English representation и проходит нормализацию после перевода.

Revision 2 validation использует deterministic thread-grouped stratified fallback. Изначально запланированный temporal split оказался технически невозможен: parseable `Date:` coverage в official train равен `0/11,314`. Этот stop-condition сработал до retraining и был зафиксирован отдельным Amendment 01. Целые normalized-Subject thread groups никогда не разделяются между train и validation.

Протокол:

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

Основные файлы:

```text
models/text_classifier.py
utils/transfer_learning.py
scripts/train_revision2.py
```

Training сохраняет checkpoint каждой эпохи и runtime artifacts:

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

Большие веса намеренно исключены из Git.

### Calibration

Confidence калибруется scalar temperature scaling только по validation. Calibration меняет confidence, но не class argmax.

## 🏷️ Контекстное тегирование

`models/tagger.py`:

1. определяет English/Spanish;
2. выбирает соответствующий spaCy pipeline;
3. извлекает named entities;
4. использует entities как приоритетные context tags;
5. добавляет meaningful lemmas;
6. удаляет дубли;
7. обрабатывает batch через `nlp.pipe`.

| Язык | spaCy model |
| --- | --- |
| English | `en_core_web_sm` |
| Spanish | `es_core_news_sm` |

Production tagger использует окно `75` слов, classifier — `150` слов.

## ⚡ Runtime

Frozen production runtime:

- float32;
- XLA compilation;
- fixed sequence buckets `64/128/192/256/384/512`;
- attention-balanced batch sizes `32/32/16/16/4/4`;
- parallel CPU spaCy tagging + GPU classification;
- validation-fitted temperature scaling.

Revision 2 final end-to-end throughput: **130.84 docs/s**, включая classification и tagging.

Основные API:

```text
utils/inference.py
utils/production_inference.py
```

## 🧪 Протокол оценки

Revision 1 остаётся исходным первым held-out result и не прошёл только обязательный gate `85%` accuracy.

Revision 2 был явно зарегистрирован как **post-first-test protocol revision** до нового preprocessing/training. Второй final evaluation защищён irreversible marker, созданным до чтения `test.csv`. Marker теперь завершён, поэтому третий test-run запрещён независимо от результата.

Статистическая оценка Revision 2 также показывает преимущество transformer над baseline:

- McNemar EN: `434` transformer-only против `213` baseline-only, `p = 2.44e-18`;
- McNemar ES: `480` против `260`, `p = 5.05e-16`;
- pair-cluster bootstrap absolute improvement 95% CI: `+3.76`…`+5.62 п.п.`;
- pair-cluster bootstrap relative improvement 95% CI: `+4.50%`…`+6.79%`.

Point estimate проходит assignment gate `+5% relative`. Bootstrap interval публикуется как uncertainty и не подменяет правило задания по point estimate.

## 🖥️ Dashboard

```bash
streamlit run app/real_time_dashboard.py
```

Dashboard показывает category, calibrated confidence, detected language, tags, named entities, final accuracy/F1/throughput, language breakdown и example predictions.

## ✅ Проверка аудита

| Пункт | Реализация / результат |
| --- | --- |
| Recommended dataset | 20 Newsgroups |
| ≥10k documents | 23,722 EN/ES rows / 11,861 source pairs |
| ≥5 categories | 12 |
| ≥2 languages | English + Spanish |
| TensorFlow/Keras | `models/text_classifier.py` |
| Transfer learning | mDeBERTa-v3-base |
| ≥5 epochs | 5 |
| LR 2e-5…5e-5 | 2e-5 |
| Validation loss | `training_history.csv` |
| Checkpoints | `utils/transfer_learning.py` |
| Baseline | TF-IDF + Logistic Regression |
| spaCy + NER | `models/tagger.py` |
| Accuracy ≥85% | 87.79% |
| Macro F1 ≥80% | 87.80% |
| Speed ≥100 docs/s | 130.84 docs/s |
| Per-language accuracy ≥80% | EN 88.85%, ES 86.73% |
| Relative improvement ≥5% | +5.59% |
| Quantization | `utils/model_optimization.py` |
| Final evidence | `reports/` |

## 📁 Структура проекта

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

## ⚠️ Примечания

- Spanish data — machine-translated augmentation, а не независимо написанный Spanish corpus.
- Raw/processed dataset и большие model weights воспроизводимы и не хранятся в Git.
- Revision 1 и Revision 2 должны оставаться видимыми одновременно; Revision 2 нельзя представлять как untouched first test.
- Final held-out test уже consumed. Нельзя удалять marker ради повторного запуска.
- `scripts/evaluate.py` после финального evaluation намеренно работает только как evidence viewer.

## 🧑‍💻 Автор

- Nazar Yestayev (@nyestaye)
