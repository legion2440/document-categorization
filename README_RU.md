# Document Categorization

Интеллектуальная мультиязычная система категоризации документов и контекстного тегирования для задания 01-edu. Проект объединяет классификатор multilingual DistilBERT на TensorFlow/Keras, baseline TF-IDF + Logistic Regression, spaCy NER, возобновляемое EN→ES расширение датасета, полноценную оценку производительности и Streamlit-дашборд.

Репозиторий содержит весь код обучения и оценки, но **не хранит выдуманные метрики и большие веса модели**. Реальные checkpoints, history и reports создаются после фактического запуска pipeline.

· [English version](README.md)

## 📋 Содержание

- [🚀 Быстрый старт](#-быстрый-старт)
- [📝 О проекте](#-о-проекте)
- [📚 Датасет](#-датасет)
- [🧹 EDA и preprocessing](#-eda-и-preprocessing)
- [🧠 Модели](#-модели)
- [🏷️ Контекстное тегирование](#️-контекстное-тегирование)
- [⚡ Real-time pipeline](#-real-time-pipeline)
- [📊 Оценка и пороги](#-оценка-и-пороги)
- [🖥️ Dashboard](#️-dashboard)
- [🧪 Тесты и аудит](#-тесты-и-аудит)
- [📁 Структура проекта](#-структура-проекта)
- [⚠️ Примечания](#️-примечания)
- [🧑‍💻 Автор](#-автор)

## 🚀 Быстрый старт

### Требования

- Python 3.11 или 3.12;
- для NVIDIA GPU рекомендуется Linux / WSL2;
- свободное место под TensorFlow, Hugging Face cache, переведённый датасет и локальные checkpoints;
- интернет при первой загрузке датасета и моделей.

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
```

Для NVIDIA GPU под WSL2:

```bash
pip install "tensorflow[and-cuda]==2.21.0"
```

### Загрузка моделей

```bash
python scripts/download_models.py
```

Скрипт автоматически кладёт в обычные локальные кэши:

- `distilbert/distilbert-base-multilingual-cased`;
- `Helsinki-NLP/opus-mt-en-es`;
- `en_core_web_sm`;
- `es_core_news_sm`.

Вручную скачивать и класть модель в репозиторий не нужно. Финальные веса классификатора появятся после нашего fine-tuning.

### Подготовка данных

```bash
python scripts/prepare_data.py
```

Перевод EN→ES кэшируется по `document_id`, поэтому оборванную подготовку можно продолжить повторным запуском.

### Обучение

```bash
python scripts/train.py
```

Скрипт последовательно:

1. обучает baseline TF-IDF + Logistic Regression;
2. fine-tune'ит multilingual DistilBERT 5 эпох;
3. сохраняет checkpoint каждой эпохи;
4. выбирает минимальный `val_loss` как `text_classifier_best.h5`;
5. создаёт `training_history.csv` и config.

### Оценка

```bash
python scripts/evaluate.py
```

Создаются реальные:

```text
reports/performance_metrics.json
reports/example_predictions.csv
```

### Проверка

```bash
python scripts/validate_agent_contracts.py
python scripts/validate_project.py
pytest
```

### Весь pipeline одной командой

```bash
python scripts/run_pipeline.py
```

Опциональная quantization после обучения:

```bash
python scripts/optimize_model.py
```

### Dashboard

```bash
streamlit run app/real_time_dashboard.py
```

## 📝 О проекте

Pipeline принимает английский или испанский документ, определяет язык, классифицирует текст, извлекает именованные сущности, строит контекстные теги и возвращает категорию, confidence и tagging results.

Код и runtime evidence разделены. Большие веса и сгенерированный датасет не коммитятся; маленькие `training_history.csv`, configs и reports после реального запуска можно использовать как доказательные артефакты аудита.

## 📚 Датасет

Задание одновременно требует:

- один из рекомендованных датасетов;
- минимум 10 000 документов;
- минимум 5 категорий;
- минимум 2 языка.

В качестве канонического датасета используется **20 Newsgroups** с восемью категориями:

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

Каждый split зеркалируется на испанский язык офлайн-моделью `Helsinki-NLP/opus-mt-en-es`. Перевод выполняется **внутри исходной границы train/validation/test**, поэтому оригинал и его перевод не попадают в разные splits.

MLDoc намеренно не выбран как единственный датасет: он мультиязычный, но стандартная классификация содержит только 4 labels, а задание требует минимум 5 категорий.

Колонки generated CSV:

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

## 🧹 EDA и preprocessing

Обязательный notebook:

```text
notebooks/EDA_and_Training.ipynb
```

После подготовки данных он считает и строит:

- размер датасета и проверки минимальных требований;
- размеры splits;
- баланс категорий;
- баланс языков;
- category × language coverage;
- распределения длины документов;
- missing/empty checks;
- exact duplicates;
- реальные примеры документов;
- baseline metrics;
- обучение DistilBERT;
- `loss/val_loss` и accuracy curves;
- итоговые метрики и asserts порогов аудита.

`utils/text_preprocessing.py` делает NFKC-normalization, HTML decoding, нормализацию пробелов и замену URL/e-mail стабильными placeholders без агрессивного удаления NLP-контекста.

## 🧠 Модели

### Baseline

```text
TF-IDF → Logistic Regression
```

Реализация: `models/baseline.py`.

### Transfer learning

Используется:

```text
distilbert/distilbert-base-multilingual-cased
```

Настройки по умолчанию:

| Параметр | Значение |
| --- | ---: |
| Epochs | `5` |
| Learning rate | `3e-5` |
| Допустимый LR | `2e-5`–`5e-5` |
| Max length | `256` |
| Batch size | `16` |
| Best checkpoint | минимальный `val_loss` |

После обучения:

```text
models/checkpoints/
├── epoch_01.h5 ... epoch_05.h5
├── text_classifier_best.h5
├── config.json
├── best_epoch.json
├── training_history.csv
├── baseline.joblib
└── baseline_metrics.json
```

Большие `.h5`, `.tflite` и `.joblib` находятся в `.gitignore`.

### Оптимизация

`utils/model_optimization.py` реализует post-training dynamic-range quantization через TensorFlow Lite. Запускать её следует после того, как обычная модель уже прошла accuracy-проверки.

## 🏷️ Контекстное тегирование

`models/tagger.py`:

1. определяет язык;
2. выбирает English/Spanish spaCy pipeline;
3. извлекает NER через `doc.ents`;
4. использует entities как приоритетные context tags;
5. добавляет частотные meaningful lemmas;
6. удаляет дубли;
7. использует `nlp.pipe` в batch режиме.

| Язык | spaCy model |
| --- | --- |
| English | `en_core_web_sm` |
| Spanish | `es_core_news_sm` |

## ⚡ Real-time pipeline

Основной API: `utils/inference.py` → `DocumentCategorizationPipeline`.

```python
from utils.inference import DocumentCategorizationPipeline

pipeline = DocumentCategorizationPipeline()
result = pipeline.process("NASA announced a new orbital mission...")
print(result)
```

Для throughput используется `process_batch()`.

## 📊 Оценка и пороги

`python scripts/evaluate.py` измеряет test set и создаёт `performance_metrics.json`.

Валидатор требует:

| Метрика | Порог |
| --- | ---: |
| Classification accuracy | ≥ 0.85 |
| Macro F1 | ≥ 0.80 |
| Processing speed | ≥ 100 docs/s |
| Accuracy каждого языка | ≥ 0.80 |
| Улучшение над baseline | ≥ 0.05 |

Если реальные показатели ниже — `scripts/validate_project.py` падает. Метрики не подменяются ожидаемыми значениями.

## 🖥️ Dashboard

`app/real_time_dashboard.py` показывает:

- категорию и confidence в реальном времени;
- определённый язык;
- tags и named entities;
- accuracy/F1/throughput;
- category distribution;
- tag counts;
- language breakdown;
- per-language accuracy;
- example predictions.

До реального обучения dashboard явно сообщает об отсутствующих artifacts вместо mock-данных.

## 🧪 Тесты и аудит

```bash
pytest
python scripts/validate_agent_contracts.py
python scripts/validate_project.py
```

До обучения отсутствие generated artifacts считается предупреждением. После появления checkpoints/reports их структура и пороги проверяются как реальные audit gates.

Основное соответствие аудиту:

| Пункт | Реализация |
| --- | --- |
| Project structure | дерево + `validate_project.py` |
| Recommended dataset | 20 Newsgroups |
| 10k / 5 categories / 2 languages | dataset validator + notebook |
| EDA | `EDA_and_Training.ipynb` |
| TensorFlow/Keras | `models/text_classifier.py` |
| Transfer learning | multilingual DistilBERT |
| 5 epochs / LR | `ClassifierConfig` hard validation |
| Checkpoints / history | `utils/transfer_learning.py` |
| spaCy + NER | `models/tagger.py` |
| Real-time pipeline | `utils/inference.py` |
| Metrics / predictions | `scripts/evaluate.py` |
| Streamlit | `app/real_time_dashboard.py` |
| Baseline ≥5% comparison | baseline + final validator |
| Quantization | `utils/model_optimization.py` |
| Agent wrapper | `AGENTS.md`, `agent/`, validator |

## 📁 Структура проекта

```text
document-categorization/
├── agent/
├── app/
├── data/
├── docs/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── tests/
├── utils/
├── .gitignore
├── AGENTS.md
├── Makefile
├── README.md
├── README_RU.md
├── pyproject.toml
└── requirements.txt
```

## ⚠️ Примечания

- Generated dataset и большие веса намеренно не попадают в Git.
- Pretrained Hugging Face models живут в обычном локальном cache; вручную файл модели в репу класть не надо.
- Transformers закреплён на ветке 4.x, а TensorFlow использует `tf-keras` compatibility layer.
- Для RTX/NVIDIA предпочтителен WSL2/Linux; CPU fine-tuning будет значительно медленнее.
- Испанская часть — machine-translated augmentation, а не независимый испанский корпус. Это явно хранится в `is_translation` и учитывается при интерпретации per-language metrics.
- Throughput считается для end-to-end classification + spaCy tagging. Если результат ниже 100 docs/s, валидатор должен упасть — значение нельзя просто переписать в JSON.

## 🧑‍💻 Автор

- Nazar Yestayev (@nyestaye)
