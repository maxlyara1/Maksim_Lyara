# 🎯 AMML ranking competition

> Решение задачи ранжирования товаров по поисковым запросам с использованием градиентного бустинга и cross-encoder моделей

---

## 📋 О проекте

**Вход:** Поисковый запрос + карточка товара  
**Таргет:** Релевантность `relevance` ∈ {0, 1, 2, 3}  
**Метрика:** nDCG@10 (по группам `query_id`)

### 📊 Данные
- `train.csv` — ~49k строк с таргетом
- `test.csv` — ~21k строк без таргета
- `submission.csv` — формат сабмита (id/prediction)

---

## 🏗️ Архитектура решения

```
┌─────────────────┐
│   Raw Data      │
│  (train/test)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ ◄─── HTML очистка, нормализация, обрезка
│  (clean_text)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Features     │ ◄─── BM25, E5 embeddings, CE scores, TE
│  Engineering    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   CatBoost      │ ◄─── GroupKFold, YetiRank loss
│   Ensemble      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   CE Blend      │ ◄─── Смешивание CatBoost + CE scores из фичей
│  (ce_blend_w)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CE Rerank      │ ◄─── Fine-tuned MiniLM (опционально)
│  (top-K)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Submission     │
└─────────────────┘
```

---

## 🔧 Компоненты пайплайна

### 1️⃣ **Препроцессинг** (`src/preprocessing.py`)
- 🧹 Очистка HTML-тегов и entities (`clean_text`)
- ✂️ Удаление пунктуации/цифр в начале запросов
- 🔄 Нормализация плейсхолдеров (`none`/`[null]`/`brand_masked` → `""`)
- 📏 Релевантное окно ±160 символов для длинных полей
- 🎨 Нормализация брендов и цветов
- 🗑️ Удаление константных признаков (`product_locale`)

### 2️⃣ **Feature Engineering** (`src/features.py`)
- **Текстовые метрики:**
  - BM25 по title/description/bullet_point/all
  - Jaccard similarity, Levenshtein distance
  - Coverage, числовые совпадения
  
- **Семантические фичи:**
  - E5 embeddings (`intfloat/multilingual-e5-large-instruct`)
  - Cosine similarity для title/desc/bullet/all
  - Комбинированные фичи: title+brand+color
  
- **Cross-encoder:**
  - BGE-reranker-v2-m3 скоры (кешируются)
  
- **Target Encoding:**
  - TE для `product_id`, `product_title`, `query`
  
- **Категориальные:**
  - Подготовка для CatBoost (brand, color)

### 3️⃣ **Обучение CatBoost** (`src/train.py`)
- 📊 GroupKFold по `query_id` (seed: 993)
- 🎯 Loss: YetiRank (оптимизирован для ранжирования)
- 💾 Сохранение: фолды `catboost_fold*.cbm` + `metadata.json`

### 4️⃣ **Инференс** (`src/inference.py`)
- 🔄 Препроцессинг + фичи
- 🤖 CatBoost ансамбль
- 🔀 Опциональный CE blend (`ce_blend_weight`)
- 🔝 CE rerank top-K (fine-tuned MiniLM)

### 5️⃣ **Cross-Encoder Training** (`src/ce_train.py`)
- 🎓 Fine-tuning MiniLM (2 эпохи, max_len=160)
- 📦 Сохранение в `models_ce_minilm/`

---

## 🚀 Быстрый старт

### Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Запуск полного цикла
```bash
python main.py --mode full \
  --train-path data/train.csv \
  --test-path data/test.csv \
  --models-dir models \
  --submission-path results/submission.csv
```

> 💡 **Совет:** Фичи и CE-скоры кэшируются в `cache/` для ускорения повторных запусков

---

## 🎮 Режимы работы `main.py`

| Режим | Описание |
|-------|----------|
| `train` | Обучение CatBoost на train.csv |
| `infer` | Инференс на test.csv → submission.csv |
| `ce-train` | Fine-tuning Cross-Encoder MiniLM |
| `full` | Полный цикл: train → ce-train (если нужно) → infer |
| `rerank` | Применение CE rerank к готовому submission.csv |

### Примеры команд

**Обучение CatBoost:**
```bash
python main.py --mode train \
  --train-path data/train.csv \
  --models-dir models \
  --iterations 2000 \
  --learning-rate 0.03 \
  --depth 8
```

**Инференс с CE rerank:**
```bash
python main.py --mode infer \
  --test-path data/test.csv \
  --models-dir models \
  --submission-path results/submission.csv \
  --ce-model-dir models_ce_minilm \
  --ce-rerank-top-k 200 \
  --ce-rerank-weight 0.4
```

---

## 🐳 Docker

Сборка и запуск инференса:

```bash
# Сборка образа
docker build -t maksim-lyara-solution .

# Запуск (data/ должен быть примонтирован)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  maksim-lyara-solution
```

Итоговый файл: `results/submission.csv`

---

## 📁 Структура проекта

```
.
├── src/                     # Основной код
│   ├── preprocessing.py     # Очистка и нормализация данных
│   ├── features.py          # Генерация признаков
│   ├── train.py             # Обучение CatBoost
│   ├── inference.py         # Инференс
│   ├── ce_train.py          # Обучение Cross-Encoder
│   └── rerank.py            # CE rerank
│
├── data/                     # Данные
│   ├── train.csv
│   └── test.csv
│
├── models/                   # CatBoost модели
│   ├── catboost_fold*.cbm
│   └── metadata.json
│
├── models_ce_minilm/         # Fine-tuned CE
│
├── cache/                    # Кэш эмбеддингов и CE скоров
│
├── notebooks/                # EDA и эксперименты
│   └── eda.ipynb
│
├── results/                  # Результаты
│   └── submission.csv
│
├── main.py                   # Точка входа
└── requirements.txt          # Зависимости
```

---

## 💡 Ключевые особенности

- ✅ **Фиксированный seed (993)** для воспроизводимости
- ✅ **Нормализация train/test** для устранения data leakage
- ✅ **Умная обрезка текста** — релевантное окно ускоряет обработку
- ✅ **Кэширование** — эмбеддинги и CE скоры переиспользуются
- ✅ **Гибкий блендинг** — настраиваемые веса для CatBoost + CE
- ✅ **Production-ready** — Docker-контейнер для деплоя

---

## 📚 Дополнительная информация

Подробный EDA и обоснование выбора препроцессинга: `notebooks/eda.ipynb`

---

## 👤 Автор

**Maksim Lyara** — решение для AMML ranking competition
