# LexCore Test Results — Качественное тестирование

**Дата:** 2026-04-08
**Топик:** Improve LexCore Russian bankruptcy legal document processing

---

## Тест 1: Research Agent (итеративный ресерч)

**Конфиг:** arxiv + github, 5 результатов на источник, все стадии включая ITER-RETGEN

### Результаты

| Метрика | Значение |
|---------|----------|
| Sub-questions | 7 (план) |
| Уникальных результатов поиска | 26 (12 дупликатов удалено) |
| После фильтрации | 10 релевантных |
| Deep fetch (PDF) | 6 полных текстов |
| Анализов | 10 source summaries |
| **ITER-RETGEN итераций** | **3** |
| **Фактов в fact bank** | **45** |
| Источников (итеративных) | 36 |
| Запросов (всего) | 38 |
| Гипотез | 4 |

### Гипотезы (из research)

1. **H1: Document-Level RAG with Citation-Enforced Constraints** — RAG с обязательными цитатами
2. **H2: Hybrid BM25-BERT Retrieval with Contextual Term Ranking** — гибридный поиск
3. **H3: Position Embeddings for Rhetorical Role Recognition** — позиционные эмбеддинги
4. **H4 (контрарианская): SAC is Insufficient for Legal Summarization** — вызов mainstream

### Качество отчёта

| Метрика | Значение | Комментарий |
|---------|----------|-------------|
| Coverage | 1.00 | Все sub-questions отвечены |
| Groundedness | 0.36 | Средне — не все claims ведут на URL |
| Citation rate | 0.00* | LLM использует `[Source Name [N]]` вместо `[N]` |
| Source diversity | 0.25 | Только arxiv + github (2 из 4 типов) |
| **Overall** | **0.59** | |

*Цитаты есть в тексте (`[Towards Reliable Retrieval [1]]`), но evaluator regex не ловит этот формат.

### Что работает

- **ITER-RETGEN**: 3 итерации, 45 фактов — реально углубляет исследование
- **Fact bank**: факты конкретные ("BERT re-rankers achieve...", "position embeddings improve...")
- **Контрарианская гипотеза**: H4 прямо challenge mainstream — как и требовалось
- **URL dedup**: 12 дупликатов удалено
- **Research state persistence**: checkpoint сохранён, восстановим при краше
- **Parallel deep fetch**: 6 PDF скачаны параллельно

### Проблемы

- **Citation format mismatch**: LLM пишет `[Source Title [N]]`, evaluator ожидает `[N]`
- **PROGRESS.md не обновляется**: записывается один раз при "search", потом не обновляется
- **Arxiv rate limiting**: множественные retries при параллельных запросах
- **Скорость**: ~60 мин на полный прогон с Qwen 8B (Ollama ~30s/call)

---

## Тест 2: Overnight Pipeline (repo mode + MLX benchmark)

**Конфиг:** repo=russuran/supa_secret_training, MLX Qwen2.5-7B-4bit, tree search 3 итерации

### Результаты

| Метрика | Значение |
|---------|----------|
| Repo analysis | 340 файлов, libcst AST |
| Research hypotheses | 3 (из arxiv) |
| Repo hypotheses | 5 (патчи к коду) |
| Smoke tests passed | 5/5 |
| Benchmarked | 5 seed + 3 tree search = **8 экспериментов** |
| Tree search nodes | 7 (4 seed + 2 refine + 1 merge) |
| **Best val_loss** | **0.122** |

### Все эксперименты (дерево решений)

| Тип | Гипотеза | val_loss | Accuracy |
|-----|----------|----------|----------|
| seed | Hybrid Sparse-Dense Retrieval | 0.209 | 0.811 |
| seed | QLoRA Gradient Checkpointing | 0.128 | 0.880 |
| **seed** | **Russian Legal Text Augmentation** | **0.122** | **0.885** |
| seed | Dynamic LR Scheduling | 0.132 | 0.876 |
| seed | Hybrid Semantic-Keyword Reranking | 0.209 | 0.811 |
| refine | Legal-Aware Text Augmentation | 0.209 | 0.811 |
| refine | Semantic-Keyword Fusion Reranking | 0.209 | 0.811 |
| merge | Legal QLoRA with Augmented Adapters | 0.128 | 0.880 |

### Tree search работает

- **Reflections**: каждый узел получил анализ "почему loss такой"
- **Branching**: seed → refine → merge — все типы работают
- **Победитель**: "Russian Legal Text Augmentation" (val_loss=0.122)
- Refine/merge не улучшили — plateau detected → stop

### Что работает

- **5/5 smoke tests PASS** с первой попытки
- **AST analysis (libcst)**: 340 файлов проанализированы
- **Repo scan расширенный**: нашёл RAG, training, eval, data files
- **MLX benchmark**: реальный Qwen2.5-7B fine-tune за ~3 мин на гипотезу
- **Tree search**: reflections, branch selection, plateau detection
- **Git branches**: 5 веток `ai/hypothesis-*` в клоне
- **Solution tree persistence**: JSON с полным деревом экспериментов

### Проблемы

- **Refine/merge не улучшают**: все возвращают val_loss=0.209 (template fallback)
- **Wrapper generation всё ещё пустой**: Qwen3:8b возвращает 0 chars для wrapper, все используют один шаблон
- **Data acquisition не нашёл датасет**: fallback на синтетику
- **Скорость**: ~90 мин total (research 30 мин + 5 smoke 20 мин + 5 benchmark 25 мин + tree search 15 мин)

---

## Сводная таблица: что реализовано и проверено

| Фича | Реализована | Протестирована | Работает |
|------|------------|---------------|----------|
| ITER-RETGEN (итеративный ресерч) | да | да | **да** — 3 итерации, 45 фактов |
| Fact bank | да | да | **да** — конкретные claims с confidence |
| Source quality scoring | да | да | **да** — authority/recency/corroboration |
| Query expansion | да | да | **да** — synonym variants в запросах |
| Adaptive stopping | да | да | **да** — остановка при <10% novelty |
| Citation pre-embedding | да | да | **частично** — citations есть, формат не стандартный |
| Cross-source claim verification | да | да | **да** — contradictions detected |
| Confidence re-retrieval | да | да | **да** — доп. поиски для низко-confidence фактов |
| Context compaction | да | не проверена | ? — нужно >3 итераций для триггера |
| Autorater | да | да | **да** — quality score после каждого ресерча |
| Research persistence | да | да | **да** — checkpoint восстанавливается |
| Parallel filtering | да | да | **да** — 4 concurrent LLM calls |
| Parallel deep fetch | да | да | **да** — 8 concurrent downloads |
| URL dedup | да | да | **да** — 12 дупликатов убрано |
| Knowledge recall | да | не проверена | ? — нужен повторный прогон того же топика |
| Hybrid search (TF-IDF + n-gram) | да | не проверена | ? |
| PROGRESS.md | да | да | **частично** — пишется, но не обновляется |
| Fact-grounded hypotheses | да | да | **да** — H4 контрарианская |
| Research depth estimation | да | не проверена | ? |
| libcst AST analysis | да | да | **да** — 340 файлов, 0 ошибок |
| Tree search (AIDE) | да | да | **да** — 7 nodes, reflections |
| MLX benchmark | да | да | **да** — val_loss различается |

---

## Следующие шаги

1. **Фикс citation format** — стандартизировать `[1]` вместо `[Source [N]]`
2. **Фикс PROGRESS.md** — обновлять при каждой фазе
3. **Knowledge recall тест** — повторный прогон того же топика, проверить cache hit rate
4. **Wrapper generation** — программная генерация вместо LLM (Qwen 8B не справляется)
5. **Data acquisition** — добавить прямой HuggingFace Hub API search
6. **Tree search refine** — передавать конкретные hyp_params при refine/merge
