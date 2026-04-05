# Статус-отчёт: Autonomous Research Agent

## Дата: 2026-04-03

---

## 1. Что сделано

### Реализован полный pipeline по ТЗ (milestone1.md)

| Модуль | Файл | Статус | Описание |
|--------|------|--------|----------|
| LLM-абстракция | `core/llm.py` | Готов | 6 провайдеров (Ollama, vLLM, LM Studio, OpenAI, Anthropic, Gemini) через litellm. Два режима: thinking/fast |
| Реестр инструментов | `core/tools.py` | Готов | Decorator-based registration, ReAct-паттерн, единый формат ToolResult |
| Поиск | `core/searcher.py` | Готов | arXiv, Semantic Scholar (httpx), GitHub REST API, Papers With Code |
| Web-crawling | `core/browser.py` | Готов | Playwright async + httpx fallback, singleton browser |
| PDF/LaTeX/Code | `core/pdf_parser.py` | Готов | Download PDF, parse PDF (PyMuPDF), parse LaTeX, inspect GitHub code |
| Промпты | `core/prompts.py` | Готов | 6 шаблонов: ReAct system, plan generation, search refinement, analysis, synthesis, comparison |
| Планировщик | `core/planner.py` | Готов | Генерация плана (подвопросы + приоритеты), адаптивный refine |
| Агент | `core/agent.py` | Готов | Bi-level: outer loop (plan→search→analyze→synthesize), inner loop (ReAct), JSONL-журнал, intervention mode |
| CLI | `cli.py` | Готов | Click CLI с опциями: --output-dir, --sources, --max-results, --verbose, --intervene, --config |
| Docker | `Dockerfile` + `docker-compose.yml` | Готов | Python 3.11-slim + Playwright + Ollama как сервис |
| Конфиг | `config.yaml` | Готов | Все секции по ТЗ, default model = qwen3:8b |

### Тесты

**55 тестов, все проходят.**

| Тест-файл | Покрытие | Тестов |
|-----------|----------|--------|
| `test_tools.py` | Регистрация, execute, ошибки, format_for_prompt, ToolResult | 10 |
| `test_llm.py` | Конфиг, model string для всех 6 провайдеров, api_base, generate, generate_structured | 14 |
| `test_searcher.py` | arXiv, Semantic Scholar, GitHub, Papers With Code (все на моках) | 4 |
| `test_pdf_browser.py` | Download PDF, parse PDF, truncation, LaTeX, inspect_code, browse_web | 6 |
| `test_planner.py` | JSON extraction (5 кейсов), generate_plan, bad JSON handling, refine_plan | 8 |
| `test_agent.py` | ReAct parser (4 кейса), journal, full pipeline e2e, intervention abort, react loop | 8 |
| `test_cli.py` | Help, run, missing config, sources override | 4 |

Запуск: `.venv/bin/python3 -m pytest tests/ -v`

---

## 2. Тестовый запуск

### Параметры

- **Тема:** "методы парсинга данных из паспорта РФ: OCR, распознавание текста, библиотеки компьютерного зрения"
- **Модель:** qwen3:8b (Q4_K_M, 5.2GB) через Ollama
- **Железо:** MacBook Pro M4 Pro, 24GB RAM
- **Время выполнения:** ~35 минут

### Результаты по фазам

| Фаза | Результат | Файл |
|------|-----------|------|
| Plan | 5 подвопросов с приоритетами и ключевыми словами | `01_plan.json` (2.8KB) |
| Search | 100 результатов из API (GitHub, Semantic Scholar) | `02_sources.json` (99KB) |
| Analyze | 86 уникальных источников проанализированы | `05_summaries/` (86 файлов) |
| Synthesize | Markdown-отчёт: обзор, методы, таблица, рекомендации | `07_synthesis.md` (9KB) |
| References | Библиография | `08_references.md` (11KB) |

### Что работало

- Ollama + qwen3:8b — стабильная генерация, модель не падала
- GitHub Search API — нашёл релевантные репозитории (tesseract, yolov5, opencv)
- Semantic Scholar API — нашёл академические статьи по OCR
- Полный pipeline plan→search→analyze→synthesize отработал от начала до конца
- JSONL-журнал записывался корректно

### Проблемы при запуске

| Проблема | Причина | Статус |
|----------|---------|--------|
| arXiv API не отвечал | DNS resolution error (сеть) | Не критично — другие источники компенсировали |
| Papers With Code — 302 redirect | API изменил URL-схему | Исправлено (`follow_redirects=True`), но PwC отвечал с 429 при повторах |
| Semantic Scholar — 429 rate limit | Параллельные запросы превысили лимит | Исправлено: добавлен delay + httpx вместо SDK |
| `06_comparison.md` пустой | compare_methods не вернул данные (модель не сгенерировала таблицу в нужном формате) | Требует доработки |
| Отчёт поверхностный | qwen3:8b — маленькая модель, не хватает глубины анализа | Ожидаемо для 8B |

### Оценка качества отчёта (qwen3:8b)

- **Структура:** корректная, все секции по шаблону
- **Содержание:** общие рекомендации (Tesseract, YOLO, OpenCV, CNN), без глубоких деталей
- **Релевантность:** модель сама отметила, что источники не специфичны для паспортов РФ
- **Вывод:** для production-качества нужна модель крупнее (Claude/GPT-4) или qwen3:32b+ на машине с 48GB+

---

## 3. Архитектура (итог)

```
cli.py → ResearchAgent
              │
              ├─ Planner (LLM: thinking mode)
              │    └─ generate_plan() → ResearchPlan
              │
              ├─ Search (parallel tools)
              │    ├─ search_arxiv
              │    ├─ search_semantic_scholar
              │    ├─ search_github
              │    └─ search_papers_with_code
              │
              ├─ Analyze (LLM: fast mode × N sources)
              │    └─ per-source structured analysis
              │
              └─ Synthesize (LLM: thinking mode)
                   ├─ 07_synthesis.md
                   ├─ 06_comparison.md
                   └─ 08_references.md
```

Смена модели/провайдера — одна строка в `config.yaml`:
```yaml
llm:
  provider: anthropic        # вместо ollama
  model: claude-sonnet-4-20250514  # вместо qwen3:8b
```

---

## 4. Что дальше (рекомендации)

1. **Подключить облачную модель** (Claude/GPT-4) для качественных отчётов — архитектура готова
2. **Починить compare_methods** — парсинг ответа модели для сравнительных таблиц
3. **Добавить retry/fallback** для arXiv и Papers With Code API
4. **Протестировать на 2-3 реальных задачах** с разными темами
5. **Добавить browse_web** в pipeline для парсинга найденных страниц (сейчас не используется в основном цикле)
6. **Рассмотреть qwen3:32b** если доступна машина с 48GB+ RAM
