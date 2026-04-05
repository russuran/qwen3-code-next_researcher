# Техническое задание

## Автономная система поиска и анализа технологических решений

***

## 1. Назначение системы

Система предназначена для автоматизированного поиска, анализа и синтеза информации о существующих технологических решениях, библиотеках, подходах и best practices по заданной теме.

**Входные данные:** текстовый запрос с описанием области исследования (например: *"методы обработки файлов в распределённых системах"*).

**Выходные данные:** структурированный отчёт в формате Markdown, содержащий:
- список найденных решений с описанием подхода
- сравнительную таблицу методов
- примеры кода и ссылки на исходные репозитории
- библиографию источников

Система **не** предназначена для:
- генерации научных публикаций
- выполнения экспериментального кода
- проведения бенчмарков

***

## 2. Архитектурные требования

### 2.1. Общий принцип работы

Система работает по циклическому плану: **планирование → поиск → анализ → синтез**.

**Планирование:** на основе входного запроса формируется план исследования — разбивка на под-вопросы, определение источников поиска, ключевых слов, приоритетных направлений.

**Поиск:** параллельное выполнение поисковых запросов через API и web-источники.

**Анализ:** загрузка, парсинг и извлечение ключевой информации из найденных материалов (текст, PDF, исходный код).

**Синтез:** агрегация результатов, построение сравнительных таблиц, формирование итогового отчёта.

**Рекомендуемые референсные реализации:**
- Архитектура planner/execution/aggregator: https://github.com/assafelovic/gpt-researcher#architecture
- Multi-agent research pipeline: https://github.com/assafelovic/gpt-researcher/blob/main/multi_agents/README.md

### 2.2. Модельный бэкенд (свайпаемый)

Система должна поддерживать работу с различными провайдерами вычислительных моделей через единый интерфейс. Выбор провайдера и конкретной модели осуществляется через конфигурационный файл без изменения кода.

**Поддерживаемые провайдеры:**

| Провайдер | Тип | Примечание |
|---|---|---|
| Ollama | локальный | основной режим работы |
| vLLM | self-hosted | высокий throughput |
| LM Studio | локальный GUI-сервер | альтернативный локальный режим |
| OpenAI | облачный API | режим повышенной мощности |
| Anthropic | облачный API | режим quality-check |
| Google Gemini | облачный API | fallback-режим |

**Интерфейс бэкенда:**

Каждый бэкенд реализует метод `generate(prompt, mode)` с двумя режимами:
- **thinking** — низкая температура, высокая точность (планирование, анализ, критика)
- **fast** — высокая температура, скорость (генерация текста, простые запросы)

**Конфигурация:**

Выбор провайдера и модели задаётся в конфигурационном файле:
```yaml
llm:
  provider: ollama
  model: qwen3-coder-next
  host: http://localhost:11434
  modes:
    thinking:
      temperature: 0.2
      max_tokens: 4096
    fast:
      temperature: 0.7
      max_tokens: 2048
```

**Рекомендуемые референсные реализации:**
- Абстракция модели (SakanaAI): https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py
- Абстракция через LiteLLM (smolagents): https://github.com/huggingface/smolagents/blob/main/src/smolagents/agents.py

### 2.3. Инструментарий (ReAct-паттерн)

Система должна использовать паттерн ReAct (reason → act → observe) для вызова инструментов. Каждый инструмент регистрируется в центральном реестре и вызывается по имени.

| Инструмент | Назначение | Референс |
|---|---|---|
| `search_arxiv` | Поиск в arXiv через API | https://arxiv.org/help/api |
| `search_semantic_scholar` | Поиск через Semantic Scholar API | https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/generate_ideas.py |
| `search_github` | Поиск репозиториев через GitHub REST API | https://docs.github.com/en/rest |
| `search_papers_with_code` | Поиск ML-методов с кодом | https://paperswithcode.com/api |
| `browse_web` | Web-crawling (Selenium/Playwright) | https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/run_gaia.py |
| `download_pdf` | Загрузка PDF-файлов | — |
| `parse_pdf` | Извлечение текста из PDF | — |
| `parse_latex` | Парсинг LaTeX-исходников | https://github.com/arxiv-vanity/engrafo |
| `inspect_code` | Анализ фрагментов исходного кода | — |
| `compare_methods` | Построение сравнительной таблицы | — |

**Рекомендуемые референсные реализации:**
- Tool abstraction в smolagents: https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/run_gaia.py#L108-L150
- Query routing в DeepSearcher: https://github.com/zilliztech/deep-searcher

### 2.4. Внешний и внутренний цикл (bi-level architecture)

Система должна поддерживать двухуровневую архитектуру выполнения:

**Внешний цикл (outer loop):** планировщик и анализатор. Генерирует под-вопросы, анализирует промежуточные результаты, принимает решение о необходимости дополнительных поисковых итераций.

**Внутренний цикл (inner loop):** поисковик и парсер. Выполняет конкретные запросы к API и источникам, загружает и парсит материалы.

**Рекомендуемые референсные реализации:**
- SAGA bi-level architecture: https://arxiv.org/html/2512.21782v2
- SAGA исходный код: https://github.com/btyu/SAGA

### 2.5. Логирование и возможность интервенции

Все действия системы должны записываться в журнал (JSONL-формат) с указанием:
- временной метки
- этапа выполнения (plan, search, analyze, synthesize)
- выполненного действия (вызов инструмента, ответ модели и т.д.)
- результата

Журнал должен быть доступен для чтения в реальном времени, что позволяет оператору:
- отслеживать текущее состояние выполнения
- выявлять этапы, на которых система застряла
- вручную вмешиваться в процесс при необходимости

**Рекомендуемые референсные реализации:**
- Journal-based experiment tracking: https://github.com/SakanaAI/AI-Scientist#introduction
- Логирование в AgentLaboratory: https://github.com/SamuelSchmidgall/AgentLaboratory

***

## 3. Структура проекта

```
<project_root>/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config.yaml
│
├── core/
│   ├── __init__.py
│   ├── agent.py              # главный цикл управления
│   ├── planner.py            # генерация плана исследования
│   ├── searcher.py           # API-интеграции (arXiv, S2, GitHub, PWC)
│   ├── browser.py            # web-crawling (Selenium/Playwright)
│   ├── pdf_parser.py         # загрузка и парсинг PDF
│   ├── llm.py                # абстракция модельного бэкенда
│   ├── tools.py              # реестр инструментов (ReAct)
│   └── prompts.py            # шаблоны промптов
│
├── output/
│   └── {topic_slug}/
│       ├── 01_plan.json
│       ├── 02_sources.json
│       ├── 03_papers/
│       ├── 04_code/
│       ├── 05_summaries/
│       ├── 06_comparison.md
│       ├── 07_synthesis.md
│       └── 08_references.md
│
├── journal/
│   └── {topic_slug}.jsonl
│
└── cli.py
```

**Рекомендуемые референсные реализации структуры:**
- SakanaAI: https://github.com/SakanaAI/AI-Scientist/tree/main
- DeepSearcher: https://github.com/zilliztech/deep-searcher

***

## 4. Требования к контейнеризации

Система должна поставляться в виде Docker-образа, содержащего:
- Среда выполнения Python 3.11
- Сервер моделей (Ollama или совместимый)
- Playwright для web-crawling
- PyMuPDF для работы с PDF
- Все зависимости из requirements.txt

**Рекомендуемые референсные реализации:**
- Dockerfile от сообщества (SakanaAI): https://github.com/SakanaAI/AI-Scientist/blob/main/experimental/Dockerfile
- DeepSearcher Docker-конфигурация: https://github.com/zilliztech/deep-searcher

***

## 5. Конфигурация

Конфигурационный файл `config.yaml` должен содержать следующие секции:

```yaml
llm:
  provider: ollama
  model: qwen3-coder-next
  host: http://localhost:11434
  api_key: ${LLM_API_KEY}
  modes:
    thinking:
      temperature: 0.2
      max_tokens: 4096
    fast:
      temperature: 0.7
      max_tokens: 2048

search:
  sources:
    - arxiv
    - semantic_scholar
    - github
    - papers_with_code
  max_results_per_source: 20
  parallel: true

output:
  base_dir: ./output
  format: markdown
```

***

## 6. Командный интерфейс (CLI)

Система должна предоставлять CLI-интерфейс для запуска исследования:

```bash
python cli.py "тема исследования"
```

**Поддерживаемые опции:**

| Опция | Описание |
|---|---|
| `--output-dir` | Каталог для сохранения результатов |
| `--sources` | Список источников (arxiv,github,s2,pwc) |
| `--max-results` | Максимальное количество результатов на источник |
| `--verbose` | Режим подробного вывода прогресса |
| `--intervene` | Интерактивный режим с паузами между этапами |

**Рекомендуемые референсные реализации:**
- GPT-Researcher CLI: https://github.com/assafelovic/gpt-researcher
- DeepSearcher CLI: https://github.com/zilliztech/deep-searcher

***

## 7. API-интеграции

### 7.1. arXiv

- Официальная документация: https://arxiv.org/help/api
- Python-клиент: пакет `arxiv` (PyPI)
- Извлекаемые данные: заголовок, аннотация, авторы, категории, ссылка на PDF

### 7.2. Semantic Scholar

- Документация API: https://api.semanticscholar.org/api-docs/
- Референсная реализация: https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/generate_ideas.py
- Альтернатива (OpenAlex): https://github.com/SakanaAI/AI-Scientist#openalex-api-literature-search-alternative

### 7.3. GitHub

- REST API документация: https://docs.github.com/en/rest
- Поиск репозиториев, парсинг README, анализ структуры кода

### 7.4. Papers With Code

- API документация: https://paperswithcode.com/api
- Поиск методов машинного обучения с привязанным кодом

***

## 8. Порядок реализации

| № | Этап | Описание |
|---|---|---|
| 1 | Dockerfile + docker-compose | Поднятие среды выполнения с сервером моделей |
| 2 | core/llm.py | Абстракция бэкенда, фабрика провайдеров, режимы thinking/fast |
| 3 | core/tools.py | Реестр инструментов, заглушки для всех tools |
| 4 | core/searcher.py | Интеграция arXiv API + Semantic Scholar API |
| 5 | core/pdf_parser.py | Загрузка и парсинг PDF-файлов |
| 6 | core/planner.py | Генерация плана исследования (под-вопросы) |
| 7 | core/agent.py | Главный ReAct-цикл управления |
| 8 | cli.py | Командный интерфейс |
| 9 | config.yaml | Конфигурационный файл |
| 10 | Тестовый запуск | End-to-end проверка на простой теме |

***

## 9. Критерии приёмки

1. Система запускается через Docker Compose одной командой
2. Модельный бэкенд меняется изменением одной строки в config.yaml
3. CLI принимает тему и выводит прогресс в реальном времени
4. Результаты сохраняются в структурированную папку output/{topic}
5. Журнал действий записывается в JSONL и доступен для чтения
6. Интерактивный режим (`--intervene`) позволяет паузу между этапами
7. Все API-интеграции (arXiv, Semantic Scholar, GitHub, PWC) работают
8. PDF-загрузка и парсинг функционируют корректно
9. Итоговый отчёт содержит сравнительную таблицу и библиографию

***

## 10. Референсные репозитории

При реализации рекомендуется ознакомиться со следующими проектами:

| Репозиторий | Назначение референса |
|---|---|
| https://github.com/assafelovic/gpt-researcher | Архитектура planner→execution→aggregator, параллелизм |
| https://github.com/SakanaAI/AI-Scientist | Абстракция модели, API-интеграции, структура проекта |
| https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/run_gaia.py | ReAct-цикл, абстракция инструментов, browser tools |
| https://github.com/zilliztech/deep-searcher | Query routing, CLI, конфигурационный подход |
| https://github.com/btyu/SAGA | Двухуровневая архитектура outer/inner loop |
| https://github.com/SamuelSchmidgall/AgentLaboratory | Журналирование, multi-agent структура |
| https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/generate_ideas.py | Пример использования Semantic Scholar API |
| https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py | Паттерн абстракции модельного бэкенда |
| https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_review.py | Загрузка и парсинг PDF |
| https://github.com/SakanaAI/AI-Scientist/blob/main/experimental/Dockerfile | Конфигурация Docker-образа |
| https://arxiv.org/html/2512.21782v2 | Научная статья SAGA — описание bi-level архитектуры |
| https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/ | Архитектурные идеи агента для long-horizon задач |

***

## 11. Исключения

Следующие функциональности **не входят** в область данного проекта:
- Генерация публикаций в формате LaTeX
- Выполнение экспериментального/ML-кода
- Многоагентные системы с распределёнными ролями
- Веб-интерфейс (только CLI)