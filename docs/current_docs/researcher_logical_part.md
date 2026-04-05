# 1. Логическая часть

## 1.1. Область системы

Система представляет собой платформу для автоматизированного исследования, проектирования, модификации и валидации технологических решений.

Система должна поддерживать следующие классы задач:
- исследование технологического ландшафта;
- сравнение open-source и vendor‑решений;
- проектирование архитектуры решения;
- формирование roadmap реализации;
- построение benchmark и evaluation plan;
- проверка гипотез через sandbox execution;
- доработка существующего репозитория или модуля;
- генерация нескольких вариантов реализации в отдельных git‑ветках;
- сравнение вариантов на основе тестов, метрик и sandbox validation.

## 1.2. Цели

Система должна:
- автоматически преобразовывать пользовательскую постановку в формальную модель задачи;
- строить граф исполнения (DAG) с контролируемым параллелизмом;
- выполнять поиск, извлечение, анализ, синтез и валидацию;
- повторно использовать ранее накопленные знания и выполнять selective refresh;
- в режиме brownfield понимать структуру и архитектуру репозитория и безопасно его модифицировать;
- версионировать все изменения в git;
- использовать Docker‑sandbox для воспроизводимых экспериментов и проверок;
- иметь eval/benchmark‑слой для оценки качества решения задач и качества патчей;
- обеспечивать наблюдаемость, replay и анализ трасс.

## 1.3. Режимы работы

### 1.3.1. Greenfield mode

Режим используется для проектирования нового решения.

- Вход: текстовая постановка, ограничения, файлы, ссылки, лимиты.
- Выход: архитектурный пакет артефактов — отчёт, comparison‑матрица, roadmap, benchmark‑план, при необходимости результаты sandbox‑проверок.

### 1.3.2. Brownfield mode

Режим используется для модификации существующего решения.

- Вход: репозиторий, архив, локальная директория, модуль, набор файлов.
- Выход: один или несколько вариантов реализации в git‑ветках, результаты проверок, PR‑пакеты, артефакты анализа.

Поток выполнения для brownfield‑задачи:

```text
User: задача + repo
        |
        v
Intake: RunSpec (mode=brownfield, repo spec)
        |
        v
Repo ingest -> Snapshot -> Codebase graph -> Architecture map
        |
        v
Task model + Change plan
        |
        v
Planner -> DAG (research + code nodes)
        |
        v
Scheduler / Worker pool
   |                |
   |         GitVersioning: base branch -> ai/task/... -> ai/variant/...
   |                          |                |
   |                          v                v
   |                   PatchEditor        PatchEditor (variants)
   |                          |                |
   |                          v                v
   +--------------------> Sandbox / Tests / Eval
                               |
                               v
                        PR package + artifacts
```

## 1.4. Функциональные требования

### 1.4.1. Intake и формализация задачи

Система должна:
- принимать текстовый запрос, файлы, URL, git‑репозиторий, архив или локальный модуль;
- определять тип задачи: `factual_lookup`, `comparative_research`, `architecture_design`, `benchmark_task`, `repository_adaptation`, `patch_task`;
- выделять goal, expected output, constraints, assumptions, unknowns и target artifacts;
- формировать нормализованный `RunSpec`;
- запрашивать уточнения при недостаточной спецификации;
- сохранять входные данные и derived metadata в persistent storage.

### 1.4.2. Динамическое моделирование задачи

Система должна:
- строить формальную TaskModel без использования жёстких task profiles как основного механизма маршрутизации;
- выделять сущности, подзадачи, зависимости, риски и типы артефактов;
- определять требуемые capabilities;
- синтезировать execution policy на конкретный run;
- различать reusable priors и evidence, собранные под текущий run.

### 1.4.3. Граф исполнения

Система должна:
- представлять исполнение как DAG;
- поддерживать типы узлов: `clarify`, `search`, `fetch`, `parse`, `extract`, `analyze`, `compare`, `sandbox_run`, `synthesize`, `evaluate`, `publish`;
- выполнять dependency‑aware scheduling;
- поддерживать barrier‑узлы и aggregation points;
- обеспечивать bounded parallelism;
- поддерживать cancellation, retry, timeout и partial failure handling.

### 1.4.4. Исследовательский контур

Система должна:
- строить query portfolio;
- выполнять multi‑source search;
- получать данные из web, API, GitHub, документации, локальных файлов и репозиториев;
- парсить HTML, PDF, Markdown, JSON, README и API responses;
- извлекать claims, facts, metrics, constraints, code snippets и evidence spans;
- выполнять межисточниковый анализ и synthesis;
- выявлять противоречия и зоны неопределённости;
- формировать deliverables в Markdown, JSON и HTML.

### 1.4.5. Работа с существующей реализацией

Система должна:
- принимать репозиторий, архив, каталог или модуль как `RepositorySpec`;
- строить repository manifest;
- строить repository‑level codebase graph;
- восстанавливать архитектурные слои, boundaries и точки расширения;
- локализовать изменяемые сущности;
- выполнять impact analysis;
- строить patch plan до генерации diff;
- генерировать несколько patch candidates;
- валидировать patch candidates тестами и sandbox‑запусками;
- ранжировать кандидатов;
- формировать PR‑пакеты.

### 1.4.6. Git‑версионирование и варианты реализации

Система должна:
- использовать git как обязательный слой версионирования изменений;
- инициализировать репозиторий, если вход представлен архивом или папкой без `.git`;
- создавать ветку на задачу и отдельные ветки на варианты;
- поддерживать branch‑per‑task и branch‑per‑variant;
- поддерживать git worktrees для параллельной работы вариантов;
- фиксировать все патчи коммитами с metadata `run_id`, `task_id`, `patch_id`;
- сохранять diff между базовой и candidate веткой;
- формировать PR package;
- запрещать прямые изменения в protected branches.

### 1.4.7. Sandbox execution

Система должна:
- запускать Docker‑based sandbox jobs;
- поддерживать hypothesis checks, benchmark runs, patch validation и regression tests;
- задавать ограничения по CPU, RAM, времени и сетевому доступу;
- собирать stdout, stderr, exit status, артефакты и метрики;
- связывать sandbox job с веткой, commit SHA и patch candidate;
- поддерживать воспроизводимый rerun.

### 1.4.8. Quality и evaluation layer

Система должна:
- вычислять retrieval, parsing, synthesis, groundedness, efficiency и patch‑quality метрики;
- поддерживать deterministic evaluators и LLM‑as‑judge;
- выполнять claim verification;
- выполнять contradiction detection;
- поддерживать benchmark sets: train, validation и holdout;
- поддерживать replay выполненных runs;
- использовать benchmark и eval результаты для ранжирования patch candidates и run outcomes.

### 1.4.9. Improvement layer

Система может содержать controlled improvement loop, который:
- анализирует трассы и результаты eval;
- предлагает изменения prompts, policies, thresholds и routing rules;
- валидирует изменения на benchmark‑наборах;
- запрещает автоматическое изменение production source code без review;
- поддерживает patch registry, rollback и holdout validation.

### 1.4.10. UI и API

Система должна предоставлять:
- REST API;
- web dashboard;
- CLI как thin client поверх API;
- SSE для live updates;
- просмотр task graph, логов, артефактов, веток, результатов sandbox jobs и eval runs.

## 1.5. Нефункциональные требования

### 1.5.1. Надёжность

- Все run‑состояния должны быть recoverable после перезапуска сервиса.
- Все изменения в задачах, патчах и ветках должны быть детерминированно журналированы.
- Все sandbox jobs должны быть изолированы и иметь жёсткие resource limits.

### 1.5.2. Масштабируемость

- Система должна поддерживать параллельное исполнение нескольких runs.
- Планировщик должен поддерживать конфигурируемые concurrency limits на уровне run, node type, domain и sandbox jobs.
- Хранилище артефактов должно быть вынесено в object storage.

### 1.5.3. Воспроизводимость

- Каждый run должен иметь versioned config snapshot.
- Каждый patch candidate должен быть привязан к commit SHA.
- Каждая sandbox‑проверка должна быть повторяема по job manifest.
- Каждая eval‑конфигурация должна быть versioned.

### 1.5.4. Безопасность

- Пользовательский код не должен исполняться вне sandbox.
- Секреты провайдеров должны передаваться через env/config, а не жёстко кодироваться.
- Доступ к protected веткам должен быть ограничен policy‑слоем.
- Система не должна выполнять неконтролируемое самоизменение production‑кода.

## 1.6. Логическая архитектура

```text
User / API / Web UI / CLI
          |
          v
+------------------------------+
| FastAPI Application Layer    |
| runs, files, events, repos   |
+------------------------------+
          |
          v
+------------------------------+
| Intake & Task Modeling       |
| intent, clarify, task model  |
| capability match, policy     |
+------------------------------+
          |
          v
+------------------------------+
| Execution Graph Runtime      |
| planner, DAG, scheduler      |
| workers, barriers, retry     |
+------------------------------+
     /        |         \
    v         v          v
+------+  +--------+  +---------+
|Knowledge|Sandbox | |Quality   |
|Layer    |Layer   | |Layer     |
+------+  +--------+  +---------+
    \         |          /
     \        |         /
      v       v        v
     +----------------------+
     | Improvement Layer    |
     | controlled patches   |
     +----------------------+
```

## 1.7. Компонентная архитектура

### 1.7.1. Application layer

- `app/main.py` — точка входа FastAPI.
- `app/api/` — REST API модули.
- `app/dashboard/` — web dashboard.
- `cli.py` — CLI‑клиент поверх API.

### 1.7.2. Intake & task modeling

- `intent_interpreter.py`
- `clarifier.py`
- `task_model_builder.py`
- `capability_matcher.py`
- `policy_synthesizer.py`

### 1.7.3. Orchestration

- `planner.py`
- `task_graph.py`
- `scheduler.py`
- `worker_pool.py`
- `aggregator.py`
- `state.py`

### 1.7.4. Research core

- `searcher.py`
- `fetcher.py`
- `reader.py`
- `parser.py`
- `extractor.py`
- `analyzer.py`
- `synthesizer.py`
- `reporter.py`

### 1.7.5. Repository adaptation

- `repo_ingest.py`
- `repo_manifest_builder.py`
- `codebase_graph_builder.py`
- `architecture_mapper.py`
- `change_locator.py`
- `impact_analyzer.py`
- `change_planner.py`
- `patch_editor.py`
- `test_oracle.py`
- `patch_ranker.py`
- `pr_packager.py`
- `git_versioning.py`

### 1.7.6. Sandbox layer

- `sandbox_runner.py`
- `docker_runner.py`
- `sandbox_job_manager.py`
- `experiment_runner.py`

### 1.7.7. Knowledge layer

- `source_registry.py`
- `document_store.py`
- `cache_manager.py`
- `change_detector.py`
- `refresh_scheduler.py`
- `index_manager.py`

### 1.7.8. Network layer

- `network_policy.py`
- `proxy_manager.py`
- `session_pool.py`
- `rate_limiter.py`
- `proxy_health.py`

### 1.7.9. Quality layer

- `evaluator.py`
- `judges.py`
- `benchmark_manager.py`
- `trace_analyzer.py`
- `claim_verifier.py`
- `contradiction_detector.py`

### 1.7.10. Improvement layer

- `meta_agent.py`
- `patch_generator.py`
- `patch_validator.py`
- `registry.py`

## 1.8. Архитектура хранения

### 1.8.1. Выбор хранилищ

| Тип данных | Хранилище | Назначение |
|---|---|---|
| runs, tasks, configs, metadata, repo entities | PostgreSQL | источник истины |
| queues, locks, SSE buffers, hot runtime state | Redis | low‑latency coordination |
| uploads, reports, logs, screenshots, sandbox outputs | MinIO / S3‑compatible storage | object artifacts |
| embeddings, semantic index | pgvector или Qdrant | semantic retrieval |
| git repo working copies / worktrees | filesystem volume | branch/worktree execution |

### 1.8.2. Основные сущности данных

Система должна хранить:
- `runs`
- `run_inputs`
- `task_nodes`
- `task_edges`
- `events`
- `sources`
- `documents`
- `chunks`
- `facts`
- `artifacts`
- `repositories`
- `repo_snapshots`
- `repo_graphs`
- `architecture_maps`
- `change_requests`
- `patch_sets`
- `patch_validation_runs`
- `sandbox_jobs`
- `eval_runs`
- `patch_candidates`
- `patch_registry`

### 1.8.3. Артефактный pipeline

```text
source
  -> raw_content
  -> parsed_document
  -> chunks
  -> embeddings
  -> facts
  -> summaries
  -> comparisons
  -> final_report
```

### 1.8.4. Repository adaptation pipeline

```text
repo input
  -> snapshot
  -> manifest
  -> codebase graph
  -> architecture map
  -> change plan
  -> candidate branches
  -> patch validation
  -> PR package
```

## 1.9. Инструменты поиска и извлечения информации

Система должна предоставлять набор инструментов поиска и извлечения, которые используются DAG‑узлами `search`, `fetch`, `parse`, `extract`, `analyze`.

### 1.9.1. Web/API search

Инструмент `web_search` должен:
- принимать query portfolio, ограничения по доменам и бюджет по результатам;
- выполнять поиск по web и специализированным API;
- возвращать список кандидатов‑источников с URL, типом, сниппетом, оценкой релевантности и признаками дубликатов;
- поддерживать source diversity.

### 1.9.2. Repo search

Инструмент `repo_search` должен:
- искать и ранжировать репозитории по задаче;
- поддерживать поиск по GitHub и локальному каталогу репозиториев;
- отбирать кандидатов для глубокого ingest.

### 1.9.3. Document ingestion

Инструменты `fetcher`, `reader`, `parser` должны:
- скачивать содержимое через HTTP, browser и proxy policy;
- читать HTML, PDF, Markdown, README, JSON и API responses;
- нормализовать документы и разбивать их на чанки с метаданными.

### 1.9.4. Semantic search / RAG

Инструмент `vector_search` через `index_manager` должен:
- индексировать документы и чанки embeddings;
- поддерживать semantic и hybrid search;
- возвращать `ChunkRef` со ссылкой на исходный документ и источник.

### 1.9.5. Внутренний поиск по знаниям

Инструмент `docs_search` должен:
- искать по уже загруженным документам без обращения в сеть;
- использовать метаданные `source_registry`;
- отдавать результаты с указанием актуальности.

Инструмент `code_search` должен:
- выполнять поиск по кодовой базе;
- поддерживать поиск по именам файлов, классов, функций и текстовым совпадениям;
- при наличии graph‑данных поддерживать поиск по связям.

## 1.10. Инкрементальное обновление и кэширование

Система должна поддерживать dependency‑aware invalidation и selective refresh.

### 1.10.1. Обязательные механизмы

- content hashing;
- source versioning;
- parser versioning;
- embedding model versioning;
- prompt/version stamps для extractor и synthesizer;
- cache manifests;
- freshness policies;
- scheduled refresh;
- on‑demand refresh.

### 1.10.2. Типы кэшей

- fetch cache;
- parse cache;
- chunk cache;
- embedding cache;
- extraction cache;
- synthesis cache;
- evaluator cache.

### 1.10.3. Change signals

Для источников и репозиториев должны учитываться:
- `content_hash`
- `etag`
- `last_modified`
- `commit_sha`
- `tag/release`
- `parser_version`
- `prompt_version`

## 1.11. Инструменты актуализации уже изученных данных

Система должна уметь актуализировать знания и кэши выборочно, а не полным пересбором.

### 1.11.1. Реестр источников

Модуль `source_registry` должен:
- хранить canonical metadata источников;
- хранить версию источника:
  - для web/API — `etag`, `last_modified`, `content_hash`;
  - для репозиториев — `commit_sha`, `tag/release`;
- предоставлять API регистрации и обновления источников.

### 1.11.2. Детектор изменений

Модуль `change_detector` должен:
- сравнивать текущие и сохранённые метаданные источника;
- формировать dirty set изменившихся источников;
- определять список зависимых артефактов.

### 1.11.3. Планировщик обновлений

Модуль `refresh_scheduler` должен:
- поддерживать on‑demand, scheduled и event‑driven refresh;
- запускать pipeline:

```text
fetch -> parse -> chunk -> embed -> extract -> rebuild summaries -> update indexes
```

- ограничивать объём обновлений по времени, типу источников и бюджету.

### 1.11.4. Управление кэшами

Модуль `cache_manager` должен:
- управлять fetch, parse, chunk, embedding, extraction, synthesis и evaluator cache;
- поддерживать `get`, `set`, invalidate по источнику, артефакту и глобальным признакам;
- вести cache manifest со связью источник → производные артефакты.

### 1.11.5. Обновление индексов

Модуль `index_manager` должен:
- поддерживать vector index и при необходимости полнотекстовый индекс;
- удалять устаревшие и добавлять новые векторы;
- обновлять метаданные свежести и версии;
- поддерживать bulk update и частичную пересборку.

### 1.11.6. Поведение при устаревании знаний

Система должна:
- уметь отмечать артефакты как устаревшие, но временно используемые;
- в отчётах указывать, от каких версий источников получены данные;
- поддерживать режим:

```yaml
refresh_mode: strict | relaxed
```

- `strict` — refresh обязателен перед использованием при существенных изменениях;
- `relaxed` — допускается использование старых артефактов с пометкой о возможном устаревании.

## 1.12. Сетевой слой

Сетевой слой должен быть конфигурируемым и отделённым от бизнес‑логики.

### 1.12.1. Режимы доступа

- direct HTTP
- direct browser
- proxy HTTP
- proxy browser
- unlocker/unblock mode

### 1.12.2. Bright Data

Система должна поддерживать Bright Data как внешний proxy/unblock provider.

### 1.12.3. Domain policy

Для домена должны настраиваться:
- preferred transport;
- fallback order;
- session affinity;
- max concurrency;
- retry strategy;
- cooldown policy.

## 1.13. Compute и LLM‑конфигурация

Конфигурация приложения должна собираться через env/yaml и typed settings, а выбор моделей должен проходить через единый gateway‑слой, что соответствует типовым практикам FastAPI/Pydantic settings и multi‑provider LLM routing. [fastapi.tiangolo](https://fastapi.tiangolo.com/advanced/settings/)

Система должна поддерживать два слоя выбора:
- **Compute‑слой**: `local` или `vast`;
- **LLM‑слой**: `local_llm` или `openai_llm`.

Vast.ai предоставляет программный API для управления GPU‑инстансами и документирует сценарий поиска оффера и создания инстанса через API‑ключ, поэтому compute‑режим `vast` должен строиться как отдельный backend, а не как ручная внешняя операция. [vast](https://vast.ai/developers/api)

Документация Vast.ai также показывает использование `VAST_API_KEY` как переменной окружения и OpenAI‑совместимых endpoints для vLLM‑подобных workloads, что делает режим `vast + local_llm` совместимым с unified gateway‑подходом. [docs.vast](https://docs.vast.ai/documentation/serverless/quickstart)

Схема выбора compute и LLM:

```text
                +----------------------+
User / Config   |  config: compute,   |
--------------> |  llm backend        |
                +----------+-----------+
                           |
                           v
                    +--------------+
                    | Orchestrator |
                    +------+-------+
                           |
             +-------------+--------------+
             |                            |
             v                            v
     +---------------+             +---------------+
     | ComputeBackend|             |  LLMGateway   |
     +-------+-------+             +-------+-------+
             |                             |
   +---------+---------+          +--------+--------+
   |                   |          |                 |
   v                   v          v                 v
+--------+        +--------+  +----------+   +--------------+
| local  |        | Vast.ai|  | local_llm|   |  openai_llm  |
| Docker |        | GPU VM |  |  (vLLM)  |   |  (OpenAI)    |
+--------+        +--------+  +----------+   +--------------+
```

## 1.14. Git и branch orchestration

### 1.14.1. Стратегия ветвления

Система должна поддерживать:
- `ai/task/<run_id>-<slug>`
- `ai/variant/<run_id>-<variant_id>`
- `ai/explore/<run_id>-<slug>`
- `ai/ship/<run_id>-<slug>`
- `ai/stack/<run_id>-<step>`

### 1.14.2. Инварианты

- Один patch candidate — одна ветка.
- Один набор параллельных candidate implementations — отдельные worktrees.
- Все промежуточные изменения должны быть зафиксированы commit history.
- Main/release ветки не должны изменяться автоматически.
- Результаты тестов должны быть связаны с конкретной веткой и commit SHA.

### 1.14.3. Обязательные функции

- init repo
- clone/fetch/checkout branch
- create branch
- create worktree
- commit changes
- diff base..candidate
- merge simulation
- package PR artifacts

## 1.15. Docker sandbox

### 1.15.1. Job model

Каждый sandbox job должен включать:
- `job_id`
- `run_id`
- `job_type`
- `image`
- `entrypoint`
- `timeout_sec`
- `memory_mb`
- `cpu_limit`
- `network_policy`
- `mounted_inputs`
- `expected_outputs`
- `branch_name`
- `commit_sha`

### 1.15.2. Поддерживаемые типы job

- hypothesis check
- benchmark run
- unit test run
- integration test run
- regression test run
- build/packaging run
- sample‑based validation

## 1.16. API

### 1.16.1. REST endpoints

```text
POST   /runs
GET    /runs/{id}
POST   /runs/{id}/start
POST   /runs/{id}/cancel
POST   /runs/{id}/files
GET    /runs/{id}/events
GET    /runs/{id}/trace
GET    /runs/{id}/task-graph
GET    /runs/{id}/artifacts
POST   /runs/{id}/repositories
GET    /runs/{id}/branches
GET    /runs/{id}/patches
POST   /runs/{id}/patches/{patch_id}/validate
POST   /benchmarks/run
GET    /benchmarks/{id}
```

### 1.16.2. Live updates

Система должна предоставлять SSE stream с типами событий:
- `run_state_changed`
- `task_started`
- `task_completed`
- `task_failed`
- `branch_created`
- `patch_generated`
- `sandbox_started`
- `sandbox_completed`
- `artifact_published`
- `eval_completed`

## 1.17. Форматы данных

### 1.17.1. RunSpec

```yaml
run_spec:
  query: string
  mode: greenfield | brownfield
  constraints: []
  files: []
  urls: []
  repository:
    url: string | null
    branch: string | null
    commit: string | null
    path: string | null
  change_target:
    module: string | null
    file_scope: []
  budgets:
    max_parallel_subtasks: int
    max_parallel_fetches: int
    max_parallel_sandbox_jobs: int
  git_policy:
    init_if_missing: bool
    create_branches: bool
    create_worktrees: bool
    push_enabled: bool
  allow_code_changes: none | sandbox_only | repo
  autonomy_level: full | semi | advisory
  refresh_mode: strict | relaxed
```

### 1.17.2. PatchCandidate

```yaml
patch_candidate:
  patch_id: string
  run_id: string
  base_branch: string
  candidate_branch: string
  commit_sha: string
  changed_files: []
  rationale: string
  validation_plan: []
  status: draft | validating | accepted | rejected
```

## 1.18. Наблюдаемость

### 1.18.1. Журналирование

Все события должны записываться в JSONL и/или PostgreSQL events stream.

### 1.18.2. Поля события

Каждое событие должно включать:
- `timestamp`
- `run_id`
- `task_node_id`
- `stage`
- `action`
- `status`
- `latency_ms`
- `cost_estimate`
- `branch_name`
- `commit_sha`
- `artifact_refs`
- `parent_event_id`

## 1.19. Тонкие моменты и инварианты поведения

### 1.19.1. Полуготовые решения и точка остановки

Система должна:
- иметь явный критерий автоматической остановки;
- различать режимы `draft` и `final`;
- при “почти решено” прекращать бесконечный loop и формировать “remaining gaps”.

### 1.19.2. Конфликты целей и ограничений

Система должна:
- иметь слой согласования ограничений;
- приоритизировать hard constraints, затем budget, затем quality;
- при конфликте явно предлагать trade‑off варианты.

### 1.19.3. Когда разрешены изменения кода

В `RunSpec` должен быть флаг:

```yaml
allow_code_changes: none | sandbox_only | repo
```

### 1.19.4. Обращение с серой зоной в выводах

Система должна:
- иметь обязательный раздел “Ограничения и неопределённости”;
- различать supported claims и hypotheses;
- не включать неподтверждённые ключевые утверждения без явной маркировки.

### 1.19.5. Приоритет человеческого вмешательства

В `RunSpec` должен быть флаг:

```yaml
autonomy_level: full | semi | advisory
```

Система должна поддерживать approve/deny в ключевых точках.

### 1.19.6. Поведение при инфраструктурных сбоях

Система должна:
- различать временные и постоянные ошибки;
- поддерживать fallback по compute и LLM;
- логировать тип сбоя и предпринятые действия.

### 1.19.7. Ограничение improvement loop

Система должна:
- ограничивать поверхность изменений prompts/policies/config;
- не трогать production code и holdout benchmarks;
- применять изменения только после validation;
- поддерживать rollback.

### 1.19.8. Эскалация неразрешимых задач

Система должна поддерживать статус:

```yaml
resolution_status: solved | partial | unsolvable | blocked
```

### 1.19.9. Жизненный цикл гипотез

Система должна:
- рассматривать каждую гипотезу как независимый экземпляр;
- не смешивать гипотезы неявно;
- поддерживать статусы:

```yaml
hypothesis_status: draft | running | validated | rejected | combined
```

## 1.20. Критерии приёмки

Система считается принятой при выполнении условий:

1. Платформа разворачивается через Docker Compose.
2. Доступны FastAPI backend, CLI и web dashboard.
3. Система принимает текстовые задачи, файлы и git‑репозитории.
4. Для каждого run формируется `RunSpec` и TaskModel.
5. Planner строит DAG и исполняет его с bounded parallelism.
6. Система сохраняет источники, документы, chunks, артефакты и события.
7. Реализован selective refresh без полного перерасчёта knowledge base.
8. Реализован network layer с direct/proxy/browser режимами.
9. Реализована интеграция с Bright Data как configurable provider.
10. Реализован repository adaptation workflow.
11. Система строит repository graph и локализует область изменений.
12. Система создаёт git‑ветки для task run и candidate variants.
13. Система поддерживает worktree для параллельных вариантов.
14. Реализован Docker sandbox для benchmark и patch validation.
15. Каждый patch candidate валидируется через тесты и/или sandbox.
16. Доступны diff, branch metadata и PR‑ready артефакты.
17. Реализован evaluator layer и benchmark manager.
18. Доступен replay по журналу.
19. Improvement layer ограничен prompts/policies/config и проходит validation.
20. Все protected branches защищены от прямых автоматических изменений.

## 1.21. Ограничения

- Система не должна выполнять неконтролируемое изменение production‑кода.
- Система не должна автоматически merge‑ить candidate branch в protected branch без явного approval policy.
- Система не должна исполнять пользовательский код вне sandbox.
- Система не должна использовать жёсткие task profiles как primary mechanism understanding.

## 1.22. Риски и меры снижения

### 1.22.1. Неверная декомпозиция задачи

Меры:
- clarification gate;
- coverage checks;
- replanning.

### 1.22.2. Неполное понимание кодовой базы

Меры:
- repository graph;
- architecture map;
- impact analysis;
- staged patching.

### 1.22.3. Слишком крупные diffs

Меры:
- branch‑per‑task;
- branch‑per‑variant;
- diff size thresholds;
- stacked branches.

### 1.22.4. Ложноположительные patch validations

Меры:
- test oracle;
- regression suite;
- sample‑based validation;
- sandbox rerun.

### 1.22.5. Reward hacking improvement loop

Меры:
- holdout benchmarks;
- patch registry;
- rollback;
- high‑impact review.

# 2. Практическая часть

## 2.1. Структура проекта

```text
project_root/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env
├── config/
│   ├── app.yaml
│   ├── llm.yaml
│   ├── compute.yaml
│   ├── network.yaml
│   ├── evals.yaml
│   └── policies.yaml
├── app/
│   ├── main.py
│   ├── deps.py
│   ├── api/
│   ├── dashboard/
│   └── schemas/
├── core/
│   ├── intent_interpreter.py
│   ├── clarifier.py
│   ├── task_model_builder.py
│   ├── capability_matcher.py
│   ├── policy_synthesizer.py
│   ├── planner.py
│   ├── task_graph.py
│   ├── scheduler.py
│   ├── worker_pool.py
│   ├── aggregator.py
│   ├── state.py
│   ├── searcher.py
│   ├── fetcher.py
│   ├── reader.py
│   ├── parser.py
│   ├── extractor.py
│   ├── analyzer.py
│   ├── synthesizer.py
│   ├── reporter.py
│   ├── evaluator.py
│   ├── judges.py
│   ├── benchmark_manager.py
│   ├── trace_analyzer.py
│   ├── claim_verifier.py
│   ├── contradiction_detector.py
│   ├── meta_agent.py
│   ├── patch_generator.py
│   ├── patch_validator.py
│   ├── registry.py
│   ├── llm.py
│   ├── tools.py
│   └── prompts.py
├── repo_adaptation/
│   ├── repo_ingest.py
│   ├── repo_manifest_builder.py
│   ├── codebase_graph_builder.py
│   ├── architecture_mapper.py
│   ├── change_locator.py
│   ├── impact_analyzer.py
│   ├── change_planner.py
│   ├── patch_editor.py
│   ├── test_oracle.py
│   ├── patch_ranker.py
│   ├── pr_packager.py
│   └── git_versioning.py
├── sandbox/
│   ├── sandbox_runner.py
│   ├── docker_runner.py
│   ├── sandbox_job_manager.py
│   └── experiment_runner.py
├── knowledge/
│   ├── source_registry.py
│   ├── document_store.py
│   ├── cache_manager.py
│   ├── change_detector.py
│   ├── refresh_scheduler.py
│   └── index_manager.py
├── network/
│   ├── network_policy.py
│   ├── proxy_manager.py
│   ├── session_pool.py
│   ├── rate_limiter.py
│   └── proxy_health.py
├── benchmarks/
│   ├── research/
│   │   ├── train.jsonl
│   │   └── validation.jsonl
│   └── repo/
│       ├── train.jsonl
│       └── validation.jsonl
├── artifacts/
├── uploads/
├── repos/
├── journal/
└── cli.py
```

## 2.2. Переменные окружения

```text
APP_ENV=dev
LOG_LEVEL=info

# LLM
OPENAI_API_KEY=...
VLLM_BASE_URL=http://localhost:8000/v1
OLLAMA_BASE_URL=http://localhost:11434

# Vast.ai
VAST_API_KEY=...
VAST_DEFAULT_OFFER_QUERY=gpu_ram>=24 num_gpus=1 dph<=1.0

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=agent
POSTGRES_USER=agent
POSTGRES_PASSWORD=agent_password

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# MinIO / S3
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minio_access
S3_SECRET_KEY=minio_secret
S3_BUCKET=agent-artifacts
S3_REGION=us-east-1

# Proxy provider
BRIGHTDATA_USERNAME=...
BRIGHTDATA_PASSWORD=...
BRIGHTDATA_DEFAULT_ZONE=...
BRIGHTDATA_BROWSER_ZONE=...
```

## 2.3. Основные YAML‑конфиги

### 2.3.1. `config/app.yaml`

```yaml
app:
  env: ${APP_ENV}
  log_level: ${LOG_LEVEL}
  base_url: http://localhost:8001
  enable_dashboard: true

runtime:
  max_parallel_runs: 4
  max_parallel_subtasks: 4
  max_parallel_fetches: 8
  max_parallel_sandbox_jobs: 2

storage:
  artifacts_bucket: ${S3_BUCKET}
  uploads_bucket: ${S3_BUCKET}
  journal_path: ./journal
```

### 2.3.2. `config/llm.yaml`

```yaml
llm:
  default_backend: local_llm

  local_llm:
    kind: vllm
    base_url: ${VLLM_BASE_URL}
    model: qwen2.5-32b

  openai_llm:
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: gpt-4.5
```

### 2.3.3. `config/compute.yaml`

```yaml
compute:
  provider: local

  local:
    docker_host: unix:///var/run/docker.sock

  vast:
    api_key: ${VAST_API_KEY}
    offer_query: ${VAST_DEFAULT_OFFER_QUERY}
    image: vllm/vllm-openai:latest
    disk_gb: 100
    ssh_enabled: false
```

### 2.3.4. `config/network.yaml`

```yaml
network:
  default_mode: direct

  providers:
    brightdata:
      username: ${BRIGHTDATA_USERNAME}
      password: ${BRIGHTDATA_PASSWORD}
      zones:
        default: ${BRIGHTDATA_DEFAULT_ZONE}
        browser: ${BRIGHTDATA_BROWSER_ZONE}

  domain_policies:
    github.com:
      mode: direct
      max_concurrency: 4

    arxiv.org:
      mode: proxy
      provider: brightdata
      zone: default
      fallback_order: [direct, proxy, browser]
```

### 2.3.5. `config/evals.yaml`

```yaml
evals:
  default:
    groundedness: true
    coverage: true
    source_diversity: true

  benchmark_sets:
    research_train: ./benchmarks/research/train.jsonl
    research_validation: ./benchmarks/research/validation.jsonl
    repo_train: ./benchmarks/repo/train.jsonl
    repo_validation: ./benchmarks/repo/validation.jsonl
```

### 2.3.6. `config/policies.yaml`

```yaml
policies:
  execution:
    max_parallel_subtasks: 4
    max_parallel_fetches: 8
    max_parallel_sandbox_jobs: 2
    default_retry_count: 2

  sandbox:
    default_timeout_sec: 600
    default_memory_mb: 4096

  git:
    protected_branches:
      - main
      - master
      - release/*
    branch_prefixes:
      task: ai/task
      variant: ai/variant
      explore: ai/explore
      ship: ai/ship
```

## 2.4. Сущности БД

```text
runs(
  id, created_at, mode, status, spec_json
)

run_inputs(
  run_id, raw_query, constraints_json, repository_spec_json
)

repositories(
  id, run_id, url, branch, commit_sha, local_path
)

repo_snapshots(
  id, repository_id, created_at, commit_sha, path
)

repo_graphs(
  id, repository_id, snapshot_id, graph_json
)

change_requests(
  id, run_id, repository_id, target_module, status
)

patch_sets(
  id, change_request_id, base_branch, meta_json
)

patch_validation_runs(
  id, patch_set_id, status, metrics_json
)

artifacts(
  id, run_id, kind, uri, meta_json
)

sandbox_jobs(
  id, run_id, patch_set_id, job_type, status, meta_json
)

eval_runs(
  id, run_id, eval_type, metrics_json
)
```

## 2.5. Docker Compose

```yaml
version: "3.9"
services:
  app:
    build: .
    env_file: .env
    command: uvicorn app.main:app --host 0.0.0.0 --port 8001
    ports:
      - "8001:8001"
    depends_on:
      - postgres
      - redis
      - minio

  worker:
    build: .
    env_file: .env
    command: python -m worker
    depends_on:
      - postgres
      - redis

  scheduler:
    build: .
    env_file: .env
    command: python -m scheduler
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  minio:
    image: minio/minio
    environment:
      MINIO_ACCESS_KEY: ${S3_ACCESS_KEY}
      MINIO_SECRET_KEY: ${S3_SECRET_KEY}
    command: server /data
    ports:
      - "9000:9000"
```