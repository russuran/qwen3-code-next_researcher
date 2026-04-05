# Researcher Platform — Usage Guide

## Quick Start

### 1. Legacy CLI (без сервера, напрямую)

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить Ollama с моделью
ollama serve &
ollama pull qwen3:8b

# Запустить исследование
python cli.py "методы парсинга данных из паспорта РФ" --verbose
```

Результаты: `output/{topic_slug}/` — план, источники, анализ, отчёт, библиография.

### 2. API Server (FastAPI + Postgres + Redis)

```bash
# Скопировать env-файл
cp .env.example .env

# Поднять инфраструктуру
docker compose up -d postgres redis ollama

# Применить миграции
alembic upgrade head

# Запустить сервер
uvicorn app.main:app --port 8000

# Или через Docker Compose (всё вместе)
docker compose up -d
```

### 3. Thin CLI Client (через API)

```bash
python cli_api.py "vector databases for similarity search" --verbose
python cli_api.py "OCR methods comparison" --sources arxiv,github --max-results 10
```

### 4. REST API напрямую

```bash
# Создать исследование
curl -X POST http://localhost:8000/runs \
  -H 'Content-Type: application/json' \
  -d '{"topic": "transformer architectures", "sources": ["arxiv", "github"]}'

# Проверить статус
curl http://localhost:8000/runs/{run_id}

# Получить события
curl http://localhost:8000/runs/{run_id}/events

# SSE stream (live)
curl http://localhost:8000/runs/{run_id}/events/stream

# Артефакты
curl http://localhost:8000/runs/{run_id}/artifacts

# Отменить
curl -X POST http://localhost:8000/runs/{run_id}/cancel

# Health check
curl http://localhost:8000/health
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    User Interface                         │
│  cli.py (legacy)  │  cli_api.py (thin)  │  REST API     │
└────────┬──────────┴──────────┬──────────┴───────┬────────┘
         │                     │                  │
         ▼                     ▼                  ▼
┌──────────────────────────────────────────────────────────┐
│              FastAPI Application Layer                    │
│  app/main.py  │  app/api/  │  app/services/              │
│  Settings     │  Health    │  RunService                  │
│  (Pydantic)   │  Runs API  │  InstrumentedAgent          │
└───────────────┴──────┬─────┴─────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│  Postgres  │ │   Redis    │ │   Ollama   │
│  runs      │ │   queues   │ │   LLM      │
│  events    │ │   cache    │ │   models   │
│  artifacts │ │            │ │            │
└────────────┘ └────────────┘ └────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
┌────────────────────────────────────────────────────────┐
│                   Core Business Logic                   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Execution Engine (Phase 1)                     │   │
│  │  task_graph.py → scheduler.py → worker_pool.py  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Research Pipeline                              │   │
│  │  planner.py → searcher.py → agent.py            │   │
│  │  prompts.py   tools.py     llm.py               │   │
│  └─────────────────────────────────────────────────┘   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Knowledge Layer (Phase 2)                      │   │
│  │  source_registry → document_store → cache       │   │
│  │  index_manager  → change_detector               │   │
│  └─────────────────────────────────────────────────┘   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Repo Adaptation (Phase 3, Python-only)         │   │
│  │  repo_ingest → codebase_graph → patch_editor    │   │
│  │  git_versioning → test_oracle                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Sandbox & Eval (Phase 4)                       │   │
│  │  sandbox_runner → evaluator → benchmark_manager │   │
│  └─────────────────────────────────────────────────┘   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Network & Meta (Phase 5)                       │   │
│  │  network_policy → rate_limiter → meta_agent     │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── app/                          # FastAPI application layer
│   ├── main.py                   # App factory, lifespan, CORS
│   ├── settings.py               # Pydantic Settings from config/*.yaml
│   ├── deps.py                   # Dependency injection
│   ├── api/
│   │   ├── health.py             # GET /health, /ready
│   │   └── runs.py               # CRUD + SSE + cancel + artifacts
│   ├── schemas/runs.py           # Request/response models
│   ├── db/
│   │   ├── models.py             # SQLAlchemy: Run, Event, Artifact
│   │   └── session.py            # Async engine + session
│   └── services/
│       └── run_service.py        # InstrumentedAgent, background execution
│
├── core/                         # Business logic (research pipeline)
│   ├── llm.py                    # LLM abstraction (6 providers via litellm)
│   ├── tools.py                  # Tool registry (ReAct pattern)
│   ├── agent.py                  # ResearchAgent (plan→search→analyze→synthesize)
│   ├── planner.py                # Research plan generation
│   ├── task_graph.py             # DAG-based execution graph
│   ├── scheduler.py              # DAG scheduler with concurrency control
│   ├── worker_pool.py            # Node type → handler routing
│   ├── meta_agent.py             # Improvement loop for prompts/policies
│   ├── searcher.py               # arXiv, Semantic Scholar, GitHub, PWC
│   ├── browser.py                # Playwright + httpx fallback
│   ├── pdf_parser.py             # PDF/LaTeX/code tools
│   └── prompts.py                # Prompt templates
│
├── knowledge/                    # Knowledge management layer
│   ├── source_registry.py        # Source tracking with content hashing
│   ├── document_store.py         # Document storage + chunking
│   ├── cache_manager.py          # File-based cache (→ Redis later)
│   ├── index_manager.py          # TF-IDF search (→ pgvector later)
│   └── change_detector.py        # Staleness detection
│
├── repo_adaptation/              # Brownfield mode (Python repos)
│   ├── repo_ingest.py            # Clone + manifest builder
│   ├── codebase_graph.py         # Python AST → entity/edge graph
│   ├── git_versioning.py         # Branch management per task/variant
│   ├── patch_editor.py           # LLM-generated code patches
│   └── test_oracle.py            # Run existing test suite
│
├── sandbox/                      # Isolated execution + evaluation
│   ├── sandbox_runner.py         # Docker-based sandbox jobs
│   ├── evaluator.py              # Quality metrics (groundedness, coverage...)
│   └── benchmark_manager.py      # Benchmark suites + comparison
│
├── network/                      # Network access layer
│   └── network_policy.py         # Domain policies, rate limiting, retries
│
├── config/                       # YAML configuration files
│   ├── app.yaml                  # Server settings
│   ├── llm.yaml                  # LLM provider/model config
│   ├── compute.yaml              # Compute backend (local/vast)
│   ├── network.yaml              # Proxy/rate limit policies
│   ├── evals.yaml                # Evaluation config
│   └── policies.yaml             # Execution policies
│
├── alembic/                      # Database migrations
│   ├── env.py
│   └── versions/001_initial_schema.py
│
├── tests/                        # 96 tests
│   ├── test_agent.py             # Agent phases, ReAct parser
│   ├── test_api.py               # FastAPI endpoints
│   ├── test_cli.py               # Legacy CLI
│   ├── test_db.py                # ORM models
│   ├── test_knowledge.py         # Source registry, doc store, cache, index
│   ├── test_llm.py               # LLM config, providers, generate
│   ├── test_pdf_browser.py       # PDF/web tools
│   ├── test_planner.py           # Plan generation, JSON parsing
│   ├── test_repo_adaptation.py   # Ingest, AST graph, git, test oracle
│   ├── test_sandbox_eval.py      # Evaluator, benchmarks
│   ├── test_searcher.py          # Search API tools
│   ├── test_settings.py          # Config loading
│   └── test_task_graph.py        # DAG, scheduler
│
├── cli.py                        # Legacy CLI (direct mode)
├── cli_api.py                    # Thin CLI client (via API)
├── config.yaml                   # Legacy config (backward compat)
├── docker-compose.yml            # Postgres + Redis + Ollama + App
├── Dockerfile                    # Legacy CLI container
├── Dockerfile.api                # API server container
├── alembic.ini                   # Alembic config
├── requirements.txt              # All dependencies
└── .env.example                  # Environment variables template
```

---

## Configuration

### Смена LLM-провайдера

Редактируем `config/llm.yaml`:

```yaml
# Локальная модель через Ollama
provider: ollama
model: qwen3:8b
host: http://localhost:11434

# Или OpenAI
provider: openai
model: gpt-4o
api_key: sk-...

# Или Anthropic
provider: anthropic
model: claude-sonnet-4-20250514
api_key: sk-ant-...

# Или vLLM (self-hosted)
provider: vllm
model: qwen2.5-32b
host: http://gpu-server:8000
```

### Режимы генерации

```yaml
modes:
  thinking:        # Для планирования, анализа, синтеза
    temperature: 0.2
    max_tokens: 4096
  fast:            # Для быстрых задач, простых запросов
    temperature: 0.7
    max_tokens: 2048
```

---

## Brownfield Mode (работа с репозиториями)

```python
from repo_adaptation.repo_ingest import RepoIngest
from repo_adaptation.codebase_graph import CodebaseGraphBuilder
from repo_adaptation.git_versioning import GitVersioning
from repo_adaptation.patch_editor import PatchEditor
from repo_adaptation.test_oracle import TestOracle

# 1. Ingest repo
ingest = RepoIngest()
manifest = ingest.ingest("/path/to/repo")
print(f"Files: {manifest.total_files}, Languages: {manifest.languages}")

# 2. Build codebase graph
builder = CodebaseGraphBuilder()
graph = builder.build("/path/to/repo")
print(f"Entities: {len(graph.entities)}, Edges: {len(graph.edges)}")

# 3. Create branch for changes
git = GitVersioning("/path/to/repo")
git.create_branch("ai/task/fix-auth")

# 4. Generate patch
editor = PatchEditor(llm)
patch = await editor.generate_patch("auth.py", source_code, "Add input validation")
editor.apply_patch("/path/to/repo", patch)

# 5. Run tests
oracle = TestOracle("/path/to/repo")
result = oracle.run_tests()
print(f"Tests passed: {result.passed} ({result.total} total)")

# 6. Commit
git.commit("fix: add input validation to auth module", run_id="run-123")
```

---

## Running Tests

```bash
# Все тесты (96)
python -m pytest tests/ -v

# По модулю
python -m pytest tests/test_agent.py -v
python -m pytest tests/test_api.py -v
python -m pytest tests/test_knowledge.py -v
python -m pytest tests/test_repo_adaptation.py -v

# С покрытием
python -m pytest tests/ --cov=core --cov=app --cov=knowledge
```

---

## Docker Compose Services

| Service | Port | Purpose |
|---------|------|---------|
| `app` | 8000 | FastAPI server |
| `postgres` | 5432 | Database (runs, events, artifacts) |
| `redis` | 6379 | Cache, queues, SSE |
| `ollama` | 11434 | Local LLM server |
| `researcher` | — | Legacy CLI (docker compose run) |

```bash
# Полный стек
docker compose up -d

# Только инфра (для локальной разработки)
docker compose up -d postgres redis ollama

# Legacy CLI через Docker
docker compose run researcher "topic" --verbose
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/ready` | Readiness (DB + Redis) |
| POST | `/runs` | Create and start research run |
| GET | `/runs/{id}` | Get run status |
| GET | `/runs/{id}/events` | Get run events (journal) |
| GET | `/runs/{id}/events/stream` | SSE live event stream |
| POST | `/runs/{id}/cancel` | Cancel running research |
| GET | `/runs/{id}/artifacts` | List output artifacts |
| GET | `/runs/{id}/task-graph` | Get DAG structure |

---

## Key Metrics

- **96 tests**, all passing
- **6 LLM providers** supported (Ollama, vLLM, LM Studio, OpenAI, Anthropic, Gemini)
- **4 search sources** (arXiv, Semantic Scholar, GitHub, Papers With Code)
- **5 phases** implemented (research, knowledge, repo adaptation, sandbox, network)
- **3 storage backends** (Postgres, Redis, filesystem)
