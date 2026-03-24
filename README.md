# AI Research Assistant — Multi-Agent + Vector Memory

A production-grade multi-agent research pipeline with **persistent vector memory** via ChromaDB.
Built with **LangGraph**, **Claude**, **Tavily**, and **ChromaDB + SentenceTransformers**.

## What's new: Vector Memory

Every completed research session is automatically embedded and stored in ChromaDB.
On the next query, the Supervisor and Writer agents retrieve semantically similar past research
and use it to avoid redundant searches and enrich new reports.

```
Query → Supervisor checks ChromaDB → adapts search plan
                                          ↓
                                    Search + Reader + Fact-check
                                          ↓
                          Writer pulls relevant prior chunks → richer report
                                          ↓
                          Memory Agent stores full report + chunks to ChromaDB
```

## Architecture

```
backend/
├── main.py           FastAPI server + SSE streaming + /memory/* endpoints
├── graph.py          LangGraph StateGraph (6 nodes including store_memory)
├── agents.py         5 agents, all ChromaDB-aware
├── vector_store.py   ChromaDB wrapper (store, retrieve, stats, clear)
├── state.py          Shared state TypedDict
├── requirements.txt
├── .env.example
└── chroma_db/        Auto-created — persistent vector store on disk

frontend/
└── index.html        3-panel UI: Memory sidebar · Report · Agent trace
```

## Quick Start

### 1. Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in ANTHROPIC_API_KEY and TAVILY_API_KEY

uvicorn main:app --reload --port 8000
```

SentenceTransformers (~80MB) downloads automatically on first run.

### 2. Frontend

```bash
# Just open in browser:
open frontend/index.html

# Or serve:
cd frontend && python -m http.server 3000
```

### 3. API Keys

- **Anthropic** → https://console.anthropic.com
- **Tavily** (free) → https://tavily.com

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/research` | Run pipeline, stream SSE events |
| GET | `/memory/stats` | Report + chunk counts, recent topics |
| GET | `/memory/search?q=...` | Semantic search over past reports |
| DELETE | `/memory/clear` | Wipe all ChromaDB data |
| GET | `/health` | Health check |

## SSE Event Types

| Event | Payload | When |
|---|---|---|
| `step` | `{node, step}` | Each agent starts |
| `log` | `{message, node}` | Agent activity log |
| `prior_knowledge` | `{hits[]}` | Supervisor finds past research |
| `report` | `{report}` | Writer completes |
| `saved` | `{report_id}` | Memory agent stores to ChromaDB |
| `error` | `{message}` | Any failure |

## ChromaDB Collections

| Collection | Contents | Embedding |
|---|---|---|
| `research_reports` | Full markdown reports | all-MiniLM-L6-v2 (384-dim) |
| `source_chunks` | Individual source summaries | all-MiniLM-L6-v2 (384-dim) |


