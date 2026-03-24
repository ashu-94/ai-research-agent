"""
main.py — FastAPI backend with SSE streaming + ChromaDB endpoints.

Endpoints:
  POST /research          — run pipeline, stream agent logs via SSE
  GET  /memory/stats      — ChromaDB collection stats for the UI
  GET  /memory/search     — semantic search over past reports
  DELETE /memory/clear    — wipe the ChromaDB collections
  GET  /health
"""

import json
import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from graph import research_graph
from vector_store import get_collection_stats, retrieve_similar

app = FastAPI(title="AI Research Assistant API")

@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str


# ── Research endpoint ─────────────────────────────────────────────────────────
@app.post("/research")
async def research(request: ResearchRequest):
    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            initial_state = {
                "query": request.query,
                "subtasks": [],
                "prior_knowledge": [],
                "search_results": [],
                "summaries": [],
                "fact_check_notes": [],
                "final_report": "",
                "report_id": "",
                "agent_logs": [],
                "current_step": "supervisor",
            }

            async for event in research_graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in event.items():

                    if "current_step" in node_output:
                        yield {
                            "event": "step",
                            "data": json.dumps({
                                "step": node_output["current_step"],
                                "node": node_name,
                            }),
                        }

                    if "agent_logs" in node_output:
                        for log in node_output["agent_logs"]:
                            yield {
                                "event": "log",
                                "data": json.dumps({"message": log, "node": node_name}),
                            }
                            await asyncio.sleep(0.04)

                    # Stream prior knowledge hits so UI can show them
                    if "prior_knowledge" in node_output and node_output["prior_knowledge"]:
                        yield {
                            "event": "prior_knowledge",
                            "data": json.dumps({"hits": node_output["prior_knowledge"]}),
                        }

                    if "final_report" in node_output and node_output["final_report"]:
                        yield {
                            "event": "report",
                            "data": json.dumps({"report": node_output["final_report"]}),
                        }

                    if "report_id" in node_output and node_output["report_id"]:
                        yield {
                            "event": "saved",
                            "data": json.dumps({"report_id": node_output["report_id"]}),
                        }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}),
            }

    return EventSourceResponse(event_generator())


# ── Memory / ChromaDB endpoints ───────────────────────────────────────────────
@app.get("/memory/stats")
def memory_stats():
    return get_collection_stats()


@app.get("/memory/search")
def memory_search(q: str = Query(..., description="Semantic search query")):
    results = retrieve_similar(q, k=5)
    return {"results": results}


@app.delete("/memory/clear")
def memory_clear():
    """Wipe all ChromaDB collections (dev/demo use)."""
    import chromadb, os, shutil
    from vector_store import CHROMA_DIR, _chroma_client
    try:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        # Reset the singleton so next call re-creates fresh
        import vector_store
        vector_store._chroma_client = None
        vector_store._embeddings = None
        return {"status": "cleared"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as f:
        return f.read()
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
