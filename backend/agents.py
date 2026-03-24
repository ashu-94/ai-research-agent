"""
agents.py — Five specialist agents with ChromaDB vector memory integration.

Speed optimisations (vs original):
  ① Right-sized models per agent
        Supervisor  → claude-haiku-4-5-20251001  (simple JSON planning)
        Reader      → claude-haiku-4-5-20251001  (repetitive summarisation)
        Fact-check  → claude-haiku-4-5-20251001  (bullet-point review)
        Writer      → claude-opus-4-5            (quality report synthesis)

  ② Parallel Search  — all Tavily queries fire simultaneously via ThreadPoolExecutor
  ③ Parallel Reader  — all summaries generated simultaneously via ThreadPoolExecutor
     (was the single biggest sequential bottleneck: 4 queries × ~15s = ~60s → now ~15s)

  ④ Retry logic      — exponential back-off on 529 overloaded, version-agnostic
"""

import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from vector_store import retrieve_similar, retrieve_relevant_chunks

logger = logging.getLogger(__name__)

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_BASE  = 3.0   # seconds; doubles each attempt: 3 → 6 → 12 → 24 → 48
RETRY_MAX   = 60.0


def _is_overloaded(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 529:
        return True
    msg = str(exc).lower()
    return "overloaded" in msg or "529" in msg


def _invoke_with_retry(llm, prompt: str) -> object:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            if not _is_overloaded(exc) or attempt == MAX_RETRIES:
                raise
            wait = min(RETRY_BASE * (2 ** (attempt - 1)), RETRY_MAX)
            logger.warning(f"Overloaded (attempt {attempt}/{MAX_RETRIES}), retry in {wait:.0f}s")
            time.sleep(wait)
            last_exc = exc
    raise last_exc


# ── Model factory ─────────────────────────────────────────────────────────────
def get_fast_llm():
    """Haiku — used for Supervisor, Reader, Fact-checker (fast + cheap)."""
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        max_tokens=1024,
    )

def get_quality_llm():
    """Opus — used only for the Writer (best synthesis quality)."""
    return ChatAnthropic(
        model="claude-opus-4-5",
        temperature=0,
        max_tokens=4096,
    )


# ── 1. Supervisor Agent ───────────────────────────────────────────────────────
def supervisor_agent(state: dict) -> dict:
    """
    Plans search subtasks AND checks ChromaDB for relevant prior research.
    Uses Haiku — task is simple structured JSON output.
    """
    llm = get_fast_llm()
    query = state["query"]
    logs = []

    prior = retrieve_similar(query, k=3)
    if prior:
        logs.append(f"[Supervisor] Found {len(prior)} related past research sessions in memory")
        for p in prior:
            logs.append(f"[Memory] '{p['past_query']}' (relevance: {p['score']})")
    else:
        logs.append("[Supervisor] No prior research found in vector memory — starting fresh")

    prior_context = ""
    if prior:
        prior_context = "\n\nNote: Similar past research exists on: " + \
            ", ".join(f"'{p['past_query']}'" for p in prior) + \
            ". Focus new searches on aspects not already covered."

    prompt = f"""You are a research supervisor. A user wants to research:

"{query}"{prior_context}

Break this into 3-4 specific, focused search queries for comprehensive coverage.
Return ONLY a JSON array of strings, no explanation.

Example: ["query 1", "query 2", "query 3"]
"""
    try:
        response = _invoke_with_retry(llm, prompt)
        content = response.content.strip().strip("```json").strip("```").strip()
        subtasks = json.loads(content)
    except json.JSONDecodeError:
        subtasks = [query]
    except Exception as exc:
        logs.append(f"[Supervisor] LLM error after retries: {exc}")
        subtasks = [query]

    logs.append(f"[Supervisor] Planned {len(subtasks)} search queries")
    return {
        "query":           query,
        "subtasks":        subtasks,
        "prior_knowledge": prior,
        "current_step":    "search",
        "agent_logs":      logs,
    }


# ── 2. Search Agent ───────────────────────────────────────────────────────────
def search_agent(state: dict) -> dict:
    """
    Runs all subtask queries in PARALLEL via ThreadPoolExecutor.
    Was: 4 queries x ~3s sequential = ~12s  →  now: ~3s total.
    """
    tasks = state.get("subtasks", [state["query"]])
    logs  = []
    results_map: dict[str, list] = {}

    def _search_one(task: str):
        tool = TavilySearchResults(max_results=3)
        try:
            hits = tool.invoke(task)
            return task, hits, None
        except Exception as e:
            return task, [], str(e)

    def _normalise(r, task: str) -> dict:
        """Handle both dict results and plain-string results from Tavily."""
        if isinstance(r, dict):
            return {
                "query":   task,
                "url":     r.get("url", ""),
                "title":   r.get("title", ""),
                "content": r.get("content", ""),
            }
        # Older Tavily versions return plain strings
        text = str(r)
        return {"query": task, "url": "", "title": "", "content": text}

    with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
        futures = {pool.submit(_search_one, t): t for t in tasks}
        for fut in as_completed(futures):
            task, hits, err = fut.result()
            results_map[task] = hits
            if err:
                logs.append(f"[Search] Error on '{task}': {err}")
            else:
                logs.append(f"[Search] {len(hits)} results for '{task}'")

    all_results = []
    for task in tasks:
        for r in results_map.get(task, []):
            all_results.append(_normalise(r, task))

    logs.append(f"[Search] Total sources collected: {len(all_results)}")
    return {
        "query":           state.get("query", ""),
        "subtasks":        state.get("subtasks", []),
        "prior_knowledge": state.get("prior_knowledge", []),
        "search_results":  all_results,
        "current_step":    "read",
        "agent_logs":      logs,
    }


# ── 3. Reader Agent ───────────────────────────────────────────────────────────
def reader_agent(state: dict) -> dict:
    """
    Summarises all query groups in PARALLEL with Haiku.
    Was: 4 x ~15s sequential = ~60s  →  now: ~15s total.
    """
    grouped: dict[str, list] = {}
    for r in state.get("search_results", []):
        grouped.setdefault(r["query"], []).append(r)

    logs: list[str] = []
    summary_map: dict[str, str] = {}

    def _summarise_one(q: str, results: list):
        llm = get_fast_llm()
        combined = "\n\n".join(
            f"Source: {r['title']} ({r['url']})\n{r['content'][:800]}"
            for r in results
        )
        prompt = f"""Summarise these search results for the query: "{q}"

Write 2-3 factual paragraphs citing the sources by title.

Results:
{combined}
"""
        try:
            response = _invoke_with_retry(llm, prompt)
            return q, f"## {q}\n\n{response.content.strip()}", None
        except Exception as exc:
            return q, f"## {q}\n\n[Summary unavailable — {exc}]", str(exc)

    with ThreadPoolExecutor(max_workers=min(len(grouped), 4)) as pool:
        futures = {pool.submit(_summarise_one, q, r): q for q, r in grouped.items()}
        for fut in as_completed(futures):
            q, summary, err = fut.result()
            summary_map[q] = summary
            if err:
                logs.append(f"[Reader] Failed '{q}': {err}")
            else:
                logs.append(f"[Reader] Summarised: '{q}'")

    subtask_order = state.get("subtasks", list(grouped.keys()))
    summaries = [summary_map[q] for q in subtask_order if q in summary_map]

    return {
        "query":           state.get("query", ""),
        "subtasks":        state.get("subtasks", []),
        "prior_knowledge": state.get("prior_knowledge", []),
        "search_results":  state.get("search_results", []),
        "summaries":       summaries,
        "current_step":    "fact_check",
        "agent_logs":      logs,
    }


# ── 4. Fact-Check Agent ───────────────────────────────────────────────────────
def fact_check_agent(state: dict) -> dict:
    """Reviews summaries for contradictions. Uses Haiku — bullet-point task."""
    llm = get_fast_llm()
    combined = "\n\n---\n\n".join(state.get("summaries", []))

    prompt = f"""You are a fact-checker. Review these research summaries on: "{state['query']}"

Check for:
1. Internal contradictions between sections
2. Claims needing verification
3. Missing important context
4. Overall confidence: High / Medium / Low

Be concise, use bullet points.

Summaries:
{combined}
"""
    try:
        response = _invoke_with_retry(llm, prompt)
        notes = [response.content.strip()]
    except Exception as exc:
        notes = [f"[Fact-check skipped — {exc}]"]

    return {
        "query":            state.get("query", ""),
        "subtasks":         state.get("subtasks", []),
        "prior_knowledge":  state.get("prior_knowledge", []),
        "search_results":   state.get("search_results", []),
        "summaries":        state.get("summaries", []),
        "fact_check_notes": notes,
        "current_step":     "writer",
        "agent_logs":       [f"[Fact-Check] Reviewed {len(state.get('summaries', []))} summaries"],
    }


# ── 5. Writer Agent ───────────────────────────────────────────────────────────
def writer_agent(state: dict) -> dict:
    """Synthesises the final report. Keeps Opus for quality output."""
    llm = get_quality_llm()
    logs = []

    prior_chunks = retrieve_relevant_chunks(state["query"], k=4)
    prior_context_text = ""
    if prior_chunks:
        prior_context_text = "\n\n### Relevant Prior Research (from memory):\n" + \
            "\n\n".join(f"- {c['text'][:300]}" for c in prior_chunks)
        logs.append(f"[Writer] Enriching report with {len(prior_chunks)} chunks from vector memory")
    else:
        logs.append("[Writer] No prior chunks found in memory — writing from current research only")

    combined_summaries = "\n\n---\n\n".join(state.get("summaries", []))
    fact_notes = "\n\n".join(state.get("fact_check_notes", []))

    prompt = f"""You are an expert research writer. Write a comprehensive report on: "{state['query']}"

Format in Markdown:
- Executive Summary (3-4 sentences)
- ## Key Findings (bullet points)
- ## Detailed Analysis (multiple sections with ## headings)
- ## Reliability Notes (from fact-check)
- ## Sources

Use all available information below.

### Current Research Summaries:
{combined_summaries}
{prior_context_text}

### Fact-Check Notes:
{fact_notes}
"""
    try:
        response = _invoke_with_retry(llm, prompt)
        final_report = response.content.strip()
        logs.append("[Writer] Report generated and ready for storage")
    except Exception as exc:
        final_report = (
            f"# Report Generation Failed\n\n"
            f"The Writer agent could not complete the report after {MAX_RETRIES} retries.\n\n"
            f"**Error:** {exc}\n\nPlease try again in a few minutes."
        )
        logs.append(f"[Writer] Failed after retries: {exc}")

    return {
        "query":            state.get("query", ""),
        "subtasks":         state.get("subtasks", []),
        "prior_knowledge":  state.get("prior_knowledge", []),
        "search_results":   state.get("search_results", []),
        "summaries":        state.get("summaries", []),
        "fact_check_notes": state.get("fact_check_notes", []),
        "final_report":     final_report,
        "current_step":     "store_memory",
        "agent_logs":       logs,
    }
