"""
graph.py — LangGraph pipeline with ChromaDB storage node at the end.

Flow:
  supervisor → search → reader → fact_check → writer → store_to_memory → END
"""
from langgraph.graph import StateGraph, END
from agents import (
    supervisor_agent,
    search_agent,
    reader_agent,
    fact_check_agent,
    writer_agent,
)
from vector_store import store_research


def memory_agent(state: dict) -> dict:
    """
    Persists the completed research session to ChromaDB.
    Runs after Writer so the report is available for future queries.
    """
    try:
        report_id = store_research(
            query=state.get("query", ""),
            report=state.get("final_report", ""),
            summaries=state.get("summaries", []),
        )
        return {
            
    "query": state.get("query", ""),
    "subtasks": state.get("subtasks", []),
    "prior_knowledge": state.get("prior_knowledge", []),
    "search_results": state.get("search_results", []),
    "summaries": state.get("summaries", []),
    "fact_check_notes": state.get("fact_check_notes", []),
    "final_report": state.get("final_report", ""),

    "report_id": report_id,

    "current_step": "complete",
    "agent_logs": [f"[Memory] Session saved to ChromaDB (id: {report_id})"],
}
        
    except Exception as e:
        
     return {
        "query": state.get("query", ""),
        "subtasks": state.get("subtasks", []),
        "prior_knowledge": state.get("prior_knowledge", []),
        "search_results": state.get("search_results", []),
        "summaries": state.get("summaries", []),
        "fact_check_notes": state.get("fact_check_notes", []),
        "final_report": state.get("final_report", ""),

        "report_id": "",

        "current_step": "complete",
        "agent_logs": [f"[Memory] Warning: could not save to ChromaDB — {e}"],
    }


def build_research_graph():
    graph = StateGraph(dict)

    graph.add_node("supervisor",    supervisor_agent)
    graph.add_node("search",        search_agent)
    graph.add_node("reader",        reader_agent)
    graph.add_node("fact_check",    fact_check_agent)
    graph.add_node("writer",        writer_agent)
    graph.add_node("store_memory",  memory_agent)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor",   "search")
    graph.add_edge("search",       "reader")
    graph.add_edge("reader",       "fact_check")
    graph.add_edge("fact_check",   "writer")
    graph.add_edge("writer",       "store_memory")
    graph.add_edge("store_memory", END)

    return graph.compile()

research_graph = build_research_graph()
