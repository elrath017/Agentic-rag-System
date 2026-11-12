from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import RAGNodes

def should_generate(state: AgentState) -> str:
    """Decides whether to generate the answer, rewrite the query, or fallback to web search."""
    if state.is_relevant:
        return "generate_answer"
    elif state.retry_count < 1:
        return "rewrite_query"
    else:
        return "web_search_fallback"

def create_agentic_rag_graph(nodes: RAGNodes):
    """Assembles and compiles the LangGraph state machine for autonomous RAG."""
    workflow = StateGraph(AgentState)

    # 1. Add all nodes
    workflow.add_node("retrieve_and_rerank", nodes.retrieve_and_rerank)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    workflow.add_node("web_search_fallback", nodes.web_search_fallback)
    workflow.add_node("generate_answer", nodes.generate_answer)
    workflow.add_node("grade_hallucination", nodes.grade_hallucination)

    # 2. Set Entry Point
    workflow.set_entry_point("retrieve_and_rerank")

    # 3. Add Edges
    workflow.add_edge("retrieve_and_rerank", "grade_documents")

    # Conditional branching based on relevance
    workflow.add_conditional_edges(
        "grade_documents",
        should_generate,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "web_search_fallback": "web_search_fallback",
        }
    )

    workflow.add_edge("rewrite_query", "retrieve_and_rerank")
    workflow.add_edge("web_search_fallback", "generate_answer")
    workflow.add_edge("generate_answer", "grade_hallucination")
    workflow.add_edge("grade_hallucination", END)

    return workflow.compile()
