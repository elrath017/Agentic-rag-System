import json
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from src import config
from src.retrieval.hybrid import HybridRetriever
from .state import AgentState

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif hasattr(p, "text"):
                parts.append(getattr(p, "text", ""))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)

class RAGNodes:
    """Implements all reasoning, retrieval, grading, and generation nodes in the agent graph."""

    def __init__(self, hybrid_retriever: HybridRetriever, reranker=None):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.0,
        )

    def retrieve_and_rerank(self, state: AgentState) -> Dict[str, Any]:
        """Retrieves candidates using hybrid search and re-ranks to the top most relevant."""
        query_to_search = state.rewritten_query if state.rewritten_query else state.query
        log_entry = f"Searching: '{query_to_search}' using Hybrid Retrieval (BM25 + Dense Chroma)..."

        # 1. Hybrid retrieve
        candidates = self.hybrid_retriever.search(query_to_search, top_k=config.TOP_K_RETRIEVAL)

        # 2. Re-rank
        if self.reranker and candidates:
            final_docs = self.reranker.rerank(query_to_search, candidates, top_k=config.TOP_K_RERANKED)
        else:
            final_docs = candidates[:config.TOP_K_RERANKED]

        sources = list(set([doc.metadata.get("source", "Document") for doc in final_docs]))
        return {
            "documents": final_docs,
            "sources": sources,
            "logs": state.logs + [log_entry, f"Retrieved & reranked {len(final_docs)} chunks."]
        }

    def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """Evaluates whether retrieved documents contain enough factual context to answer the query."""
        query = state.query
        docs = state.documents

        if not docs:
            return {
                "is_relevant": False,
                "logs": state.logs + ["No documents found. Routing to query rewriter / web fallback."]
            }

        context_str = "\n\n".join([f"Doc {i+1}: {d.page_content}" for i, d in enumerate(docs)])

        prompt = f"""You are a strict Retrieval Evaluator.
Determine if the provided context contains enough relevant factual information to answer the user query.

User Query: "{query}"

Context:
{context_str}

Respond with JSON ONLY in this format:
{{"is_relevant": true, "reason": "concise explanation"}}
or
{{"is_relevant": false, "reason": "concise explanation"}}
"""
        response = self.llm.invoke(prompt)
        content = _extract_text(response.content).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        try:
            res = json.loads(content.strip())
            is_relevant = bool(res.get("is_relevant", False))
            reason = res.get("reason", "")
        except Exception:
            is_relevant = len(docs) > 0
            reason = "Defaulted to available documents."

        return {
            "is_relevant": is_relevant,
            "logs": state.logs + [f"Relevance Assessment: {is_relevant} ({reason})"]
        }

    def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """Rewrites user query to optimize semantic search if previous retrieval was insufficient."""
        query = state.query
        prompt = f"""You are an Information Retrieval Query Optimizer.
The user's query failed to retrieve relevant documents.
Rewrite the query to be clearer, more specific, and optimized for vector semantic search.

Original Query: "{query}"

Return ONLY the rewritten query text without any formatting, quotes, or explanation.
"""
        response = self.llm.invoke(prompt)
        rewritten = _extract_text(response.content).strip().replace('"', '')
        return {
            "rewritten_query": rewritten,
            "retry_count": state.retry_count + 1,
            "logs": state.logs + [f"Optimized Query -> '{rewritten}' (Retry {state.retry_count + 1})"]
        }

    def web_search_fallback(self, state: AgentState) -> Dict[str, Any]:
        """Queries the web when local documents do not contain the answer."""
        query = state.rewritten_query if state.rewritten_query else state.query
        log_entry = f"Executing Web Search Fallback for: '{query}'..."

        # Priority 1: Tavily API (if configured in .env)
        if config.TAVILY_API_KEY:
            try:
                import requests
                resp = requests.post(
                    "https://api.tavily.com/search",
                    json={"api_key": config.TAVILY_API_KEY, "query": query, "max_results": 3},
                    timeout=10
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    web_docs = []
                    sources = []
                    for r in results:
                        web_docs.append(Document(
                            page_content=f"{r.get('title', '')}: {r.get('content', '')}",
                            metadata={"source": r.get("url", "Tavily Search"), "page": "web"}
                        ))
                        sources.append(r.get("url", "Tavily Search"))

                    if web_docs:
                        return {
                            "documents": state.documents + web_docs,
                            "sources": list(set(state.sources + sources)),
                            "is_relevant": True,
                            "logs": state.logs + [log_entry, f"Tavily retrieved {len(web_docs)} web passages successfully."]
                        }
            except Exception as e:
                log_entry += f" (Tavily failed: {e})"

        # Priority 2: DuckDuckGo Fallback
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            web_docs = [
                Document(
                    page_content=f"{r.get('title', '')}: {r.get('body', '')}",
                    metadata={"source": r.get("href", "DuckDuckGo"), "page": "web"}
                )
                for r in results
            ]
            sources = [r.get("href", "DuckDuckGo") for r in results]
            return {
                "documents": state.documents + web_docs,
                "sources": list(set(state.sources + sources)),
                "is_relevant": True,
                "logs": state.logs + [log_entry, "DuckDuckGo results gathered successfully."]
            }
        except Exception as e:
            return {
                "logs": state.logs + [log_entry, f"Web search failed: {e}. Proceeding with best-effort docs."]
            }

    def generate_answer(self, state: AgentState) -> Dict[str, Any]:
        """Generates a grounded, factual answer with exact citations from the context."""
        query = state.query
        docs = state.documents

        context_str = "\n\n".join([
            f"[Source: {d.metadata.get('source', 'Unknown')} | Page: {d.metadata.get('page', 1)}]\n{d.page_content}"
            for d in docs
        ])

        system_prompt = """You are a rigorous, truthful AI assistant.
Answer the user's question STRICTLY based on the provided context.
Rules:
1. Every major fact or claim MUST cite its source like [Source: filename, Page: X].
2. If the context does not contain the answer, explicitly state: 'The provided documents do not contain enough information to answer this question.' Do not fabricate details.
3. Be structured, concise, and professional.
"""

        user_prompt = f"""Context:
{context_str}

User Question: {query}

Answer:"""

        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        return {
            "generation": _extract_text(response.content).strip(),
            "logs": state.logs + ["Answer generated with citations."]
        }

    def grade_hallucination(self, state: AgentState) -> Dict[str, Any]:
        """Verifies if the generated answer is strictly grounded in the retrieved documents."""
        docs = state.documents
        answer = state.generation
        context_str = "\n\n".join([d.page_content for d in docs])

        prompt = f"""You are a strict Fact-Checking Auditor.
Verify if the substantive factual claims in the generated answer are grounded in and supported by the context.

Guidelines:
1. Ignore standard citation brackets like [Source: ..., Page: ...].
2. If the answer explicitly states that the provided documents do not contain enough information, mark is_faithful as true.
3. Mark is_faithful as false ONLY if the answer fabricates facts, numbers, or assertions not supported by the context.

Context:
{context_str}

Generated Answer:
{answer}

Respond with JSON ONLY in this format:
{{"is_faithful": true, "reason": "concise explanation"}}
or
{{"is_faithful": false, "reason": "concise explanation"}}
"""
        response = self.llm.invoke(prompt)
        content = _extract_text(response.content).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        try:
            res = json.loads(content.strip())
            is_faithful = bool(res.get("is_faithful", True))
            reason = res.get("reason", "")
        except Exception:
            is_faithful = True
            reason = "Auditor check passed by default."

        return {
            "is_faithful": is_faithful,
            "logs": state.logs + [f"Faithfulness / Hallucination Check: {is_faithful} ({reason})"]
        }
