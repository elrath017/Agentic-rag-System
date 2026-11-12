import json
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src import config

RERANK_PROMPT = """You are an expert Information Retrieval Re-ranker.
Your task is to evaluate the relevance of candidate text passages to a user query.

User Query: "{query}"

Candidate Passages:
{passages}

Task:
For each passage [ID], assign an integer relevance score from 0 to 10:
- 10: Directly and completely answers the query.
- 7-9: Highly relevant, contains essential contextual facts.
- 4-6: Marginally relevant or tangential context.
- 0-3: Irrelevant, noisy, or off-topic.

Output JSON ONLY in this format:
[
  {{"id": 0, "score": 9}},
  {{"id": 1, "score": 2}}
]
"""

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

class LLMReranker:
    """Re-ranks retrieved candidate passages using LLM relevance scoring."""

    def __init__(self, model_name: str = config.LLM_MODEL):
        api_key = config.GEMINI_API_KEY
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
        )

    def rerank(self, query: str, documents: List[Document], top_k: int = config.TOP_K_RERANKED) -> List[Document]:
        """Re-scores candidate documents and returns the top_k most relevant."""
        if not documents:
            return []
        if len(documents) <= top_k:
            return documents

        # Format candidate passages
        passages_text = ""
        for idx, doc in enumerate(documents):
            source = doc.metadata.get("source", "unknown")
            snippet = doc.page_content.replace("\n", " ")[:300]
            passages_text += f"[{idx}] Source: {source} | Content: {snippet}\n\n"

        prompt = RERANK_PROMPT.format(query=query, passages=passages_text)

        try:
            response = self.llm.invoke(prompt)
            content = _extract_text(response.content).strip()
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            scores = json.loads(content.strip())

            # Map scores back to documents
            scored_docs = []
            for item in scores:
                idx = item.get("id")
                score = item.get("score", 0)
                if isinstance(idx, int) and 0 <= idx < len(documents):
                    scored_docs.append((documents[idx], score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:top_k]]
        except Exception as e:
            # Fallback gracefully to original top_k if parsing fails
            print(f"[Warning] Reranker parsing failed: {e}. Falling back to default top_k.")
            return documents[:top_k]
