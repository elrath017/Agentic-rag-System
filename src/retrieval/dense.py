import os
from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src import config

class DenseRetriever:
    """Manages dense vector storage and cosine similarity retrieval using Gemini embeddings."""

    def __init__(self, persist_dir: str | Path = config.CHROMA_DIR):
        self.persist_dir = str(persist_dir)
        api_key = config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=api_key,
        )
        self.vector_store = Chroma(
            collection_name="agentic_rag_dense",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Embeds and indexes documents into Chroma."""
        if not documents:
            return
        self.vector_store.add_documents(documents)

    def search(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        """Performs dense similarity search returning (Document, similarity_score)."""
        # Chroma returns distance (smaller is closer)
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        return results_with_scores
