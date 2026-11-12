import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src import config

class SparseBM25Retriever:
    """Keyword-based sparse retrieval using BM25Okapi algorithm."""

    def __init__(self):
        self.documents: List[Document] = []
        self.bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Lowercase, strip punctuation and split into tokens
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def index_documents(self, documents: List[Document]) -> None:
        """Builds the BM25 index over the provided documents."""
        self.documents = list(documents)
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        """Queries the BM25 index and returns top-k documents with BM25 scores."""
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Pair documents with scores and sort descending
        doc_scores = [(self.documents[i], float(scores[i])) for i in range(len(self.documents))]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]
