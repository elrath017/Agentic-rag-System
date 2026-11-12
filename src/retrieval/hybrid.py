from typing import List, Dict
from langchain_core.documents import Document
from src import config
from .dense import DenseRetriever
from .sparse import SparseBM25Retriever

class HybridRetriever:
    """Combines Dense Vector Search and Sparse BM25 Search using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseBM25Retriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever

    def search(self, query: str, top_k: int = config.TOP_K_RETRIEVAL, rrf_k: int = config.RRF_K_CONSTANT) -> List[Document]:
        """
        Executes both dense and sparse retrieval in parallel,
        fuses the ranks using Reciprocal Rank Fusion (RRF),
        and returns the top_k fused documents.
        """
        dense_results = self.dense.search(query, top_k=top_k)
        sparse_results = self.sparse.search(query, top_k=top_k)

        # RRF formula: RRF_score(doc) = sum(1 / (k + rank))
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Document] = {}

        # 1. Process dense results
        for rank, (doc, _score) in enumerate(dense_results, start=1):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            doc_store[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # 2. Process sparse results
        for rank, (doc, _score) in enumerate(sparse_results, start=1):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            doc_store[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # 3. Sort documents by their combined RRF score descending
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)

        fused_documents = [doc_store[d_id] for d_id in sorted_doc_ids[:top_k]]
        return fused_documents
