import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from src import config
from src.ingestion.parser import DocumentParser
from src.ingestion.splitter import DocumentChunker
from src.ingestion.url_parser import URLArticleParser
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseBM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import LLMReranker
from src.agent.state import AgentState
from src.agent.nodes import RAGNodes
from src.agent.graph import create_agentic_rag_graph

app = FastAPI(
    title="Autonomous Agentic RAG API",
    description="Production-ready Agentic RAG microservice with hybrid search, reranking, and self-correction.",
    version="1.0.0"
)

# Global instances initialized on startup
dense_retriever: Optional[DenseRetriever] = None
sparse_retriever: Optional[SparseBM25Retriever] = None
hybrid_retriever: Optional[HybridRetriever] = None
reranker: Optional[LLMReranker] = None
agent_app = None

class QueryRequest(BaseModel):
    query: str

class URLIngestRequest(BaseModel):
    urls: List[str]

class QueryResponse(BaseModel):
    query: str
    generation: str
    sources: List[str]
    is_relevant: bool
    is_faithful: bool
    retry_count: int
    execution_logs: List[str]

@app.on_event("startup")
def startup_event():
    global dense_retriever, sparse_retriever, hybrid_retriever, reranker, agent_app
    if not config.GEMINI_API_KEY:
        print("[Warning] GEMINI_API_KEY is not set in environment or .env file.")
        return

    try:
        dense_retriever = DenseRetriever()
        sparse_retriever = SparseBM25Retriever()
        hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
        reranker = LLMReranker()
        rag_nodes = RAGNodes(hybrid_retriever=hybrid_retriever, reranker=reranker)
        agent_app = create_agentic_rag_graph(rag_nodes)
        print("[System] Agentic RAG Graph successfully compiled and ready.")
    except Exception as e:
        print(f"[Error] Failed to initialize agent graph: {e}")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "chroma_dir": str(config.CHROMA_DIR),
        "agent_ready": agent_app is not None
    }

@app.post("/api/query", response_model=QueryResponse)
def query_agent(req: QueryRequest):
    if not agent_app:
        raise HTTPException(status_code=500, detail="Agent graph not initialized. Check GEMINI_API_KEY.")

    initial_state = AgentState(query=req.query)
    final_output = agent_app.invoke(initial_state)

    return QueryResponse(
        query=req.query,
        generation=final_output["generation"],
        sources=final_output["sources"],
        is_relevant=final_output["is_relevant"],
        is_faithful=final_output["is_faithful"],
        retry_count=final_output["retry_count"],
        execution_logs=final_output["logs"],
    )

@app.post("/api/ingest")
def ingest_file(file: UploadFile = File(...)):
    if not dense_retriever or not sparse_retriever:
        raise HTTPException(status_code=500, detail="Retrievers not initialized.")

    upload_dir = config.DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Ingestion pipeline
    raw_docs = DocumentParser.parse_file(file_path)
    chunker = DocumentChunker()
    chunks = chunker.split_documents(raw_docs)

    dense_retriever.add_documents(chunks)
    sparse_retriever.index_documents(chunks)

    return {
        "message": f"Successfully ingested {file.filename}",
        "raw_pages": len(raw_docs),
        "indexed_chunks": len(chunks)
    }

@app.post("/api/ingest-url")
def ingest_urls(req: URLIngestRequest):
    if not dense_retriever or not sparse_retriever:
        raise HTTPException(status_code=500, detail="Retrievers not initialized.")

    if not req.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")

    parsed_docs = URLArticleParser.parse_urls(req.urls)
    if not parsed_docs:
        raise HTTPException(status_code=400, detail="Could not extract content from the provided URLs.")

    chunker = DocumentChunker()
    chunks = chunker.split_documents(parsed_docs)

    dense_retriever.add_documents(chunks)
    sparse_retriever.index_documents(chunks)

    return {
        "message": f"Successfully indexed {len(parsed_docs)} article(s)",
        "indexed_chunks": len(chunks),
        "articles": [doc.metadata.get("title", doc.metadata.get("source")) for doc in parsed_docs]
    }
