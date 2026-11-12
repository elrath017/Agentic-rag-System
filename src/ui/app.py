import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
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

st.set_page_config(
    page_title="Autonomous Agentic RAG Engine",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Autonomous Agentic RAG System")
st.caption("Hybrid Search (BM25 + Dense Chroma) • Cross-Encoder Reranking • News & Web Links Ingestion • Self-Correcting LangGraph")

# Sidebar Configuration (API key is loaded securely from .env, hidden from UI)
with st.sidebar:
    st.header("⚙️ System Status")
    if config.GEMINI_API_KEY:
        st.success("🟢 Gemini API Connected")
        st.caption(f"LLM: `{config.LLM_MODEL}` | Embeddings: `{config.EMBEDDING_MODEL}`")
    else:
        st.error("🔴 GEMINI_API_KEY Missing in .env")

    st.divider()
    st.subheader("📥 Knowledge Base Ingestion")

    # Tabs for File Upload vs. Web/News Links
    ingest_tab1, ingest_tab2 = st.tabs(["📄 Local Files", "🌐 News & Articles"])

    chunker = DocumentChunker()

    def get_or_create_retrievers():
        if "dense_retriever" not in st.session_state:
            st.session_state["dense_retriever"] = DenseRetriever()
        if "sparse_retriever" not in st.session_state:
            st.session_state["sparse_retriever"] = SparseBM25Retriever()
        if "indexed_chunks_list" not in st.session_state:
            st.session_state["indexed_chunks_list"] = []
        return st.session_state["dense_retriever"], st.session_state["sparse_retriever"]

    with ingest_tab1:
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )

        if st.button("Index Files", type="primary"):
            if not config.GEMINI_API_KEY:
                st.error("Gemini API key is not configured in .env.")
            elif not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Processing & indexing files..."):
                    upload_dir = config.DATA_DIR / "uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    new_chunks = []

                    for f in uploaded_files:
                        save_path = upload_dir / f.name
                        with open(save_path, "wb") as buffer:
                            buffer.write(f.read())
                        raw_docs = DocumentParser.parse_file(save_path)
                        chunks = chunker.split_documents(raw_docs)
                        new_chunks.extend(chunks)

                    dense, sparse = get_or_create_retrievers()
                    dense.add_documents(new_chunks)
                    
                    st.session_state["indexed_chunks_list"].extend(new_chunks)
                    sparse.index_documents(st.session_state["indexed_chunks_list"])
                    st.session_state["indexed_count"] = len(st.session_state["indexed_chunks_list"])

                    st.success(f"Indexed {len(new_chunks)} chunks from {len(uploaded_files)} file(s)!")

    with ingest_tab2:
        st.markdown("Ingest text directly from online news and article URLs:")
        urls_input = st.text_area(
            "Article / News URLs (one per line):",
            placeholder="https://techcrunch.com/article-slug\nhttps://bbc.com/news/article-slug",
            height=120
        )

        if st.button("Fetch & Index URLs", type="primary"):
            if not config.GEMINI_API_KEY:
                st.error("Gemini API key is not configured in .env.")
            else:
                raw_urls = [u.strip() for u in urls_input.split("\n") if u.strip()]
                if not raw_urls:
                    st.warning("Please enter at least one valid URL.")
                else:
                    with st.spinner(f"Fetching and parsing {len(raw_urls)} web article(s)..."):
                        parsed_docs = URLArticleParser.parse_urls(raw_urls)

                        if not parsed_docs:
                            st.error("Could not extract article text from the provided URL(s). Check if the links are public.")
                        else:
                            new_chunks = chunker.split_documents(parsed_docs)
                            dense, sparse = get_or_create_retrievers()
                            dense.add_documents(new_chunks)

                            st.session_state["indexed_chunks_list"].extend(new_chunks)
                            sparse.index_documents(st.session_state["indexed_chunks_list"])
                            st.session_state["indexed_count"] = len(st.session_state["indexed_chunks_list"])

                            st.success(f"Successfully scraped & indexed {len(new_chunks)} chunks from {len(parsed_docs)} article(s)!")
                            for doc in parsed_docs:
                                st.caption(f"• `{doc.metadata.get('title', doc.metadata.get('source'))}`")

    if "indexed_count" in st.session_state and st.session_state["indexed_count"] > 0:
        st.divider()
        st.metric("Total Indexed Chunks", st.session_state["indexed_count"])

# Initialize Agent Pipeline
def get_agent_pipeline():
    if "dense_retriever" in st.session_state and "sparse_retriever" in st.session_state:
        dense = st.session_state["dense_retriever"]
        sparse = st.session_state["sparse_retriever"]
    else:
        dense = DenseRetriever()
        sparse = SparseBM25Retriever()
    
    hybrid = HybridRetriever(dense, sparse)
    reranker = LLMReranker()
    nodes = RAGNodes(hybrid, reranker)
    return create_agentic_rag_graph(nodes)

# Query & Chat Interface
tabs = st.tabs(["💬 Agent Query & Execution Graph", "📊 Ragas Evaluation"])

with tabs[0]:
    user_query = st.text_input(
        "Ask a question to your ingested knowledge base (documents + articles):",
        placeholder="e.g. What were the key announcements or financial figures reported?"
    )

    if st.button("Submit Query", type="primary") and user_query:
        if not config.GEMINI_API_KEY:
            st.error("Please ensure GEMINI_API_KEY is present in your .env file.")
        else:
            try:
                with st.spinner("Executing autonomous agent loop..."):
                    agent_graph = get_agent_pipeline()
                    initial_state = AgentState(query=user_query)
                    result = agent_graph.invoke(initial_state)

                col1, col2 = st.columns([3, 2])

                with col1:
                    st.subheader("💡 Grounded Answer")
                    st.markdown(result["generation"])

                    st.markdown("### 📚 Cited Sources")
                    for src in result["sources"]:
                        st.markdown(f"- `{src}`")

                with col2:
                    st.subheader("🔍 Agent Reasoning & Flow")
                    st.write(f"**Relevance Confirmed:** `{'✅ Yes' if result['is_relevant'] else '❌ No'}`")
                    st.write(f"**Faithfulness / Grounded:** `{'✅ Passed' if result['is_faithful'] else '⚠️ Flagged'}`")
                    st.write(f"**Query Rewrites / Retries:** `{result['retry_count']}`")

                    st.markdown("#### Execution Trace")
                    for log in result["logs"]:
                        st.info(log)

            except Exception as e:
                st.error(f"Execution Error: {e}")

with tabs[1]:
    st.subheader("📊 Quantitative Ragas Evaluation Benchmark")
    st.markdown("""
    This module evaluates your RAG pipeline using the **Ragas** framework:
    - **Faithfulness**: Measures factual consistency of the answer with the context (hallucination detector).
    - **Answer Relevance**: Measures how pertinent the generated answer is to the user question.
    - **Context Precision**: Evaluates if the highest-ranking retrieved chunks contain the ground truth.
    - **Context Recall**: Measures if all information needed to answer the question was retrieved.
    """)
    st.info("To run the full evaluation suite, execute in terminal: `python -m src.evaluation.run_benchmark`")
