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
st.caption("Hybrid Search (BM25 + Dense Chroma) • Cross-Encoder Reranking • Self-Correcting LangGraph • Ragas Evaluation")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    gemini_key = st.text_input(
        "Gemini API Key",
        value=config.GEMINI_API_KEY,
        type="password",
        help="Get your key from Google AI Studio (https://aistudio.google.com/)",
    )
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        config.GEMINI_API_KEY = gemini_key

    st.divider()
    st.subheader("📁 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Index Documents", type="primary"):
        if not gemini_key:
            st.error("Please enter a valid Gemini API Key first.")
        elif not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing & Indexing documents..."):
                upload_dir = config.DATA_DIR / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                all_chunks = []
                chunker = DocumentChunker()

                for f in uploaded_files:
                    save_path = upload_dir / f.name
                    with open(save_path, "wb") as buffer:
                        buffer.write(f.read())
                    raw_docs = DocumentParser.parse_file(save_path)
                    chunks = chunker.split_documents(raw_docs)
                    all_chunks.extend(chunks)

                dense = DenseRetriever()
                sparse = SparseBM25Retriever()
                dense.add_documents(all_chunks)
                sparse.index_documents(all_chunks)

                st.session_state["dense_retriever"] = dense
                st.session_state["sparse_retriever"] = sparse
                st.session_state["indexed_count"] = len(all_chunks)
                st.success(f"Indexed {len(all_chunks)} chunks across {len(uploaded_files)} documents!")

    if "indexed_count" in st.session_state:
        st.metric("Total Indexed Chunks", st.session_state["indexed_count"])

# Initialize Agents in Session State
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
        "Ask a question to your knowledge base:",
        placeholder="e.g. What were the quarterly revenues and key risk factors?"
    )

    if st.button("Submit Query", type="primary") and user_query:
        if not config.GEMINI_API_KEY:
            st.error("Please set your Gemini API Key in the sidebar or in .env")
        else:
            try:
                with st.spinner("Running autonomous agent loop..."):
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
    This module runs quantitative evaluation on your RAG pipeline using the **Ragas** framework.
    It evaluates the core Data Science metrics:
    - **Faithfulness**: Measures factual consistency of the answer with the context (hallucination detector).
    - **Answer Relevance**: Measures how pertinent the generated answer is to the user question.
    - **Context Precision**: Evaluates if the highest-ranking retrieved chunks contain the ground truth.
    - **Context Recall**: Measures if all information needed to answer the question was retrieved.
    """)
    st.info("To run the full evaluation suite, execute: `python -m src.evaluation.run_benchmark`")
