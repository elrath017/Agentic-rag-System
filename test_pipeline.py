import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

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

def run_test():
    print("=" * 70)
    print("🤖 STARTING END-TO-END AGENTIC RAG PIPELINE TEST")
    print("=" * 70)

    # 1. Check API Key
    if not config.GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY is missing from .env file!")
        return

    print(f"🔑 Gemini API Key configured: {config.GEMINI_API_KEY[:8]}...{config.GEMINI_API_KEY[-4:]}")
    print(f"🧠 Configured LLM Model: {config.LLM_MODEL}")
    print(f"📐 Configured Embedding Model: {config.EMBEDDING_MODEL}")

    # 2. Ingest Sample Document
    sample_file = BASE_DIR / "data" / "sample_docs" / "enterprise_ai_report_2025.txt"
    print(f"\n[Step 1] Ingesting document: {sample_file.name}...")
    raw_docs = DocumentParser.parse_file(sample_file)
    chunker = DocumentChunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.split_documents(raw_docs)
    print(f"✅ Generated {len(chunks)} chunks.")

    # 3. Build Retrievers
    print("\n[Step 2] Initializing Dense ChromaDB & Sparse BM25 Indexes...")
    test_db_dir = BASE_DIR / "chroma_db_test"
    if test_db_dir.exists():
        import shutil
        shutil.rmtree(test_db_dir, ignore_errors=True)
    dense = DenseRetriever(persist_dir=test_db_dir)
    sparse = SparseBM25Retriever()

    dense.add_documents(chunks)
    sparse.index_documents(chunks)
    print("✅ Ingestion & Indexing complete.")

    # 4. Assemble Agent Graph
    print("\n[Step 3] Compiling LangGraph State Machine...")
    hybrid = HybridRetriever(dense, sparse)
    reranker = LLMReranker()
    nodes = RAGNodes(hybrid, reranker)
    agent = create_agentic_rag_graph(nodes)
    print("✅ LangGraph compiled successfully.")

    # 5. Test Case 1: In-Domain Query
    print("\n" + "=" * 50)
    print("TEST CASE 1: In-Domain Factual Query")
    query_1 = "What was NovaCorp's consolidated gross revenue and what percentage did R&D account for?"
    print(f"User Query: '{query_1}'")
    print("=" * 50)

    initial_state_1 = AgentState(query=query_1)
    res_1 = agent.invoke(initial_state_1)

    print("\n--- Execution Trace ---")
    for log in res_1["logs"]:
        print(f"  • {log}")

    print("\n--- Final Answer ---")
    print(res_1["generation"])

    print(f"\n--- Grounding Verification ---")
    print(f"Relevance Passed: {res_1['is_relevant']}")
    print(f"Faithfulness Passed: {res_1['is_faithful']}")
    print(f"Cited Sources: {res_1['sources']}")

    # 6. Test Case 2: Out-of-Domain Query (Triggers Web Search Fallback)
    print("\n" + "=" * 50)
    print("TEST CASE 2: Out-of-Domain Query (Triggering Fallback)")
    query_2 = "What is the mission of NASA's Artemis program?"
    print(f"User Query: '{query_2}'")
    print("=" * 50)

    initial_state_2 = AgentState(query=query_2)
    res_2 = agent.invoke(initial_state_2)

    print("\n--- Execution Trace ---")
    for log in res_2["logs"]:
        print(f"  • {log}")

    print("\n--- Final Answer ---")
    print(res_2["generation"])

    print(f"\n--- Grounding Verification ---")
    print(f"Cited Sources: {res_2['sources']}")
    print("\n✅ All tests executed successfully!")

if __name__ == "__main__":
    run_test()
