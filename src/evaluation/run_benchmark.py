import os
from pathlib import Path
from src import config
from src.evaluation.ragas_eval import RagasEvaluator

# Sample benchmark dataset based on real-world QA
SAMPLE_BENCHMARK = [
    {
        "question": "What is the primary role of the Cross-Encoder in the retrieval pipeline?",
        "contexts": [
            "In hybrid search, initial candidates are retrieved using BM25 and dense embeddings. A cross-encoder reranker performs joint full self-attention over the query and candidate passage pairs, scoring non-linear semantic relevance and re-ordering the top results to place the most critical information first."
        ],
        "answer": "The primary role of the Cross-Encoder reranker is to perform full joint cross-attention across the query and candidate passages, accurately scoring semantic relevance and ensuring the highest quality chunks are prioritized for the generator.",
        "ground_truth": "The cross-encoder reranker re-scores and re-ranks candidate passages retrieved by hybrid search by performing full self-attention over query-passage pairs to improve relevance."
    },
    {
        "question": "How does the system handle queries when documents do not contain the answer?",
        "contexts": [
            "When the Document Relevance Grader flags that retrieved passages lack the required facts, the LangGraph agent routes the query to an automated web search fallback tool (such as DuckDuckGo or Tavily) or attempts query rewriting before answer synthesis."
        ],
        "answer": "If internal documents lack the required information, the LangGraph agent evaluates the context as irrelevant and triggers an automated web search fallback tool or rewrites the query.",
        "ground_truth": "The system detects lack of relevant context via a document grader node and triggers a web search fallback or query rewriter."
    }
]

def main():
    print("=" * 60)
    print("Starting Quantitative Ragas Evaluation Benchmark...")
    print("=" * 60)

    if not config.GEMINI_API_KEY:
        print("[Error] GEMINI_API_KEY is not set. Please set it in your .env file.")
        return

    evaluator = RagasEvaluator()
    output_csv = config.BASE_DIR / "data" / "benchmark_results.csv"

    print("Running evaluation across metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall...")
    df = evaluator.evaluate_test_set(SAMPLE_BENCHMARK, output_path=output_csv)

    print("\n--- Benchmark Evaluation Results ---")
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

    summary = evaluator.generate_summary_metrics(df)
    print("\n--- Aggregate Metric Summary ---")
    for metric, score in summary.items():
        print(f"  {metric.capitalize()}: {score:.4f} / 1.0000")

    print(f"\nReport successfully saved to: {output_csv}")

if __name__ == "__main__":
    main()
