import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src import config

class RagasEvaluator:
    """Runs rigorous quantitative evaluation on RAG responses using Ragas metrics."""

    def __init__(self):
        api_key = config.GEMINI_API_KEY
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=api_key,
            temperature=0.0
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=api_key
        )

    def evaluate_test_set(self, samples: List[Dict[str, Any]], output_path: str | Path = None) -> pd.DataFrame:
        """
        Expects samples with format:
        [
            {
                "question": "What is ...?",
                "contexts": ["Chunk text 1...", "Chunk text 2..."],
                "answer": "Generated answer...",
                "ground_truth": "Expected ground truth answer..."
            }
        ]
        """
        # Convert to Hugging Face Dataset
        data_dict = {
            "question": [s["question"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "ground_truth": [s["ground_truth"] for s in samples],
        }
        dataset = Dataset.from_dict(data_dict)

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        # Run Ragas evaluation with Gemini backend
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        df = result.to_pandas()

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"Evaluation report saved to {out}")

        return df

    @staticmethod
    def generate_summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
        """Calculates mean scores across all evaluated metrics."""
        summary = {}
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in df.columns:
                summary[col] = round(float(df[col].mean()), 4)
        return summary
