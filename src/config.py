import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Load .env file
load_dotenv(BASE_DIR / ".env")

# API Keys & LLM Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Recommended Gemini model identifiers
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Search & Retrieval Parameters
TOP_K_RETRIEVAL = 10
TOP_K_RERANKED = 4
RRF_K_CONSTANT = 60
