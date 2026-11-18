import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Load .env file
load_dotenv(BASE_DIR / ".env")

def get_config_var(key: str, default: str = "") -> str:
    """Retrieves config from environment variables, falling back to Streamlit secrets if present."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

# API Keys & LLM Settings
GEMINI_API_KEY = get_config_var("GEMINI_API_KEY", "")
TAVILY_API_KEY = get_config_var("TAVILY_API_KEY", "")

# Recommended Gemini model identifiers
LLM_MODEL = get_config_var("LLM_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = get_config_var("EMBEDDING_MODEL", "models/text-embedding-004")

# Search & Retrieval Parameters
TOP_K_RETRIEVAL = 10
TOP_K_RERANKED = 4
RRF_K_CONSTANT = 60
