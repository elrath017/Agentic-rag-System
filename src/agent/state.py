from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class AgentState(BaseModel):
    """Represents the complete state of the autonomous RAG agent graph."""
    query: str
    rewritten_query: str = ""
    documents: List[Document] = Field(default_factory=list)
    web_documents: List[Document] = Field(default_factory=list)
    generation: str = ""
    sources: List[str] = Field(default_factory=list)
    is_relevant: bool = False
    is_faithful: bool = False
    retry_count: int = 0
    logs: List[str] = Field(default_factory=list)
