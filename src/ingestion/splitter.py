import uuid
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    """Splits documents into contextual chunks with persistent unique chunk IDs."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk.metadata.get('source', 'doc')}_p{chunk.metadata.get('page', 1)}_c{i}_{uuid.uuid4().hex[:6]}"
            chunk.metadata["chunk_id"] = chunk_id
        return chunks
