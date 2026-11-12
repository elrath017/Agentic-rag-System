from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

class DocumentParser:
    """Parses various file types (.pdf, .txt, .md) into standard LangChain Documents."""

    @staticmethod
    def parse_file(file_path: str | Path) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()
        elif suffix in [".txt", ".md"]:
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: .pdf, .txt, .md")

        # Standardize metadata
        for i, doc in enumerate(docs):
            doc.metadata["source"] = path.name
            doc.metadata["file_path"] = str(path.resolve())
            if "page" not in doc.metadata:
                doc.metadata["page"] = i + 1

        return docs

    @classmethod
    def parse_directory(cls, dir_path: str | Path) -> List[Document]:
        directory = Path(dir_path)
        all_docs: List[Document] = []
        for file_p in directory.glob("**/*"):
            if file_p.suffix.lower() in [".pdf", ".txt", ".md"]:
                all_docs.extend(cls.parse_file(file_p))
        return all_docs
