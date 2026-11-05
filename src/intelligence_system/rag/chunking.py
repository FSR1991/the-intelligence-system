from __future__ import annotations

import hashlib
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config.settings import Settings, get_settings


class RAGChunkingError(RuntimeError):
    """Error controlado durante la fragmentación del contenido."""


class RAGChunkingService:
    """Fragmenta documentos y conserva los metadatos de sus fuentes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _build_splitter(self) -> RecursiveCharacterTextSplitter:
        chunk_size = self._settings.rag_chunk_size
        chunk_overlap = self._settings.rag_chunk_overlap

        if chunk_size < 1:
            raise RAGChunkingError(
                "RAG_CHUNK_SIZE debe ser mayor que cero."
            )

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise RAGChunkingError(
                "RAG_CHUNK_OVERLAP debe ser mayor o igual a cero "
                "y menor que RAG_CHUNK_SIZE."
            )

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            strip_whitespace=True,
        )

    @staticmethod
    def _create_chunk_id(
        document: Document,
        chunk_index: int,
    ) -> str:
        """Genera un identificador reproducible para cada fragmento."""

        metadata = document.metadata
        document_id = str(
            metadata.get("document_id")
            or metadata.get("source")
            or "unknown-source"
        )
        page = str(metadata.get("page", "unknown-page"))
        start_index = str(metadata.get("start_index", "unknown-offset"))

        identity = (
            f"{document_id}|{page}|{start_index}|"
            f"{chunk_index}|{document.page_content}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def split_documents(
        self,
        documents: Sequence[Document],
    ) -> list[Document]:
        """Divide documentos y añade identificadores a los fragmentos."""

        if not documents:
            return []

        if any(not isinstance(doc, Document) for doc in documents):
            raise RAGChunkingError(
                "Todos los elementos deben ser documentos LangChain."
            )

        splitter = self._build_splitter()
        chunks: list[Document] = []

        for document in documents:
            if not document.page_content.strip():
                continue

            split_documents = splitter.split_documents([document])

            for chunk_index, chunk in enumerate(split_documents):
                chunk_id = self._create_chunk_id(chunk, chunk_index)

                chunk.metadata = {
                    **chunk.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk.page_content),
                }
                chunks.append(chunk)

        return chunks