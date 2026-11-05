import pytest
from langchain_core.documents import Document

import src.intelligence_system.rag.ingestion as ingestion_module
from src.intelligence_system.rag.chunking import RAGChunkingService
from src.intelligence_system.rag.ingestion import (
    DocumentIngestionError,
    PDFIngestionService,
)


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakePdfReader:
    def __init__(self, stream, strict=False):
        self.is_encrypted = False
        self.pages = [
            FakePage("Primera página del documento de prueba."),
            FakePage("Segunda página con contenido adicional."),
        ]


@pytest.mark.asyncio
async def test_pdf_ingestion_preserves_page_metadata(monkeypatch):
    monkeypatch.setattr(ingestion_module, "PdfReader", FakePdfReader)
    service = PDFIngestionService()

    documents = await service.extract_pdf(
        b"contenido PDF simulado",
        "manual.pdf",
        {"category": "guide"},
    )

    assert len(documents) == 2
    assert documents[0].metadata["source"] == "manual.pdf"
    assert documents[0].metadata["page"] == 1
    assert documents[1].metadata["page"] == 2
    assert documents[0].metadata["category"] == "guide"
    assert documents[0].metadata["document_id"] == documents[1].metadata[
        "document_id"
    ]


@pytest.mark.asyncio
async def test_pdf_ingestion_rejects_wrong_extension():
    service = PDFIngestionService()

    with pytest.raises(DocumentIngestionError):
        await service.extract_pdf(b"contenido", "archivo.txt")


@pytest.mark.asyncio
async def test_pdf_ingestion_rejects_empty_content():
    service = PDFIngestionService()

    with pytest.raises(DocumentIngestionError):
        await service.extract_pdf(b"", "vacio.pdf")


def test_chunking_preserves_metadata_and_ids():
    service = RAGChunkingService()
    source = Document(
        page_content="Contenido técnico para recuperar información. " * 100,
        metadata={
            "source": "manual.pdf",
            "document_id": "manual-001",
            "page": 1,
        },
    )

    chunks = service.split_documents([source])

    assert len(chunks) > 1
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["source"] == "manual.pdf" for chunk in chunks)
    assert all(chunk.metadata["document_id"] == "manual-001" for chunk in chunks)
    assert all(chunk.metadata["chunk_id"] for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)