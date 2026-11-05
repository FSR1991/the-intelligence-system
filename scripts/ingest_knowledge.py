from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

# Ubicación raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.intelligence_system.config.settings import get_settings
from src.intelligence_system.rag.chunking import RAGChunkingService
from src.intelligence_system.rag.retrieval import HybridRAGRetriever


SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "entrega_final.md"
)

QUERY = (
    "¿Cuáles son los siete requisitos obligatorios "
    "de la entrega final de The Intelligence System?"
)


async def main() -> None:
    settings = get_settings()

    settings.require_values(
        "pinecone_api_key",
        "pinecone_index_name",
        "pinecone_namespace",
    )

    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            f"No se encontró la fuente documental: {SOURCE_FILE}"
        )

    print("\n=== THE INTELLIGENCE SYSTEM ===")
    print("VALIDACIÓN DE INGESTA RAG")
    
    content = await asyncio.to_thread(
        SOURCE_FILE.read_text,
        encoding="utf-8",
    )

    if not content.strip():
        raise ValueError("El documento de conocimiento está vacío.")

    source = SOURCE_FILE.relative_to(PROJECT_ROOT).as_posix()

    document_id = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    document = Document(
        page_content=content,
        metadata={
            "source": source,
            "file_name": SOURCE_FILE.name,
            "document_id": document_id,
            "content_type": "markdown",
        },
    )

    print(f"\nFuente: {source}")
    print(f"Caracteres: {len(content)}")

    # 1. Fragmentación con el servicio existente.
    chunking_service = RAGChunkingService(
        settings=settings
    )

    chunks = chunking_service.split_documents(
        [document]
    )

    if not chunks:
        raise RuntimeError(
            "La fragmentación no produjo resultados."
        )

    print(f"\nFragmentos generados: {len(chunks)}")

    for number, chunk in enumerate(chunks, start=1):
        print(
            f"Fragmento {number}: "
            f"{len(chunk.page_content)} caracteres"
        )

    # 2. Indexación y recuperación real en Pinecone.
    retriever = HybridRAGRetriever(
        settings=settings
    )

    try:
        print("\nIndexando documentos en Pinecone...")

        stored = await retriever.index_documents(chunks)

        print(f"Fragmentos confirmados por Pinecone: {stored}")

        if stored <= 0:
            raise RuntimeError(
                "Pinecone no confirmó ningún fragmento indexado."
            )

        # 3. Búsqueda híbrida real.
        print("\nEjecutando búsqueda híbrida...")

        results = await retriever.search(
            query=QUERY,
            top_k=settings.rag_top_k,
        )

        print(f"Resultados recuperados: {len(results)}")

        if not results:
            raise RuntimeError(
                "Se confirmó la indexación, pero la búsqueda "
                "no recuperó documentos."
            )

        for number, result in enumerate(results, start=1):
            print(f"\n--- RESULTADO {number} ---")
            print("Fuente:", result.metadata.get("source"))
            print("Chunk:", result.metadata.get("chunk_id"))
            print("RRF:", result.metadata.get("rrf_score"))
            print("Contenido:")
            print(result.page_content[:500])

        print("\n================================")
        print("INGESTA Y RECUPERACIÓN: EXITOSAS")
        print("================================")

    finally:
        try:
            await retriever.close()
        except Exception as exc:
            print(
                "Advertencia al liberar Pinecone:",
                type(exc).__name__,
                str(exc),
            )


if __name__ == "__main__":
    asyncio.run(main())