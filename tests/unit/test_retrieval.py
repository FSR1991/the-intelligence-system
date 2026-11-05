from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.intelligence_system.config.settings import get_settings
from src.intelligence_system.rag.retrieval import HybridRAGRetriever


class FakeDocumentsAPI:
    def __init__(self):
        self.records = {}

    async def upsert(self, namespace, documents):
        for record in documents:
            self.records[record["_id"]] = dict(record)
        return SimpleNamespace(upserted_count=len(documents))

    async def search(
        self,
        namespace,
        score_by,
        top_k,
        include_fields=None,
        filter=None,
    ):
        clause = score_by[0]
        clause_name = type(clause).__name__
        query = getattr(clause, "query", "").casefold()
        fields = getattr(clause, "fields", None) or ["chunk_text"]

        matches = []

        for record in self.records.values():
            if filter and any(
                record.get(key) != condition.get("$eq")
                if isinstance(condition, dict)
                else record.get(key) != condition
                for key, condition in filter.items()
            ):
                continue

            searchable_text = " ".join(
                str(record.get(field, "")) for field in fields
            ).casefold()

            if query and query not in searchable_text:
                continue

            # Simulación determinista de ambos tipos de búsqueda.
            score = 0.9 if clause_name == "EmbedQuery" else 0.8
            matches.append(
                SimpleNamespace(
                    id=record["_id"],
                    score=score,
                    **{
                        key: value
                        for key, value in record.items()
                        if key != "_id"
                    },
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return SimpleNamespace(matches=matches[:top_k])


class FakeIndex:
    def __init__(self):
        self.documents = FakeDocumentsAPI()

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_index_and_hybrid_search():
    index = FakeIndex()
    retriever = HybridRAGRetriever(
        settings=get_settings(),
        pinecone_client=object(),
        index=index,
    )

    documents = [
        Document(
            page_content="Informe técnico sobre ventas de productos.",
            metadata={
                "chunk_id": "sales-1",
                "source": "ventas.pdf",
                "category": "sales",
            },
        ),
        Document(
            page_content="Documento sobre seguridad de redes.",
            metadata={
                "chunk_id": "security-1",
                "source": "seguridad.pdf",
                "category": "security",
            },
        ),
    ]

    count = await retriever.index_documents(documents)
    results = await retriever.search("ventas", top_k=2)

    assert count == 2
    assert results
    assert results[0].metadata["chunk_id"] == "sales-1"
    assert results[0].metadata["rrf_score"] > 0
    assert "semantic" in results[0].metadata["retrieval_methods"]
    assert "lexical" in results[0].metadata["retrieval_methods"]


@pytest.mark.asyncio
async def test_search_applies_metadata_filter():
    index = FakeIndex()
    retriever = HybridRAGRetriever(
        settings=get_settings(),
        pinecone_client=object(),
        index=index,
    )

    await retriever.index_documents([
        Document(
            page_content="Guía de ventas comerciales.",
            metadata={"chunk_id": "sales-1", "category": "sales"},
        ),
        Document(
            page_content="Guía sobre seguridad de redes.",
            metadata={"chunk_id": "security-1", "category": "security"},
        ),
    ])

    results = await retriever.search(
        "seguridad",
        top_k=5,
        metadata_filter={"category": "security"},
    )

    assert results
    assert all(
        document.metadata["category"] == "security"
        for document in results
    )


@pytest.mark.asyncio
async def test_search_rejects_empty_query():
    retriever = HybridRAGRetriever(
        settings=get_settings(),
        pinecone_client=object(),
        index=FakeIndex(),
    )

    with pytest.raises(ValueError, match="no puede estar vacía"):
        await retriever.search("   ")