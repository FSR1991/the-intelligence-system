from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document
from pinecone import AsyncPinecone, EmbedQuery, TextQuery

from ..config.settings import Settings, get_settings


class RAGRetrievalError(RuntimeError):
    """Error base del sistema de recuperación RAG."""


class RAGConfigurationError(RAGRetrievalError):
    """Falta configuración necesaria para la recuperación."""


class RAGIndexError(RAGRetrievalError):
    """Pinecone no pudo completar una operación."""


class HybridRAGRetriever:
    """Búsqueda semántica y léxica combinada mediante RRF."""

    _BATCH_SIZE = 50

    _RETURN_FIELDS = [
        "chunk_text",
        "title",
        "source",
        "file_name",
        "document_id",
        "page",
        "chunk_id",
        "chunk_index",
        "chunk_size",
    ]

    def __init__(
        self,
        settings: Settings | None = None,
        pinecone_client: Any | None = None,
        index: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._pinecone = pinecone_client
        self._index = index

        self._owns_client = pinecone_client is None
        self._owns_index = index is None

        self._resource_lock = asyncio.Lock()

    async def _get_index(self) -> Any:
        """Inicializa el cliente y obtiene el índice configurado."""

        async with self._resource_lock:
            if self._index is not None:
                return self._index

            api_key = self._settings.pinecone_api_key
            if hasattr(api_key, "get_secret_value"):
                api_key = api_key.get_secret_value()

            index_name = (
                self._settings.pinecone_index_name or ""
            ).strip()

            if not api_key:
                raise RAGConfigurationError(
                    "Falta PINECONE_API_KEY en .env."
                )

            if not index_name:
                raise RAGConfigurationError(
                    "Falta PINECONE_INDEX_NAME en .env."
                )

            try:
                if self._pinecone is None:
                    self._pinecone = AsyncPinecone(
                        api_key=str(api_key)
                    )

                self._index = await self._pinecone.index(
                    name=index_name
                )
                return self._index

            except Exception as exc:
                if self._owns_client and self._pinecone is not None:
                    try:
                        await self._pinecone.close()
                    except Exception:
                        pass

                    self._pinecone = None

                raise RAGIndexError(
                    "No se pudo abrir el índice de Pinecone. "
                    "Verificá su nombre, estado y credenciales."
                ) from exc

    def _namespace(self) -> str:
        namespace = (
            self._settings.pinecone_namespace or ""
        ).strip()

        if not namespace:
            raise RAGConfigurationError(
                "PINECONE_NAMESPACE debe contener un nombre."
            )

        return namespace

    @staticmethod
    def _document_id(document: Document) -> str:
        existing_id = document.metadata.get("chunk_id")
        if existing_id:
            return str(existing_id)

        identity = (
            str(document.metadata.get("document_id", ""))
            + "|"
            + str(document.metadata.get("source", ""))
            + "|"
            + str(document.metadata.get("page", ""))
            + "|"
            + document.page_content
        )

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _plain_metadata(
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Filtra metadatos a tipos admitidos por Pinecone."""

        reserved = {
            "_id",
            "id",
            "chunk_text",
            "title",
            "values",
        }

        result: dict[str, Any] = {}

        for key, value in metadata.items():
            if key in reserved:
                continue

            if isinstance(value, (str, bool, int)):
                result[str(key)] = value

            elif isinstance(value, float):
                if value == value and abs(value) != float("inf"):
                    result[str(key)] = value

            elif (
                isinstance(value, list)
                and all(isinstance(item, str) for item in value)
            ):
                result[str(key)] = value

        return result

    @classmethod
    def _normalize_filter(
        cls,
        filters: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Normaliza filtros simples al formato de Pinecone."""

        if not filters:
            return None

        result: dict[str, Any] = {}

        for field, condition in filters.items():
            if field in {"$and", "$or"}:
                result[field] = [
                    cls._normalize_filter(item) or {}
                    for item in condition
                ]

            elif isinstance(condition, Mapping):
                result[field] = dict(condition)

            else:
                result[field] = {"$eq": condition}

        return result

    @classmethod
    def _filter_field_names(
        cls,
        filters: Mapping[str, Any] | None,
    ) -> list[str]:
        """Obtiene los campos de metadatos utilizados por los filtros."""

        if not filters:
            return []

        fields: list[str] = []

        for field, condition in filters.items():
            if field in {"$and", "$or"}:
                for item in condition:
                    fields.extend(cls._filter_field_names(item))

            elif not field.startswith("$"):
                fields.append(field)

        return list(dict.fromkeys(fields))

    async def index_documents(
        self,
        documents: Sequence[Document],
    ) -> int:
        """Guarda fragmentos; Pinecone genera los embeddings integrados."""

        if not documents:
            return 0

        namespace = self._namespace()
        index = await self._get_index()

        records: list[dict[str, Any]] = []

        for document in documents:
            if not isinstance(document, Document):
                raise TypeError(
                    "Todos los elementos deben ser documentos LangChain."
                )

            text = document.page_content.strip()
            if not text:
                continue

            metadata = self._plain_metadata(document.metadata)
            filename = str(
                document.metadata.get("file_name")
                or document.metadata.get("source")
                or "documento"
            )

            record = {
                "_id": self._document_id(document),
                "chunk_text": text,
                "title": filename,
                **metadata,
            }

            records.append(record)

        if not records:
            return 0

        total_upserted = 0

        for start in range(0, len(records), self._BATCH_SIZE):
            batch = records[start : start + self._BATCH_SIZE]

            try:
                response = await index.documents.upsert(
                    namespace=namespace,
                    documents=batch,
                )
                total_upserted += int(response.upserted_count)

            except Exception as exc:
                raise RAGIndexError(
                    "Pinecone no pudo indexar el lote de documentos."
                ) from exc

        return total_upserted

    @staticmethod
    def _get_field(
        match: Any,
        field: str,
    ) -> Any:
        """Lee campos de documentos devueltos por el SDK."""

        value = getattr(match, field, None)
        if value is not None:
            return value

        to_dict = getattr(match, "to_dict", None)

        if callable(to_dict):
            data = to_dict()

            if isinstance(data, dict):
                fields = data.get("fields", data)

                if isinstance(fields, dict):
                    return fields.get(field)

        return None

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        """Combina búsqueda semántica y léxica mediante RRF."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("La consulta RAG no puede estar vacía.")

        limit = (
            top_k
            if top_k is not None
            else self._settings.rag_top_k
        )

        if limit < 1:
            raise ValueError("top_k debe ser mayor que cero.")

        rrf_k = self._settings.rag_rrf_k
        if rrf_k < 1:
            raise RAGConfigurationError(
                "RAG_RRF_K debe ser mayor que cero."
            )

        namespace = self._namespace()
        index = await self._get_index()
        pinecone_filter = self._normalize_filter(metadata_filter)

        # CAMBIO 1: incluir los campos usados por los filtros
        # para poder conservarlos en los documentos de salida.
        return_fields = list(
            dict.fromkeys([
                *self._RETURN_FIELDS,
                *self._filter_field_names(metadata_filter),
            ])
        )

        candidate_k = max(
            limit,
            self._settings.rag_candidate_k,
        )

        try:
            semantic_task = index.documents.search(
                namespace=namespace,
                score_by=[
                    EmbedQuery(
                        field="chunk_text",
                        query=query.strip(),
                    )
                ],
                top_k=candidate_k,
                # CAMBIO 2: usar return_fields.
                include_fields=return_fields,
                filter=pinecone_filter,
            )

            lexical_task = index.documents.search(
                namespace=namespace,
                score_by=[
                    TextQuery(
                        query=query.strip(),
                        fields=["chunk_text", "title"],
                    )
                ],
                top_k=candidate_k,
                # CAMBIO 2: usar return_fields también aquí.
                include_fields=return_fields,
                filter=pinecone_filter,
            )

            semantic_response, lexical_response = await asyncio.gather(
                semantic_task,
                lexical_task,
            )

        except Exception as exc:
            raise RAGIndexError(
                "Falló la búsqueda semántica o léxica en Pinecone."
            ) from exc

        ranked: dict[str, dict[str, Any]] = {}

        def add_results(
            matches: Sequence[Any],
            method: str,
        ) -> None:
            for rank, match in enumerate(matches, start=1):
                document_id = str(
                    getattr(match, "id", "")
                    or getattr(match, "_id", "")
                )

                text = self._get_field(match, "chunk_text")

                if not document_id or not isinstance(text, str):
                    continue

                text = text.strip()
                if not text:
                    continue

                entry = ranked.setdefault(
                    document_id,
                    {
                        "text": text,
                        "metadata": {},
                        "semantic_score": None,
                        "lexical_score": None,
                        "rrf_score": 0.0,
                        "methods": [],
                    },
                )

                entry["rrf_score"] += 1.0 / (rrf_k + rank)

                score = getattr(match, "score", None)

                if method == "semantic":
                    entry["semantic_score"] = score
                else:
                    entry["lexical_score"] = score

                if method not in entry["methods"]:
                    entry["methods"].append(method)

                # CAMBIO 3: recorrer return_fields, incluyendo
                # los metadatos solicitados por los filtros.
                for field in return_fields:
                    value = self._get_field(match, field)

                    if value is not None and field not in {
                        "chunk_text",
                        "title",
                    }:
                        entry["metadata"][field] = value

                title = self._get_field(match, "title")

                if title is not None:
                    entry["metadata"].setdefault("source", title)

        add_results(semantic_response.matches, "semantic")
        add_results(lexical_response.matches, "lexical")

        ordered = sorted(
            ranked.items(),
            key=lambda item: item[1]["rrf_score"],
            reverse=True,
        )

        results: list[Document] = []

        for document_id, entry in ordered[:limit]:
            metadata = {
                **entry["metadata"],
                "chunk_id": document_id,
                "semantic_score": entry["semantic_score"],
                "lexical_score": entry["lexical_score"],
                "rrf_score": entry["rrf_score"],
                "retrieval_methods": entry["methods"],
            }

            results.append(
                Document(
                    page_content=entry["text"],
                    metadata=metadata,
                )
            )

        return results

    async def close(self) -> None:
        """Libera los recursos creados por este retriever."""

        first_error: Exception | None = None

        if self._owns_index and self._index is not None:
            try:
                await self._index.close()
            except Exception as exc:
                first_error = exc
            finally:
                self._index = None

        if self._owns_client and self._pinecone is not None:
            try:
                await self._pinecone.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            finally:
                self._pinecone = None

        if first_error is not None:
            raise RAGIndexError(
                "No se pudieron liberar todos los recursos de Pinecone."
            ) from first_error