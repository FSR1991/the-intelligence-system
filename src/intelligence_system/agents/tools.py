from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from ..models.schemas import (
    RAGSearchInput,
    RAGSearchOutput,
    SourceCitation,
)
from ..rag.retrieval import HybridRAGRetriever


class AgentToolError(RuntimeError):
    """Error controlado durante la ejecución de herramientas."""


class AgentTools:
    """Herramientas asíncronas disponibles para los agentes."""

    def __init__(
        self,
        retriever: HybridRAGRetriever | None = None,
    ) -> None:
        self._retriever = retriever or HybridRAGRetriever()

    async def search_knowledge_base(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RAGSearchOutput:
        """Busca información documental y devuelve fuentes verificables."""

        search_input = RAGSearchInput(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter or {},
        )

        try:
            documents = await self._retriever.search(
                query=search_input.query,
                top_k=search_input.top_k,
                metadata_filter=search_input.metadata_filter,
            )

            citations = []

            for document in documents:
                metadata = document.metadata

                source = str(
                    metadata.get("source")
                    or metadata.get("file_name")
                    or metadata.get("title")
                    or "Fuente documental"
                )

                document_id = str(
                    metadata.get("document_id")
                    or metadata.get("chunk_id")
                    or source
                )

                page = metadata.get("page")
                if not isinstance(page, int) or isinstance(page, bool):
                    page = None

                score = metadata.get("rrf_score")

                citations.append(
                    SourceCitation(
                        document_id=document_id,
                        title=source,
                        page=page,
                        excerpt=document.page_content[:3000],
                        score=float(score) if score is not None else None,
                    )
                )

            return RAGSearchOutput(results=citations)

        except Exception as exc:
            raise AgentToolError(
                "No se pudo completar la búsqueda documental."
            ) from exc

    def as_tools(self) -> list[StructuredTool]:
        """Expone las herramientas con contratos Pydantic."""

        async def search_knowledge_base_tool(
            query: str,
            top_k: int = 5,
            metadata_filter: dict[str, Any] | None = None,
        ) -> str:
            result = await self.search_knowledge_base(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            return result.model_dump_json()

        search_tool = StructuredTool.from_function(
            coroutine=search_knowledge_base_tool,
            name="search_knowledge_base",
            description=(
                "Busca información en la base documental del sistema "
                "mediante recuperación semántica y léxica. "
                "Utiliza esta herramienta cuando necesites evidencia "
                "documental para responder una consulta."
            ),
            args_schema=RAGSearchInput,
            infer_schema=False,
        )

        return [search_tool]