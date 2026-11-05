from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..models.schemas import (
    ApprovalDecisionRequest,
    SourceCitation,
    SupervisorDecision,
)
from ..services.llm_client import LLMClient
from .supervisor import Supervisor
from .tools import AgentTools


class AgentGraphError(RuntimeError):
    """Error base del grafo multiagente."""


class AgentGraphConfigurationError(AgentGraphError):
    """Configuración insuficiente para ejecutar el grafo."""


class AgentGraphState(TypedDict, total=False):
    """Estado compartido entre Supervisor y agentes."""

    query: str
    conversation_id: str
    metadata_filter: dict[str, Any]

    next_agent: str
    supervisor_reason: str

    retrieval_completed: bool
    has_retrieved_context: bool
    sources: list[dict[str, Any]]

    answer: str
    model: str
    input_tokens: int
    output_tokens: int

    requires_approval: bool
    approval_decision: str
    approval_feedback: str | None


class IntelligenceAgentGraph:
    """Grafo multiagente con Supervisor, RAG y aprobación humana."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        supervisor: Supervisor | None = None,
        agent_tools: AgentTools | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._supervisor = supervisor or Supervisor(self._llm)
        self._tools = agent_tools or AgentTools()

        self._approval_checkpointer_available = False

    async def _supervisor_node(
        self,
        state: AgentGraphState,
    ) -> dict[str, Any]:
        """Decide qué agente debe intervenir."""

        query = state.get("query", "").strip()

        decision: SupervisorDecision = await self._supervisor.decide(
            query=query,
            retrieval_completed=state.get(
                "retrieval_completed", False
            ),
            has_retrieved_context=state.get(
                "has_retrieved_context", False
            ),
        )

        return {
            "next_agent": decision.next_agent,
            "supervisor_reason": decision.reason,
        }

    async def _retrieval_agent(
        self,
        state: AgentGraphState,
    ) -> dict[str, Any]:
        """Consulta el RAG y guarda las fuentes recuperadas."""

        result = await self._tools.search_knowledge_base(
            query=state["query"],
            top_k=5,
            metadata_filter=state.get("metadata_filter", {}),
        )

        sources = [
            source.model_dump(mode="json")
            for source in result.results
        ]

        return {
            "sources": sources,
            "retrieval_completed": True,
            "has_retrieved_context": bool(sources),
        }

    async def _answer_agent(
        self,
        state: AgentGraphState,
    ) -> dict[str, Any]:
        """Genera una respuesta utilizando la evidencia disponible."""

        sources = [
            SourceCitation.model_validate(source)
            for source in state.get("sources", [])
        ]

        if sources:
            context = "\n\n".join(
                (
                    f"Fuente: {source.title}\n"
                    f"Documento: {source.document_id}\n"
                    f"Página: {source.page or 'No especificada'}\n"
                    f"Contenido: {source.excerpt}"
                )
                for source in sources
            )
        else:
            context = (
                "No se recuperaron fuentes documentales. "
                "No afirmes que la respuesta está respaldada por documentos."
            )

        system_prompt = """
Eres el agente de respuesta de The Intelligence System.

Tu responsabilidad es responder de manera clara, precisa y profesional.

Reglas:
1. Utiliza las fuentes recuperadas cuando estén disponibles.
2. No inventes datos, documentos, citas ni páginas.
3. Distingue la información documental del conocimiento general.
4. Si la evidencia no permite responder, indícalo claramente.
5. Responde en el idioma de la consulta.
6. No inventes que una búsqueda documental fue exitosa si no hay fuentes.
"""

        prompt = (
            f"CONSULTA DEL USUARIO:\n{state['query']}\n\n"
            f"CONTEXTO DOCUMENTAL:\n{context}\n\n"
            "Elabora la respuesta final."
        )

        generation = await self._llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        return {
            "answer": generation.content,
            "model": generation.model,
            "input_tokens": generation.input_tokens or 0,
            "output_tokens": generation.output_tokens or 0,
        }

    async def _approval_node(
        self,
        state: AgentGraphState,
    ) -> dict[str, Any]:
        """Pausa el grafo para solicitar una aprobación humana."""

        if not self._approval_checkpointer_available:
            raise AgentGraphConfigurationError(
                "Las tareas críticas necesitan un checkpointer. "
                "Se configurará con Redis en el paso 7."
            )

        approval = interrupt(
            {
                "type": "human_approval",
                "message": (
                    "Esta tarea requiere aprobación humana "
                    "antes de finalizar."
                ),
                "query": state.get("query", ""),
                "draft_answer": state.get("answer", ""),
                "sources": state.get("sources", []),
            },
            response_schema=ApprovalDecisionRequest,
        )

        if isinstance(approval, ApprovalDecisionRequest):
            decision = approval
        else:
            decision = ApprovalDecisionRequest.model_validate(approval)

        return {
            "approval_decision": decision.decision,
            "approval_feedback": decision.feedback,
            "answer": (
                state.get("answer", "")
                if decision.decision == "approve"
                else ""
            ),
        }

    @staticmethod
    def _route_supervisor(
        state: AgentGraphState,
    ) -> Literal[
        "retrieval_agent",
        "answer_agent",
        "__end__",
    ]:
        next_agent = state.get("next_agent")

        if next_agent in {"retrieval_agent", "answer_agent"}:
            return next_agent

        return END

    @staticmethod
    def _route_after_answer(
        state: AgentGraphState,
    ) -> Literal["approval", "__end__"]:
        if state.get("requires_approval", False):
            return "approval"

        return END

    def build(
        self,
        checkpointer: Any | None = None,
    ) -> Any:
        """Construye y compila el grafo ejecutable de LangGraph."""

        self._approval_checkpointer_available = (
            checkpointer is not None
        )

        builder = StateGraph(AgentGraphState)

        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("retrieval_agent", self._retrieval_agent)
        builder.add_node("answer_agent", self._answer_agent)
        builder.add_node("approval", self._approval_node)

        builder.add_edge(START, "supervisor")

        builder.add_conditional_edges(
            "supervisor",
            self._route_supervisor,
            {
                "retrieval_agent": "retrieval_agent",
                "answer_agent": "answer_agent",
                END: END,
            },
        )

        builder.add_edge("retrieval_agent", "supervisor")

        builder.add_conditional_edges(
            "answer_agent",
            self._route_after_answer,
            {
                "approval": "approval",
                END: END,
            },
        )

        builder.add_edge("approval", END)

        return builder.compile(
            checkpointer=checkpointer,
            name="the-intelligence-system",
        )