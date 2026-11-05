import uuid

import pytest
from langgraph.types import Command

from src.intelligence_system.agents.graph import IntelligenceAgentGraph
from src.intelligence_system.models.schemas import (
    RAGSearchOutput,
    SourceCitation,
    SupervisorDecision,
)
from src.intelligence_system.persistence.checkpointer import RedisCheckpointer
from src.intelligence_system.services.llm_client import LLMGenerationResult


class FakeLLM:
    async def generate(self, prompt, system_prompt=None):
        return LLMGenerationResult(
            content="Respuesta generada con evidencia documental.",
            model="fake-model",
            input_tokens=30,
            output_tokens=15,
            finish_reason="stop",
        )


class FakeSupervisor:
    async def decide(
        self,
        query,
        retrieval_completed=False,
        has_retrieved_context=False,
    ):
        if not retrieval_completed:
            return SupervisorDecision(
                next_agent="retrieval_agent",
                reason="Se necesita recuperar documentación.",
            )

        return SupervisorDecision(
            next_agent="answer_agent",
            reason="La recuperación documental ya finalizó.",
        )


class FakeAgentTools:
    async def search_knowledge_base(
        self,
        query,
        top_k=5,
        metadata_filter=None,
    ):
        return RAGSearchOutput(
            results=[
                SourceCitation(
                    document_id="redis-test-document",
                    title="documento-prueba.pdf",
                    page=1,
                    excerpt="Evidencia documental de prueba.",
                    score=0.85,
                )
            ]
        )


def build_test_graph(checkpointer):
    agent_graph = IntelligenceAgentGraph(
        llm_client=FakeLLM(),
        supervisor=FakeSupervisor(),
        agent_tools=FakeAgentTools(),
    )
    return agent_graph.build(checkpointer=checkpointer)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_hitl_resumes_after_reconnecting_to_redis(decision):
    thread_id = "redis-hitl-" + uuid.uuid4().hex
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state = {
        "query": "Procesar una tarea crítica de prueba.",
        "conversation_id": thread_id,
        "metadata_filter": {},
        "requires_approval": True,
    }

    try:
        # Primera conexión: ejecuta el grafo hasta la interrupción.
        async with RedisCheckpointer() as saver:
            graph = build_test_graph(saver)

            paused = await graph.ainvoke(
                initial_state,
                config=config,
            )

            interruptions = paused.get("__interrupt__", ())
            assert len(interruptions) == 1
            assert interruptions[0].value["type"] == "human_approval"

            assert paused["retrieval_completed"] is True
            assert paused["answer"]

        # La conexión se cerró mientras el grafo estaba suspendido.

        # Segunda conexión: recupera y reanuda el mismo thread.
        async with RedisCheckpointer() as saver:
            graph = build_test_graph(saver)

            resumed = await graph.ainvoke(
                Command(
                    resume={
                        "decision": decision,
                        "feedback": "Decisión de prueba persistente.",
                    }
                ),
                config=config,
            )

            assert resumed["approval_decision"] == decision

            if decision == "approve":
                assert resumed["answer"] == (
                    "Respuesta generada con evidencia documental."
                )
            else:
                assert resumed["answer"] == ""

            await saver.adelete_thread(thread_id)

    finally:
        # Limpieza adicional si se produjo algún error.
        async with RedisCheckpointer() as saver:
            await saver.adelete_thread(thread_id)