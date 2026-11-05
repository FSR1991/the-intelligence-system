import pytest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.intelligence_system.agents.graph import (
    AgentGraphConfigurationError,
    IntelligenceAgentGraph,
)
from src.intelligence_system.models.schemas import (
    RAGSearchOutput,
    SourceCitation,
    SupervisorDecision,
)
from src.intelligence_system.services.llm_client import LLMGenerationResult


class FakeLLM:
    async def generate(self, prompt, system_prompt=None):
        return LLMGenerationResult(
            content="Respuesta de prueba basada en documentos.",
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
                reason="Se necesita consultar documentos.",
            )

        return SupervisorDecision(
            next_agent="answer_agent",
            reason="La recuperación ya fue realizada.",
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
                    document_id="test-001",
                    title="documento-prueba.pdf",
                    page=1,
                    excerpt="Contenido documental de prueba.",
                    score=0.85,
                )
            ]
        )


def build_graph(checkpointer=None):
    agent_graph = IntelligenceAgentGraph(
        llm_client=FakeLLM(),
        supervisor=FakeSupervisor(),
        agent_tools=FakeAgentTools(),
    )
    return agent_graph.build(checkpointer=checkpointer)


def initial_state():
    return {
        "query": "Procesar una tarea crítica de prueba.",
        "conversation_id": "hitl-test",
        "metadata_filter": {},
        "requires_approval": True,
    }


@pytest.mark.asyncio
async def test_human_approval_resumes_graph():
    graph = build_graph(InMemorySaver())
    config = {
        "configurable": {
            "thread_id": "hitl-approval-test",
        }
    }

    paused = await graph.ainvoke(initial_state(), config=config)

    interruptions = paused.get("__interrupt__", ())
    assert len(interruptions) == 1

    interrupt_info = interruptions[0]
    assert interrupt_info.value["type"] == "human_approval"
    assert interrupt_info.response_schema is not None

    resumed = await graph.ainvoke(
        Command(
            resume={
                "decision": "approve",
                "feedback": "Aprobado para continuar.",
            }
        ),
        config=config,
    )

    assert resumed["approval_decision"] == "approve"
    assert resumed["answer"] == "Respuesta de prueba basada en documentos."


@pytest.mark.asyncio
async def test_human_rejection_resumes_graph():
    graph = build_graph(InMemorySaver())
    config = {
        "configurable": {
            "thread_id": "hitl-rejection-test",
        }
    }

    await graph.ainvoke(initial_state(), config=config)

    resumed = await graph.ainvoke(
        Command(
            resume={
                "decision": "reject",
                "feedback": "La respuesta necesita correcciones.",
            }
        ),
        config=config,
    )

    assert resumed["approval_decision"] == "reject"
    assert resumed["answer"] == ""


@pytest.mark.asyncio
async def test_critical_task_requires_checkpointer():
    graph = build_graph()

    with pytest.raises(AgentGraphConfigurationError):
        await graph.ainvoke(initial_state())