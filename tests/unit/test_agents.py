import pytest

from src.intelligence_system.agents.graph import IntelligenceAgentGraph
from src.intelligence_system.models.schemas import (
    RAGSearchOutput,
    SourceCitation,
    SupervisorDecision,
)
from src.intelligence_system.services.llm_client import LLMGenerationResult


class FakeLLM:
    async def generate(self, prompt, system_prompt=None):
        return LLMGenerationResult(
            content="La recuperación híbrida combina búsqueda semántica y léxica.",
            model="fake-model",
            input_tokens=50,
            output_tokens=20,
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
                reason="Se necesita consultar la documentación.",
            )

        return SupervisorDecision(
            next_agent="answer_agent",
            reason="La documentación ya fue recuperada.",
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
                    document_id="manual-001",
                    title="manual.pdf",
                    page=1,
                    excerpt="La recuperación híbrida combina búsqueda semántica y léxica.",
                    score=0.85,
                )
            ]
        )


@pytest.mark.asyncio
async def test_graph_retrieves_documents_and_generates_answer():
    agent_graph = IntelligenceAgentGraph(
        llm_client=FakeLLM(),
        supervisor=FakeSupervisor(),
        agent_tools=FakeAgentTools(),
    )

    graph = agent_graph.build()

    result = await graph.ainvoke({
        "query": "¿Cómo funciona la recuperación híbrida?",
        "conversation_id": "test-conversation-001",
        "metadata_filter": {},
        "requires_approval": False,
    })

    assert result["retrieval_completed"] is True
    assert result["has_retrieved_context"] is True
    assert len(result["sources"]) == 1
    assert result["sources"][0]["title"] == "manual.pdf"
    assert result["answer"]
    assert result["model"] == "fake-model"
    assert result["input_tokens"] == 50
    assert result["output_tokens"] == 20