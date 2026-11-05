from __future__ import annotations

import uuid

import pytest

from src.intelligence_system.agents.graph import IntelligenceAgentGraph
from src.intelligence_system.config.settings import get_settings
from src.intelligence_system.models.schemas import (
    ApprovalDecisionRequest,
    RAGSearchOutput,
    SourceCitation,
    SupervisorDecision,
    TaskCreateRequest,
    TaskStatus,
)
from src.intelligence_system.persistence.checkpointer import RedisCheckpointer
from src.intelligence_system.persistence.redis_client import AsyncRedisClient
from src.intelligence_system.services.llm_client import LLMGenerationResult
from src.intelligence_system.services.task_service import TaskService


class FakeLLM:
    async def generate(self, prompt, system_prompt=None):
        return LLMGenerationResult(
            content="Respuesta de prueba basada en documentación.",
            model="integration-test-model",
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
                    document_id="integration-document-001",
                    title="documento-integracion.pdf",
                    page=1,
                    excerpt="Contenido documental de integración.",
                    score=0.90,
                )
            ]
        )


def build_real_graph():
    """Construye LangGraph real con proveedores externos simulados."""

    return IntelligenceAgentGraph(
        llm_client=FakeLLM(),
        supervisor=FakeSupervisor(),
        agent_tools=FakeAgentTools(),
    )


def create_test_settings():
    """Usa colas independientes para no tocar la cola del proyecto."""

    return get_settings().model_copy(
        update={
            "redis_task_queue_key": (
                "intelligence:integration-test:"
                + uuid.uuid4().hex
                + ":pending"
            ),
            "task_ttl_seconds": 300,
            "worker_max_concurrency": 1,
        }
    )


async def create_environment():
    settings = create_test_settings()
    redis = AsyncRedisClient(settings=settings)
    await redis.connect()

    service = TaskService(
        settings=settings,
        redis_client=redis,
        graph_factory=build_real_graph,
    )

    return settings, redis, service


async def cleanup_environment(
    settings,
    redis,
    task_ids,
    thread_ids,
):
    try:
        for task_id in task_ids:
            await redis.delete(
                f"intelligence:tasks:{task_id}"
            )

        await redis.delete(settings.redis_task_queue_key)
        await redis.delete(
            f"{settings.redis_task_queue_key}:processing"
        )

        if thread_ids:
            async with RedisCheckpointer(
                settings=settings
            ) as saver:
                for thread_id in thread_ids:
                    await saver.adelete_thread(thread_id)

    finally:
        await redis.close()


@pytest.mark.asyncio
async def test_task_service_executes_graph_and_persists_result():
    settings, redis, service = await create_environment()
    task_ids = []
    thread_ids = []

    try:
        accepted = await service.submit(
            TaskCreateRequest(
                query="Analizar documentación de integración."
            )
        )
        task_ids.append(accepted.task_id)

        initial = await service.get_status(accepted.task_id)
        thread_ids.append(initial.conversation_id)

        assert initial.status == TaskStatus.QUEUED

        processed = await service.process_one(timeout=2)
        assert processed is True

        final = await service.get_status(accepted.task_id)

        assert final.status == TaskStatus.DONE
        assert final.result is not None
        assert final.result.answer == (
            "Respuesta de prueba basada en documentación."
        )
        assert len(final.result.sources) == 1

    finally:
        await cleanup_environment(
            settings, redis, task_ids, thread_ids
        )


@pytest.mark.asyncio
async def test_task_service_resumes_approval_after_reconnection():
    settings, redis, service = await create_environment()
    task_ids = []
    thread_ids = []

    try:
        accepted = await service.submit(
            TaskCreateRequest(
                query="Procesar tarea crítica de aprobación."
            ),
            requires_approval=True,
        )
        task_ids.append(accepted.task_id)

        processed = await service.process_one(timeout=2)
        assert processed is True

        paused = await service.get_status(accepted.task_id)

        thread_ids.append(paused.conversation_id)

        assert paused.status == TaskStatus.WAITING_APPROVAL

        # Nuevo objeto de servicio: simula la recuperación
        # después de cerrar y volver a abrir la aplicación.
        restarted_service = TaskService(
            settings=settings,
            redis_client=redis,
            graph_factory=build_real_graph,
        )

        resumed = await restarted_service.submit_approval(
            accepted.task_id,
            ApprovalDecisionRequest(
                decision="approve",
                feedback="Aprobación de integración.",
            ),
        )

        assert resumed.status == TaskStatus.DONE
        assert resumed.result is not None

    finally:
        await cleanup_environment(
            settings, redis, task_ids, thread_ids
        )


@pytest.mark.asyncio
async def test_task_service_rejects_critical_task_after_reconnection():
    settings, redis, service = await create_environment()
    task_ids = []
    thread_ids = []

    try:
        accepted = await service.submit(
            TaskCreateRequest(
                query="Procesar tarea crítica que será rechazada."
            ),
            requires_approval=True,
        )
        task_ids.append(accepted.task_id)

        processed = await service.process_one(timeout=2)
        assert processed is True

        paused = await service.get_status(accepted.task_id)
        thread_ids.append(paused.conversation_id)

        assert paused.status == TaskStatus.WAITING_APPROVAL

        restarted_service = TaskService(
            settings=settings,
            redis_client=redis,
            graph_factory=build_real_graph,
        )

        rejected = await restarted_service.submit_approval(
            accepted.task_id,
            ApprovalDecisionRequest(
                decision="reject",
                feedback="Rechazado en la prueba de integración.",
            ),
        )

        assert rejected.status == TaskStatus.REJECTED
        assert rejected.result is None

    finally:
        await cleanup_environment(
            settings, redis, task_ids, thread_ids
        )


@pytest.mark.asyncio
async def test_task_service_recovers_claimed_task_after_restart():
    settings, redis, service = await create_environment()
    task_ids = []
    thread_ids = []

    try:
        accepted = await service.submit(
            TaskCreateRequest(
                query="Recuperar tarea tras interrupción del trabajador."
            )
        )
        task_ids.append(accepted.task_id)

        initial = await service.get_status(accepted.task_id)
        thread_ids.append(initial.conversation_id)

        processing_key = (
            f"{settings.redis_task_queue_key}:processing"
        )

        # Simula que el trabajador retiró la tarea
        # pero se interrumpió antes de ejecutarla.
        claimed = await redis.claim_json(
            settings.redis_task_queue_key,
            processing_key,
            timeout=2,
        )

        assert claimed == {"task_id": accepted.task_id}

        # Nuevo servicio: simula un reinicio del trabajador.
        restarted_service = TaskService(
            settings=settings,
            redis_client=redis,
            graph_factory=build_real_graph,
        )

        recovered_count = (
            await restarted_service.recover_processing()
        )

        assert recovered_count == 1

        processed = await restarted_service.process_one(
            timeout=2
        )
        assert processed is True

        final = await restarted_service.get_status(
            accepted.task_id
        )

        assert final.status == TaskStatus.DONE
        assert final.result is not None

    finally:
        await cleanup_environment(
            settings, redis, task_ids, thread_ids
        )