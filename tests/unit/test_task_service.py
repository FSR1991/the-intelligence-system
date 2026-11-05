from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.intelligence_system.config.settings import get_settings
from src.intelligence_system.models.schemas import (
    TaskCreateRequest,
    TaskStatus,
)
from src.intelligence_system.services.task_service import (
    TaskNotFoundError,
    TaskService,
)


class FakeRedisClient:
    """Redis simulado para pruebas unitarias del servicio."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.queue: list[dict[str, Any]] = []
        self.fail_enqueue = False

    async def connect(self) -> None:
        return None

    async def set_json(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
    ) -> bool:
        self.data[key] = deepcopy(value)
        return True

    async def get_json(self, key: str) -> Any | None:
        value = self.data.get(key)
        return deepcopy(value) if value is not None else None

    async def delete(self, key: str) -> int:
        return int(self.data.pop(key, None) is not None)

    async def enqueue_json(
        self,
        key: str,
        value: Any,
    ) -> int:
        if self.fail_enqueue:
            raise RuntimeError("Fallo simulado de la cola.")

        self.queue.append({
            "queue_key": key,
            "payload": deepcopy(value),
        })
        return len(self.queue)

    async def enqueue_task_atomically(
        self,
        record_key: str,
        record: Any,
        queue_key: str,
        payload: Any,
        ex: int,
    ) -> int:
        """Simula la escritura atómica de registro y cola."""

        if self.fail_enqueue:
            raise RuntimeError("Fallo simulado de la transacción.")

        if ex < 1:
            raise ValueError("TTL inválido.")

        # La simulación aplica ambos cambios solamente si
        # la operación puede completarse.
        stored_record = deepcopy(record)
        queued_payload = deepcopy(payload)

        self.data[record_key] = stored_record
        self.queue.append({
            "queue_key": queue_key,
            "payload": queued_payload,
        })

        return len(self.queue)


@pytest.fixture
def redis_client():
    return FakeRedisClient()


@pytest.fixture
def task_service(redis_client):
    return TaskService(
        settings=get_settings(),
        redis_client=redis_client,
    )


@pytest.mark.asyncio
async def test_submit_persists_and_enqueues_task(
    task_service,
    redis_client,
):
    request = TaskCreateRequest(
        query="Analizar documentación del proyecto.",
        conversation_id="conversation-unit-001",
    )

    accepted = await task_service.submit(request)

    assert accepted.task_id
    assert accepted.status == TaskStatus.QUEUED

    record_key = f"intelligence:tasks:{accepted.task_id}"
    record = await redis_client.get_json(record_key)

    assert record is not None
    assert record["query"] == request.query
    assert record["conversation_id"] == "conversation-unit-001"
    assert record["status"] == TaskStatus.QUEUED.value

    assert len(redis_client.queue) == 1
    assert redis_client.queue[0]["payload"]["task_id"] == (
        accepted.task_id
    )


@pytest.mark.asyncio
async def test_submit_generates_conversation_id_when_missing(
    task_service,
    redis_client,
):
    request = TaskCreateRequest(
        query="Consulta sin conversación previa."
    )

    accepted = await task_service.submit(request)

    record = await redis_client.get_json(
        f"intelligence:tasks:{accepted.task_id}"
    )

    assert record is not None
    assert record["conversation_id"]
    assert record["checkpoint_thread_id"] == (
        record["conversation_id"]
    )


@pytest.mark.asyncio
async def test_submit_does_not_leave_record_if_transaction_fails(
    task_service,
    redis_client,
):
    redis_client.fail_enqueue = True

    request = TaskCreateRequest(
        query="Esta tarea no debe quedar huérfana."
    )

    with pytest.raises(RuntimeError):
        await task_service.submit(request)

    assert not any(
        key.startswith("intelligence:tasks:")
        for key in redis_client.data
    )
    assert redis_client.queue == []


@pytest.mark.asyncio
async def test_get_status_returns_persisted_task(
    task_service,
):
    accepted = await task_service.submit(
        TaskCreateRequest(query="Consultar estado.")
    )

    status = await task_service.get_status(
        accepted.task_id
    )

    assert status.task_id == accepted.task_id
    assert status.status == TaskStatus.QUEUED
    assert status.conversation_id


@pytest.mark.asyncio
async def test_get_status_raises_for_unknown_task(
    task_service,
):
    with pytest.raises(TaskNotFoundError):
        await task_service.get_status("task-inexistente")


@pytest.mark.asyncio
async def test_critical_task_is_marked_for_approval(
    task_service,
    redis_client,
):
    accepted = await task_service.submit(
        TaskCreateRequest(
            query="Ejecutar operación crítica."
        ),
        requires_approval=True,
    )

    record = await redis_client.get_json(
        f"intelligence:tasks:{accepted.task_id}"
    )

    assert record is not None
    assert record["requires_approval"] is True
    assert record["status"] == TaskStatus.QUEUED.value