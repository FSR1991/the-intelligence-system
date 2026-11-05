from __future__ import annotations

import asyncio
import logging
import time
import uuid

from datetime import datetime, timezone
from typing import Any

from langgraph.types import Command

from ..agents.graph import IntelligenceAgentGraph
from ..config.settings import Settings, get_settings
from ..models.schemas import (
    ApprovalDecisionRequest,
    TaskAcceptedResponse,
    TaskCreateRequest,
    TaskResult,
    TaskStatus,
    TaskStatusResponse,
)
from ..observability.costs import (
    estimate_cost_usd,
    track_token_usage,
)
from ..observability.tracing import trace_task
from ..persistence.checkpointer import RedisCheckpointer
from ..persistence.redis_client import AsyncRedisClient


logger = logging.getLogger(__name__)


class TaskServiceError(RuntimeError):
    """Error base del servicio de tareas."""


class TaskNotFoundError(TaskServiceError):
    """La tarea solicitada no existe."""


class TaskStateError(TaskServiceError):
    """La tarea no se encuentra en un estado compatible."""


class TaskService:
    """Gestiona tareas, persistencia, cola y ejecución multiagente."""

    def __init__(
        self,
        settings: Settings | None = None,
        redis_client: AsyncRedisClient | None = None,
        graph_factory: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._redis = redis_client or AsyncRedisClient(
            settings=self._settings
        )
        self._owns_redis = redis_client is None

        self._graph_factory = (
            graph_factory or IntelligenceAgentGraph
        )

        self._queue_key = self._settings.redis_task_queue_key
        self._processing_key = f"{self._queue_key}:processing"
        self._ttl = self._settings.task_ttl_seconds

        self._max_concurrency = max(
            1,
            self._settings.worker_max_concurrency,
        )

        self._stop_event = asyncio.Event()
        self._worker_running = False

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _task_key(task_id: str) -> str:
        return f"intelligence:tasks:{task_id}"

    async def _load_record(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        record = await self._redis.get_json(
            self._task_key(task_id)
        )

        if record is None:
            raise TaskNotFoundError(
                f"No existe la tarea '{task_id}'."
            )

        if not isinstance(record, dict):
            raise TaskServiceError(
                "El registro almacenado de la tarea no es válido."
            )

        return record

    async def _save_record(
        self,
        task_id: str,
        record: dict[str, Any],
    ) -> None:
        record["updated_at"] = self._now()

        saved = await self._redis.set_json(
            self._task_key(task_id),
            record,
            ex=self._ttl,
        )

        if not saved:
            raise TaskServiceError(
                "No se pudo persistir el estado de la tarea."
            )

    async def _update_record(
        self,
        task_id: str,
        **changes: Any,
    ) -> dict[str, Any]:
        record = await self._load_record(task_id)
        record.update(changes)
        await self._save_record(task_id, record)
        return record

    @staticmethod
    def _public_response(
        record: dict[str, Any],
    ) -> TaskStatusResponse:
        return TaskStatusResponse.model_validate({
            "task_id": record["task_id"],
            "status": record["status"],
            "conversation_id": record.get("conversation_id"),
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "result": record.get("result"),
            "error": record.get("error"),
        })

    async def submit(
        self,
        request: TaskCreateRequest,
        requires_approval: bool = False,
    ) -> TaskAcceptedResponse:
        """Crea y encola una tarea en una transacción Redis."""

        request = TaskCreateRequest.model_validate(
            request.model_dump()
            if isinstance(request, TaskCreateRequest)
            else request
        )

        task_id = str(uuid.uuid4())
        conversation_id = (
            request.conversation_id or str(uuid.uuid4())
        )

        now = self._now()

        record = {
            "task_id": task_id,
            "status": TaskStatus.QUEUED.value,
            "conversation_id": conversation_id,
            "checkpoint_thread_id": conversation_id,
            "created_at": now,
            "updated_at": now,
            "query": request.query,
            "metadata_filter": request.metadata_filter,
            "requires_approval": bool(requires_approval),
            "result": None,
            "error": None,
            "draft_answer": None,
            "approval_feedback": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": None,
            "token_usage": {},
            "elapsed_ms": None,
        }

        await self._redis.enqueue_task_atomically(
            record_key=self._task_key(task_id),
            record=record,
            queue_key=self._queue_key,
            payload={"task_id": task_id},
            ex=self._ttl,
        )

        return TaskAcceptedResponse(
            task_id=task_id,
            status=TaskStatus.QUEUED,
        )

    async def get_status(
        self,
        task_id: str,
    ) -> TaskStatusResponse:
        """Recupera el estado persistido de una tarea."""

        return self._public_response(
            await self._load_record(task_id)
        )

    async def _execute_task(
        self,
        task_id: str,
    ) -> None:
        """Ejecuta el grafo y conserva resultado y observabilidad."""

        record = await self._load_record(task_id)

        terminal_states = {
            TaskStatus.DONE.value,
            TaskStatus.FAILED.value,
            TaskStatus.REJECTED.value,
            TaskStatus.WAITING_APPROVAL.value,
        }

        if record["status"] in terminal_states:
            return

        await self._update_record(
            task_id,
            status=TaskStatus.RUNNING.value,
            error=None,
        )

        started = time.perf_counter()

        conversation_id = record["conversation_id"]
        thread_id = record["checkpoint_thread_id"]
        requires_approval = record.get(
            "requires_approval",
            False,
        )

        state = {
            "query": record["query"],
            "conversation_id": conversation_id,
            "metadata_filter": record.get(
                "metadata_filter",
                {},
            ),
            "requires_approval": requires_approval,
            "retrieval_completed": False,
            "has_retrieved_context": False,
            "sources": [],
            "answer": "",
        }

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Acumulador aislado para todas las llamadas LLM
        # efectuadas dentro de esta ejecución.
        usage = None

        try:
            with track_token_usage() as usage:
                # Agrupa la ejecución de LangGraph en LangSmith.
                with trace_task(
                    settings=self._settings,
                    task_id=task_id,
                    conversation_id=conversation_id,
                    operation="task_execution",
                    requires_approval=requires_approval,
                ):
                    async with RedisCheckpointer(
                        settings=self._settings
                    ) as saver:
                        agent_graph = self._graph_factory()
                        graph = agent_graph.build(
                            checkpointer=saver
                        )

                        output = await graph.ainvoke(
                            state,
                            config=config,
                        )

        except Exception:
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000

            # Conserva el consumo parcial aunque la ejecución falle.
            if usage is not None:
                try:
                    await self._update_record(
                        task_id,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        estimated_cost_usd=estimate_cost_usd(
                            usage
                        ),
                        token_usage=usage.to_dict(),
                        elapsed_ms=elapsed_ms,
                    )
                except Exception:
                    logger.exception(
                        "No se pudieron guardar las métricas "
                        "de la tarea fallida %s.",
                        task_id,
                    )

            raise

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        estimated_cost = estimate_cost_usd(usage)

        metrics = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost,
            "token_usage": usage.to_dict(),
            "elapsed_ms": elapsed_ms,
        }

        # Una interrupción HITL conserva las métricas originales
        # para recuperarlas cuando llegue la decisión humana.
        if output.get("__interrupt__"):
            await self._update_record(
                task_id,
                status=TaskStatus.WAITING_APPROVAL.value,
                draft_answer=output.get("answer", ""),
                **metrics,
            )
            return

        if output.get("approval_decision") == "reject":
            await self._update_record(
                task_id,
                status=TaskStatus.REJECTED.value,
                result=None,
                approval_feedback=output.get(
                    "approval_feedback"
                ),
                **metrics,
            )
            return

        result = TaskResult(
            answer=output.get("answer", ""),
            sources=output.get("sources", []),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
            elapsed_ms=elapsed_ms,
        )

        await self._update_record(
            task_id,
            status=TaskStatus.DONE.value,
            result=result.model_dump(mode="json"),
            error=None,
            draft_answer=None,
            **metrics,
        )

    async def _process_claimed(
        self,
        payload: dict[str, Any],
    ) -> None:
        """Ejecuta y confirma una tarea reclamada."""

        task_id = str(payload.get("task_id", ""))

        if not task_id:
            raise TaskServiceError(
                "El elemento de la cola no contiene task_id."
            )

        try:
            await self._execute_task(task_id)

        except Exception as exc:
            logger.exception(
                "Falló la ejecución de la tarea %s.",
                task_id,
            )

            # Si tampoco se puede persistir el error, no se confirma
            # la tarea: permanecerá recuperable en processing.
            await self._update_record(
                task_id,
                status=TaskStatus.FAILED.value,
                error=str(exc)[:2000],
            )

        acknowledged = await self._redis.ack_json(
            self._processing_key,
            payload,
        )

        if acknowledged != 1:
            raise TaskServiceError(
                "La tarea no pudo confirmarse en la cola."
            )

    async def process_one(
        self,
        timeout: float = 1.0,
    ) -> bool:
        """Procesa una tarea, útil para ejecución controlada."""

        await self._redis.connect()

        payload = await self._redis.claim_json(
            self._queue_key,
            self._processing_key,
            timeout=timeout,
        )

        if payload is None:
            return False

        await self._process_claimed(payload)
        return True

    async def recover_processing(self) -> int:
        """Recupera tareas no confirmadas tras una interrupción."""

        await self._redis.connect()

        recovered = await self._redis.recover_processing_json(
            self._processing_key,
            self._queue_key,
        )

        logger.info(
            "Se recuperaron %s tareas pendientes.",
            len(recovered),
        )

        return len(recovered)

    async def run_worker(self) -> None:
        """Ejecuta el trabajador con concurrencia configurable.

        Diseñado para una única instancia del worker. La recuperación
        inicial debe realizarse antes de aceptar nuevas reclamaciones.
        """

        if self._worker_running:
            raise TaskServiceError(
                "El trabajador ya está en ejecución."
            )

        self._worker_running = True
        self._stop_event.clear()

        active: set[asyncio.Task[Any]] = set()

        try:
            await self.recover_processing()

            logger.info(
                "Trabajador iniciado. Concurrencia máxima: %s.",
                self._max_concurrency,
            )

            while not self._stop_event.is_set():
                completed = {
                    task for task in active if task.done()
                }
                active -= completed

                for task in completed:
                    try:
                        task.result()
                    except Exception:
                        logger.exception(
                            "Error no controlado en un trabajador."
                        )

                if len(active) >= self._max_concurrency:
                    await asyncio.wait(
                        active,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    continue

                payload = await self._redis.claim_json(
                    self._queue_key,
                    self._processing_key,
                    timeout=1,
                )

                if payload is None:
                    continue

                active.add(
                    asyncio.create_task(
                        self._process_claimed(payload)
                    )
                )

        finally:
            if active:
                await asyncio.gather(
                    *active,
                    return_exceptions=True,
                )

            self._worker_running = False
            logger.info("Trabajador detenido.")

    async def stop_worker(self) -> None:
        """Solicita una detención controlada."""

        self._stop_event.set()

    async def submit_approval(
        self,
        task_id: str,
        decision: ApprovalDecisionRequest,
    ) -> TaskStatusResponse:
        """Reanuda una tarea suspendida con decisión humana."""

        decision = ApprovalDecisionRequest.model_validate(
            decision.model_dump()
            if isinstance(decision, ApprovalDecisionRequest)
            else decision
        )

        record = await self._load_record(task_id)

        if record["status"] != TaskStatus.WAITING_APPROVAL.value:
            raise TaskStateError(
                "La tarea no está esperando una aprobación humana."
            )

        thread_id = record["checkpoint_thread_id"]

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        approval_started = time.perf_counter()

        # La aprobación se registra como una operación independiente,
        # asociada a la tarea y conversación originales.
        with trace_task(
            settings=self._settings,
            task_id=task_id,
            conversation_id=record.get("conversation_id"),
            operation="human_approval_resume",
            requires_approval=True,
        ):
            async with RedisCheckpointer(
                settings=self._settings
            ) as saver:
                agent_graph = self._graph_factory()
                graph = agent_graph.build(
                    checkpointer=saver
                )

                output = await graph.ainvoke(
                    Command(
                        resume=decision.model_dump()
                    ),
                    config=config,
                )

        approval_elapsed_ms = (
            time.perf_counter() - approval_started
        ) * 1000

        if output.get("__interrupt__"):
            raise TaskServiceError(
                "El grafo continúa interrumpido después de la decisión."
            )

        # Se preserva el consumo registrado antes de la interrupción.
        input_tokens = record.get(
            "input_tokens",
            output.get("input_tokens", 0),
        )
        output_tokens = record.get(
            "output_tokens",
            output.get("output_tokens", 0),
        )
        estimated_cost = record.get("estimated_cost_usd")
        elapsed_ms = record.get("elapsed_ms")

        preserved_metrics = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost,
            "token_usage": record.get("token_usage", {}),
            "elapsed_ms": elapsed_ms,
            "approval_elapsed_ms": approval_elapsed_ms,
        }

        if decision.decision == "reject":
            updated_record = await self._update_record(
                task_id,
                status=TaskStatus.REJECTED.value,
                approval_feedback=decision.feedback,
                draft_answer=None,
                result=None,
                error=None,
                **preserved_metrics,
            )

            return self._public_response(updated_record)

        result = TaskResult(
            answer=output.get("answer", ""),
            sources=output.get("sources", []),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
            elapsed_ms=elapsed_ms,
        )

        updated_record = await self._update_record(
            task_id,
            status=TaskStatus.DONE.value,
            approval_feedback=decision.feedback,
            draft_answer=None,
            result=result.model_dump(mode="json"),
            error=None,
            **preserved_metrics,
        )

        return self._public_response(updated_record)

    async def close(self) -> None:
        """Cierra los recursos propios del servicio."""

        if self._worker_running:
            raise TaskServiceError(
                "Primero detené y esperá el cierre del trabajador."
            )

        if self._owns_redis:
            await self._redis.close()