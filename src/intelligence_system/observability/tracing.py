from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from langsmith import Client
from langsmith.run_helpers import tracing_context

from ..config.settings import Settings, get_settings


class TracingConfigurationError(RuntimeError):
    """La configuración de LangSmith está incompleta."""


@lru_cache(maxsize=4)
def _get_client(api_key: str) -> Client:
    """Reutiliza el cliente para evitar crear uno por cada tarea."""

    return Client(api_key=api_key)


@contextmanager
def trace_task(
    settings: Settings | None = None,
    *,
    task_id: str,
    conversation_id: str | None = None,
    operation: str = "task_execution",
    requires_approval: bool = False,
) -> Iterator[None]:
    """Agrupa una ejecución de LangGraph dentro de un proyecto LangSmith.

    Si el tracing está deshabilitado, permite ejecutar la tarea
    normalmente sin enviar trazas.
    """

    config = settings or get_settings()

    if not config.langsmith_tracing:
        yield
        return

    if not config.langsmith_api_key:
        raise TracingConfigurationError(
            "LANGSMITH_TRACING está habilitado, pero falta LANGSMITH_API_KEY."
        )

    api_key = config.langsmith_api_key.get_secret_value()

    if not api_key.strip():
        raise TracingConfigurationError(
            "LANGSMITH_API_KEY no puede estar vacía."
        )

    if not config.langsmith_project.strip():
        raise TracingConfigurationError(
            "LANGSMITH_PROJECT no puede estar vacío."
        )

    client = _get_client(api_key)

    tags = [
        "the-intelligence-system",
        operation,
        "human-approval" if requires_approval else "standard-task",
    ]

    metadata = {
        "task_id": task_id,
        "operation": operation,
        "requires_approval": requires_approval,
    }

    if conversation_id:
        metadata["conversation_id"] = conversation_id

    with tracing_context(
        project_name=config.langsmith_project,
        tags=tags,
        metadata=metadata,
        enabled=True,
        client=client,
    ):
        yield