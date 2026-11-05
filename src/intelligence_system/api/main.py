from __future__ import annotations

import asyncio
import logging

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config.settings import get_settings
from ..persistence.redis_client import (
    AsyncRedisClient,
    RedisClientError,
)
from ..services.task_service import (
    TaskNotFoundError,
    TaskService,
    TaskServiceError,
    TaskStateError,
)
from .routes import approval, health, tasks


settings = get_settings()

logging.basicConfig(
    level=getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    ),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)


def log_worker_completion(task: asyncio.Task) -> None:
    """Registra la finalización inesperada del trabajador."""

    if task.cancelled():
        logger.warning("El trabajador de tareas fue cancelado.")
        return

    exception = task.exception()

    if exception is not None:
        logger.error(
            "El trabajador de tareas terminó con un error.",
            exc_info=(
                type(exception),
                exception,
                exception.__traceback__,
            ),
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Administra conexiones y trabajador durante el ciclo de vida."""

    redis_client = AsyncRedisClient(settings=settings)

    try:
        # Redis es necesario para aceptar y persistir tareas.
        await redis_client.connect()

        task_service = TaskService(
            settings=settings,
            redis_client=redis_client,
        )

        app.state.settings = settings
        app.state.redis_client = redis_client
        app.state.task_service = task_service

        logger.info("Conexión Redis establecida.")

        # El trabajador consume la cola independientemente
        # de las solicitudes HTTP.
        worker_task = asyncio.create_task(
            task_service.run_worker(),
            name="intelligence-task-worker",
        )

        worker_task.add_done_callback(
            log_worker_completion
        )

        app.state.worker_task = worker_task

        logger.info(
            "API iniciada: %s",
            settings.app_name,
        )

        yield

    finally:
        worker_task = getattr(
            app.state,
            "worker_task",
            None,
        )

        if worker_task is not None:
            await app.state.task_service.stop_worker()

            try:
                await asyncio.wait_for(
                    worker_task,
                    timeout=15,
                )
            except TimeoutError:
                logger.warning(
                    "Se agotó el tiempo de cierre del trabajador."
                )

                worker_task.cancel()

                await asyncio.gather(
                    worker_task,
                    return_exceptions=True,
                )
            except Exception:
                logger.exception(
                    "Error durante el cierre del trabajador."
                )

        await redis_client.close()
        logger.info("Conexión Redis cerrada.")


app = FastAPI(
    title=settings.app_name,
    description=(
        "API asíncrona para procesamiento de tareas mediante "
        "RAG avanzado, agentes LangGraph, persistencia Redis "
        "y aprobación humana."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


@app.exception_handler(TaskNotFoundError)
async def task_not_found_handler(
    request: Request,
    exc: TaskNotFoundError,
) -> JSONResponse:
    from ..models.schemas import ErrorResponse

    response = ErrorResponse(
        code="TASK_NOT_FOUND",
        message=str(exc),
    )

    return JSONResponse(
        status_code=404,
        content=response.model_dump(
            mode="json",
            exclude_none=True,
        ),
    )


@app.exception_handler(TaskStateError)
async def task_state_handler(
    request: Request,
    exc: TaskStateError,
) -> JSONResponse:
    from ..models.schemas import ErrorResponse

    response = ErrorResponse(
        code="INVALID_TASK_STATE",
        message=str(exc),
    )

    return JSONResponse(
        status_code=409,
        content=response.model_dump(
            mode="json",
            exclude_none=True,
        ),
    )


@app.exception_handler(RedisClientError)
async def redis_error_handler(
    request: Request,
    exc: RedisClientError,
) -> JSONResponse:
    from ..models.schemas import ErrorResponse

    response = ErrorResponse(
        code="REDIS_UNAVAILABLE",
        message="El servicio de persistencia no está disponible.",
    )

    logger.error("Error de Redis: %s", exc)

    return JSONResponse(
        status_code=503,
        content=response.model_dump(
            mode="json",
            exclude_none=True,
        ),
    )


@app.exception_handler(TaskServiceError)
async def task_service_error_handler(
    request: Request,
    exc: TaskServiceError,
) -> JSONResponse:
    from ..models.schemas import ErrorResponse

    response = ErrorResponse(
        code="TASK_SERVICE_ERROR",
        message=str(exc),
    )

    return JSONResponse(
        status_code=500,
        content=response.model_dump(
            mode="json",
            exclude_none=True,
        ),
    )


app.include_router(
    tasks.router,
    prefix=settings.api_prefix,
)

app.include_router(
    approval.router,
    prefix=settings.api_prefix,
)

app.include_router(
    health.router,
    prefix=settings.api_prefix,
)


@app.get(
    "/",
    tags=["General"],
    summary="Información de la API",
)
async def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "status": "available",
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }