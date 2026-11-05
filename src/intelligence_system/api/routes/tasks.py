from fastapi import APIRouter, Query, Request, status

from ...models.schemas import (
    ErrorResponse,
    TaskAcceptedResponse,
    TaskCreateRequest,
    TaskStatusResponse,
)
from ...services.task_service import TaskService


router = APIRouter(
    prefix="/tasks",
    tags=["Tasks"],
)


def get_task_service(request: Request) -> TaskService:
    """Obtiene el servicio de tareas administrado por la aplicación."""
    return request.app.state.task_service


@router.post(
    "",
    response_model=TaskAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enviar una tarea",
    description=(
        "Registra una tarea en Redis y la incorpora a la cola "
        "para su procesamiento asíncrono."
    ),
)
async def submit_task(
    request: Request,
    payload: TaskCreateRequest,
    requires_approval: bool = Query(
        default=False,
        description=(
            "Indica si la tarea requiere aprobación humana "
            "antes de entregar el resultado."
        ),
    ),
) -> TaskAcceptedResponse:
    service = get_task_service(request)

    return await service.submit(
        request=payload,
        requires_approval=requires_approval,
    )


@router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    summary="Consultar una tarea",
    description=(
        "Recupera desde Redis el estado, la conversación, "
        "el resultado y los posibles errores de una tarea."
    ),
    responses={
        404: {
            "model": ErrorResponse,
            "description": "La tarea solicitada no existe.",
        },
    },
)
async def get_task_status(
    task_id: str,
    request: Request,
) -> TaskStatusResponse:
    service = get_task_service(request)

    return await service.get_status(task_id)