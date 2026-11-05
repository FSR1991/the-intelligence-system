from fastapi import APIRouter, Request, status

from ...models.schemas import (
    ApprovalDecisionRequest,
    ErrorResponse,
    TaskStatusResponse,
)
from ...services.task_service import TaskService


router = APIRouter(
    prefix="/tasks",
    tags=["Human Approval"],
)


def get_task_service(request: Request) -> TaskService:
    """Obtiene el servicio de tareas administrado por FastAPI."""
    return request.app.state.task_service


@router.post(
    "/{task_id}/approval",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Aprobar o rechazar una tarea",
    description=(
        "Registra la decisión humana y reanuda el grafo LangGraph "
        "desde su checkpoint persistido en Redis."
    ),
    responses={
        404: {
            "model": ErrorResponse,
            "description": "La tarea solicitada no existe.",
        },
        409: {
            "model": ErrorResponse,
            "description": (
                "La tarea no está esperando una aprobación humana."
            ),
        },
    },
)
async def decide_task_approval(
    task_id: str,
    payload: ApprovalDecisionRequest,
    request: Request,
) -> TaskStatusResponse:
    service = get_task_service(request)

    return await service.submit_approval(
        task_id=task_id,
        decision=payload,
    )