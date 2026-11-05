from fastapi import APIRouter, Request

from ...models.schemas import HealthResponse


router = APIRouter(
    tags=["Health"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado de salud del sistema",
    description=(
        "Comprueba la disponibilidad de la aplicación y "
        "su conexión con Redis."
    ),
)
async def health_check(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    redis_client = request.app.state.redis_client

    try:
        redis_available = await redis_client.ping()
    except Exception:
        redis_available = False

    return HealthResponse(
        status="ok" if redis_available else "degraded",
        app_name=settings.app_name,
    )