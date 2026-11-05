from __future__ import annotations

from types import TracebackType
from typing import Any

from langgraph.checkpoint.redis import AsyncRedisSaver

from ..config.settings import Settings, get_settings


class CheckpointerError(RuntimeError):
    """Error base del checkpointer."""


class CheckpointerConfigurationError(CheckpointerError):
    """La configuración de Redis es insuficiente."""


class CheckpointerConnectionError(CheckpointerError):
    """No se pudo inicializar o cerrar el checkpointer."""


class RedisCheckpointer:
    """Administra el ciclo de vida del checkpointer asíncrono de LangGraph."""

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._context_manager: Any | None = None
        self._saver: AsyncRedisSaver | None = None

    @property
    def is_connected(self) -> bool:
        """Indica si el checkpointer está inicializado."""
        return self._saver is not None

    @property
    def saver(self) -> AsyncRedisSaver:
        """Devuelve el saver para compilar o ejecutar un grafo."""

        if self._saver is None:
            raise CheckpointerError(
                "El checkpointer todavía no está conectado. "
                "Utilizá primero 'async with RedisCheckpointer(...)'."
            )

        return self._saver

    async def connect(self) -> AsyncRedisSaver:
        """Abre Redis e inicializa los índices del checkpointer."""

        if self._saver is not None:
            return self._saver

        redis_url = self._settings.redis_url

        if hasattr(redis_url, "get_secret_value"):
            redis_url = redis_url.get_secret_value()

        redis_url = (redis_url or "").strip()

        if not redis_url:
            raise CheckpointerConfigurationError(
                "Falta REDIS_URL en el archivo .env."
            )

        context_manager = AsyncRedisSaver.from_conn_string(
            redis_url
        )

        try:
            saver = await context_manager.__aenter__()

        except Exception as exc:
            raise CheckpointerConnectionError(
                "No se pudo inicializar el checkpointer de Redis. "
                "Verificá la conexión, RedisJSON y RediSearch."
            ) from exc

        self._context_manager = context_manager
        self._saver = saver

        return saver

    async def close(self) -> None:
        """Cierra el contexto y libera los recursos de Redis."""

        context_manager = self._context_manager

        if context_manager is None:
            return

        self._context_manager = None
        self._saver = None

        try:
            await context_manager.__aexit__(None, None, None)

        except Exception as exc:
            raise CheckpointerConnectionError(
                "No se pudo cerrar correctamente el checkpointer."
            ) from exc

    async def __aenter__(self) -> AsyncRedisSaver:
        """Permite utilizar el checkpointer con async with."""

        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Libera Redis al salir del contexto asíncrono."""

        context_manager = self._context_manager

        if context_manager is None:
            return

        self._context_manager = None
        self._saver = None

        try:
            await context_manager.__aexit__(
                exc_type,
                exc_value,
                traceback,
            )

        except Exception as exc:
            raise CheckpointerConnectionError(
                "No se pudo cerrar correctamente el checkpointer."
            ) from exc