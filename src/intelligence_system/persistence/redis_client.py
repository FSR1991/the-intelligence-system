import json
from typing import Any, NoReturn

from redis.asyncio import Redis
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    RedisError,
    TimeoutError as RedisTimeoutError,
)

from ..config.settings import Settings, get_settings


class RedisClientError(RuntimeError):
    """Error base del cliente de Redis."""


class RedisConfigurationError(RedisClientError):
    """Falta una configuración válida de Redis."""


class RedisConnectionFailure(RedisClientError):
    """No se pudo conectar con Redis."""


class RedisOperationError(RedisClientError):
    """Redis no pudo completar una operación."""


class RedisDataError(RedisClientError):
    """Los datos no pudieron serializarse o deserializarse."""


class AsyncRedisClient:
    """Cliente Redis asíncrono para datos, colas y recuperación de tareas."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client: Any | None = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        url = (self._settings.redis_url or "").strip()

        if not url:
            raise RedisConfigurationError(
                "Falta REDIS_URL en el archivo .env."
            )

        try:
            self._client = Redis.from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5.0,
                socket_timeout=5.0,
                health_check_interval=30,
            )
        except (ValueError, RedisError) as exc:
            raise RedisConfigurationError(
                "La URL de Redis no es válida."
            ) from exc

        return self._client

    @staticmethod
    def _raise_operation_error(exc: RedisError) -> NoReturn:
        if isinstance(exc, RedisTimeoutError):
            raise RedisConnectionFailure(
                "Redis excedió el tiempo de espera."
            ) from exc

        if isinstance(exc, RedisConnectionError):
            raise RedisConnectionFailure(
                "No se pudo establecer conexión con Redis."
            ) from exc

        raise RedisOperationError(
            "Redis devolvió un error durante la operación."
        ) from exc

    @staticmethod
    def _encode_json(value: Any) -> str:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise RedisDataError(
                "El valor no se puede serializar como JSON."
            ) from exc

    @staticmethod
    def _decode_json(payload: str) -> Any:
        try:
            return json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise RedisDataError(
                "El contenido almacenado no es JSON válido."
            ) from exc

    async def connect(self) -> None:
        """Comprueba la conexión mediante un ping real."""

        client = self._get_client()

        try:
            await client.ping()
        except RedisError as exc:
            self._raise_operation_error(exc)

    async def ping(self) -> bool:
        """Comprueba la disponibilidad de Redis."""

        client = self._get_client()

        try:
            return bool(await client.ping())
        except RedisError as exc:
            self._raise_operation_error(exc)

    async def set_json(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
    ) -> bool:
        """Guarda un valor JSON; ex indica el TTL en segundos."""

        if not key.strip():
            raise ValueError("La clave de Redis no puede estar vacía.")

        if ex is not None and ex < 1:
            raise ValueError("El TTL debe ser de al menos un segundo.")

        payload = self._encode_json(value)
        client = self._get_client()

        try:
            result = await client.set(key, payload, ex=ex)
            return bool(result)
        except RedisError as exc:
            self._raise_operation_error(exc)

    async def get_json(self, key: str) -> Any | None:
        """Devuelve el valor JSON o None si la clave no existe."""

        if not key.strip():
            raise ValueError("La clave de Redis no puede estar vacía.")

        client = self._get_client()

        try:
            payload = await client.get(key)
        except RedisError as exc:
            self._raise_operation_error(exc)

        if payload is None:
            return None

        return self._decode_json(payload)

    async def delete(self, key: str) -> int:
        """Elimina una clave y devuelve la cantidad eliminada."""

        if not key.strip():
            raise ValueError("La clave de Redis no puede estar vacía.")

        client = self._get_client()

        try:
            return int(await client.delete(key))
        except RedisError as exc:
            self._raise_operation_error(exc)

    async def enqueue_json(
        self,
        key: str,
        value: Any,
    ) -> int:
        """Agrega un elemento JSON al final de una cola Redis."""

        if not key.strip():
            raise ValueError("La clave de Redis no puede estar vacía.")

        payload = self._encode_json(value)
        client = self._get_client()

        try:
            return int(await client.rpush(key, payload))
        except RedisError as exc:
            self._raise_operation_error(exc)

    async def enqueue_task_atomically(
        self,
        record_key: str,
        record: Any,
        queue_key: str,
        payload: Any,
        ex: int,
    ) -> int:
        """Guarda el registro y encola la tarea en una transacción Redis."""

        if not record_key.strip() or not queue_key.strip():
            raise ValueError("Las claves Redis son obligatorias.")

        if record_key == queue_key:
            raise ValueError("Las claves deben ser diferentes.")

        if ex < 1:
            raise ValueError("El TTL debe ser de al menos un segundo.")

        record_json = self._encode_json(record)
        payload_json = self._encode_json(payload)
        client = self._get_client()

        try:
            async with client.pipeline(transaction=True) as pipeline:
                pipeline.set(record_key, record_json, ex=ex)
                pipeline.rpush(queue_key, payload_json)
                results = await pipeline.execute()

            if len(results) != 2 or not results[0]:
                raise RedisOperationError(
                    "No se pudo completar la transacción de la tarea."
                )

            return int(results[1])

        except RedisError as exc:
            self._raise_operation_error(exc)

    async def dequeue_json(
        self,
        key: str,
        timeout: int = 1,
    ) -> Any | None:
        """Retira el primer elemento de una cola sin confirmación."""

        if not key.strip():
            raise ValueError("La clave de Redis no puede estar vacía.")

        if timeout < 0:
            raise ValueError("El timeout no puede ser negativo.")

        client = self._get_client()

        try:
            item = await client.blpop(key, timeout=timeout)
        except RedisError as exc:
            self._raise_operation_error(exc)

        if item is None:
            return None

        return self._decode_json(item[1])

    async def claim_json(
        self,
        pending_key: str,
        processing_key: str,
        timeout: float = 1.0,
    ) -> Any | None:
        """Mueve atómicamente una tarea pendiente a procesamiento."""

        if not pending_key.strip() or not processing_key.strip():
            raise ValueError("Las claves de las colas son obligatorias.")

        if pending_key == processing_key:
            raise ValueError("Las colas deben utilizar claves diferentes.")

        if timeout < 0:
            raise ValueError("El timeout no puede ser negativo.")

        client = self._get_client()

        try:
            payload = await client.blmove(
                pending_key,
                processing_key,
                timeout,
                src="LEFT",
                dest="RIGHT",
            )
        except RedisError as exc:
            self._raise_operation_error(exc)

        if payload is None:
            return None

        return self._decode_json(payload)

    async def ack_json(
        self,
        processing_key: str,
        value: Any,
    ) -> int:
        """Confirma una tarea eliminándola de procesamiento."""

        if not processing_key.strip():
            raise ValueError("La clave de procesamiento es obligatoria.")

        payload = self._encode_json(value)
        client = self._get_client()

        try:
            return int(
                await client.lrem(processing_key, 1, payload)
            )
        except RedisError as exc:
            self._raise_operation_error(exc)

    async def recover_processing_json(
        self,
        processing_key: str,
        pending_key: str,
    ) -> list[Any]:
        """Devuelve a pendientes las tareas no confirmadas."""

        if not processing_key.strip() or not pending_key.strip():
            raise ValueError("Las claves de las colas son obligatorias.")

        if processing_key == pending_key:
            raise ValueError("Las colas deben utilizar claves diferentes.")

        client = self._get_client()
        recovered: list[Any] = []

        try:
            while True:
                payload = await client.lmove(
                    processing_key,
                    pending_key,
                    "RIGHT",
                    "LEFT",
                )

                if payload is None:
                    break

                recovered.append(self._decode_json(payload))

        except RedisError as exc:
            self._raise_operation_error(exc)

        return recovered

    async def close(self) -> None:
        """Cierra el cliente y libera el pool de conexiones."""

        client, self._client = self._client, None

        if client is None:
            return

        try:
            await client.aclose()
        except RedisError as exc:
            self._raise_operation_error(exc)