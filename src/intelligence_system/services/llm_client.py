from dataclasses import dataclass
from typing import Any

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from ..config.settings import Settings, get_settings
from ..observability.costs import record_token_usage


class LLMClientError(RuntimeError):
    """Error base del cliente de lenguaje."""


class LLMConfigurationError(LLMClientError):
    """Falta configuración para inicializar el proveedor."""


class LLMTimeoutError(LLMClientError):
    """El proveedor excedió el tiempo de espera."""


class LLMRateLimitError(LLMClientError):
    """El proveedor rechazó la solicitud por límite de uso."""


class LLMAuthenticationError(LLMClientError):
    """Las credenciales fueron rechazadas."""


class LLMProviderError(LLMClientError):
    """El proveedor no pudo completar la solicitud."""


class LLMResponseError(LLMClientError):
    """La respuesta del proveedor no contiene texto utilizable."""


@dataclass(frozen=True, slots=True)
class LLMGenerationResult:
    content: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None


class LLMClient:
    """Cliente asíncrono y reutilizable del modelo de lenguaje."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: ChatGroq | None = None,
    ) -> None:
        self._settings = settings or get_settings()

        # Permite inyectar un modelo de prueba sin contactar al proveedor.
        self._model = model

    def _get_model(self) -> ChatGroq:
        if self._model is not None:
            return self._model

        settings = self._settings

        missing: list[str] = []

        if not settings.groq_api_key:
            missing.append("GROQ_API_KEY")

        if not settings.groq_model or not settings.groq_model.strip():
            missing.append("GROQ_MODEL")

        if missing:
            raise LLMConfigurationError(
                "Faltan variables necesarias para inicializar Groq: "
                + ", ".join(missing)
            )

        assert settings.groq_api_key is not None
        assert settings.groq_model is not None

        try:
            self._model = ChatGroq(
                model=settings.groq_model,
                api_key=settings.groq_api_key.get_secret_value(),
                timeout=settings.groq_timeout_seconds,
                max_retries=settings.groq_max_retries,
                temperature=0.0,
            )
        except Exception as exc:
            raise LLMConfigurationError(
                "No se pudo inicializar el cliente de Groq. "
                "Verificá la configuración y las dependencias."
            ) from exc

        return self._model

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        # Admite respuestas con bloques de contenido.
        if isinstance(content, list):
            parts = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and isinstance(block.get("text"), str)
            ]

            return "\n".join(parts).strip()

        return ""

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool):
            return value if value >= 0 else None

        return None

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMGenerationResult:
        """Genera una respuesta usando una invocación asíncrona real."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("El prompt no puede estar vacío.")

        messages = []

        if system_prompt and system_prompt.strip():
            messages.append(
                SystemMessage(content=system_prompt.strip())
            )

        messages.append(HumanMessage(content=prompt.strip()))

        model = self._get_model()

        try:
            response = await model.ainvoke(messages)

        except RateLimitError as exc:
            raise LLMRateLimitError(
                "Groq aplicó un límite de solicitudes o tokens."
            ) from exc

        except APITimeoutError as exc:
            raise LLMTimeoutError(
                "La solicitud a Groq excedió el tiempo de espera."
            ) from exc

        except APIConnectionError as exc:
            raise LLMProviderError(
                "No se pudo establecer conexión con Groq."
            ) from exc

        except APIStatusError as exc:
            if exc.status_code in (401, 403):
                raise LLMAuthenticationError(
                    "Groq rechazó las credenciales o los permisos."
                ) from exc

            raise LLMProviderError(
                f"Groq devolvió un error HTTP {exc.status_code}."
            ) from exc

        except Exception as exc:
            raise LLMProviderError(
                "Ocurrió un error inesperado durante la generación."
            ) from exc

        content = self._extract_text(response.content)

        if not content:
            raise LLMResponseError(
                "Groq devolvió una respuesta sin contenido textual."
            )

        metadata = response.response_metadata or {}
        usage = metadata.get("token_usage") or {}
        usage_metadata = getattr(response, "usage_metadata", None) or {}

        input_tokens = self._optional_int(
            usage.get(
                "prompt_tokens",
                usage_metadata.get("input_tokens"),
            )
        )

        output_tokens = self._optional_int(
            usage.get(
                "completion_tokens",
                usage_metadata.get("output_tokens"),
            )
        )

        model_name = (
            metadata.get("model_name")
            or self._settings.groq_model
            or "unknown"
        )

        result = LLMGenerationResult(
            content=content,
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=metadata.get("finish_reason"),
        )

        # Registra el consumo para calcular tokens y costos por tarea.
        record_token_usage(
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

        return result