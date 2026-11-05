from types import SimpleNamespace

import pytest

from src.intelligence_system.config.settings import Settings
from src.intelligence_system.services.llm_client import (
    LLMClient,
    LLMConfigurationError,
    LLMResponseError,
)


class FakeAsyncModel:
    """Simula una respuesta del proveedor sin realizar llamadas externas."""

    async def ainvoke(self, messages):
        assert len(messages) == 2
        return SimpleNamespace(
            content="Respuesta de prueba",
            response_metadata={
                "model_name": "fake-model",
                "token_usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                },
                "finish_reason": "stop",
            },
            usage_metadata={},
        )


class EmptyAsyncModel:
    async def ainvoke(self, messages):
        return SimpleNamespace(
            content="",
            response_metadata={},
            usage_metadata={},
        )


@pytest.mark.asyncio
async def test_generate_returns_text_and_usage():
    settings = Settings(groq_api_key=None, groq_model="fake-model")
    client = LLMClient(settings=settings, model=FakeAsyncModel())

    result = await client.generate(
        prompt="Pregunta de prueba",
        system_prompt="Respondé con precisión.",
    )

    assert result.content == "Respuesta de prueba"
    assert result.model == "fake-model"
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_generate_rejects_empty_prompt():
    client = LLMClient(
        settings=Settings(groq_api_key=None, groq_model=None),
        model=FakeAsyncModel(),
    )

    with pytest.raises(ValueError, match="prompt"):
        await client.generate("   ")


@pytest.mark.asyncio
async def test_generate_requires_provider_configuration():
    client = LLMClient(
        settings=Settings(groq_api_key=None, groq_model=None),
    )

    with pytest.raises(LLMConfigurationError):
        await client.generate("Pregunta de prueba")


@pytest.mark.asyncio
async def test_generate_rejects_empty_provider_response():
    client = LLMClient(
        settings=Settings(groq_api_key=None, groq_model="fake-model"),
        model=EmptyAsyncModel(),
    )

    with pytest.raises(LLMResponseError):
        await client.generate("Pregunta de prueba")