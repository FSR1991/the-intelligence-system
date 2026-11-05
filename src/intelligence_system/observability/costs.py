from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterator


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal


# Tarifas públicas de Groq para los modelos utilizados.
# Se mantiene la tarifa asociada al identificador exacto del modelo.
MODEL_PRICING: dict[str, ModelPricing] = {
    "openai/gpt-oss-20b": ModelPricing(
        input_usd_per_million=Decimal("0.075"),
        output_usd_per_million=Decimal("0.30"),
    ),
}


@dataclass(slots=True)
class ModelTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


@dataclass(slots=True)
class TokenUsage:
    """Acumula el consumo de todas las llamadas LLM de una tarea."""

    by_model: dict[str, ModelTokenUsage] = field(default_factory=dict)
    calls: int = 0
    missing_usage_calls: int = 0

    def record(
        self,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        model_name = (model or "unknown").strip().lower()

        usage = self.by_model.setdefault(
            model_name,
            ModelTokenUsage(),
        )

        usage.calls += 1
        self.calls += 1

        if input_tokens is None or output_tokens is None:
            self.missing_usage_calls += 1

        usage.input_tokens += input_tokens or 0
        usage.output_tokens += output_tokens or 0

    @property
    def input_tokens(self) -> int:
        return sum(item.input_tokens for item in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(item.output_tokens for item in self.by_model.values())

    @property
    def complete(self) -> bool:
        return self.calls > 0 and self.missing_usage_calls == 0

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usage_complete": self.complete,
            "by_model": {
                model: {
                    "input_tokens": item.input_tokens,
                    "output_tokens": item.output_tokens,
                    "calls": item.calls,
                }
                for model, item in self.by_model.items()
            },
        }


_CURRENT_USAGE: ContextVar[TokenUsage | None] = ContextVar(
    "intelligence_system_token_usage",
    default=None,
)


@contextmanager
def track_token_usage() -> Iterator[TokenUsage]:
    """Crea un acumulador aislado para la ejecución actual."""

    usage = TokenUsage()
    token: Token[TokenUsage | None] = _CURRENT_USAGE.set(usage)

    try:
        yield usage
    finally:
        _CURRENT_USAGE.reset(token)


def record_token_usage(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> None:
    """Registra una llamada LLM si existe una ejecución activa."""

    usage = _CURRENT_USAGE.get()

    if usage is not None:
        usage.record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def estimate_cost_usd(usage: TokenUsage) -> float | None:
    """Calcula el costo estimado usando las tarifas conocidas.

    Devuelve None cuando faltan métricas o no existe una tarifa
    conocida para alguno de los modelos utilizados.
    """

    if not usage.complete:
        return None

    total = Decimal("0")
    one_million = Decimal("1000000")

    for model, tokens in usage.by_model.items():
        pricing = MODEL_PRICING.get(model)

        if pricing is None:
            return None

        input_cost = (
            Decimal(tokens.input_tokens)
            * pricing.input_usd_per_million
            / one_million
        )

        output_cost = (
            Decimal(tokens.output_tokens)
            * pricing.output_usd_per_million
            / one_million
        )

        total += input_cost + output_cost

    return float(total.quantize(Decimal("0.0000000001")))