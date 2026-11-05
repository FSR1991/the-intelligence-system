from __future__ import annotations

import json
import re

from ..models.schemas import SupervisorDecision
from ..services.llm_client import LLMClient


class SupervisorError(RuntimeError):
    """Error base del Supervisor."""


class SupervisorResponseError(SupervisorError):
    """La respuesta del modelo no contiene una decisión válida."""


class Supervisor:
    """Coordina la selección del siguiente agente especializado."""

    SYSTEM_PROMPT = """
Eres el Supervisor de The Intelligence System.

Tu única responsabilidad es decidir qué agente debe ejecutarse a
continuación.

Agentes disponibles:
- retrieval_agent: recupera información de la base documental mediante RAG.
- answer_agent: construye la respuesta final utilizando la consulta y
  el contexto documental disponible.
- finish: finaliza el flujo cuando no se necesita otro agente.

Reglas:
1. Selecciona retrieval_agent cuando sea necesario consultar documentos,
   información institucional o conocimientos que puedan estar en el RAG.
2. Selecciona answer_agent cuando ya exista contexto documental suficiente
   o la consulta pueda responderse sin recuperación documental.
3. Si la recuperación ya se realizó, selecciona answer_agent para elaborar
   la respuesta final.
4. No inventes agentes ni herramientas.
5. No ejecutes tareas: únicamente decides la delegación.
6. Devuelve exclusivamente un objeto JSON válido con esta estructura:
   {
       "next_agent": "retrieval_agent",
       "reason": "Motivo concreto de la decisión."
   }
7. next_agent solo puede ser retrieval_agent, answer_agent o finish.
"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()

    @staticmethod
    def _extract_json(content: str) -> dict:
        """Extrae el objeto JSON de la respuesta del modelo."""

        cleaned = content.strip()

        # Elimina posibles bloques Markdown.
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Recupera un objeto JSON si el modelo agregó texto externo.
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

            if not match:
                raise SupervisorResponseError(
                    "El Supervisor no devolvió un objeto JSON válido."
                )

            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as exc:
                raise SupervisorResponseError(
                    "No se pudo interpretar la decisión del Supervisor."
                ) from exc

        if not isinstance(data, dict):
            raise SupervisorResponseError(
                "La decisión del Supervisor debe ser un objeto JSON."
            )

        return data

    async def decide(
        self,
        query: str,
        retrieval_completed: bool = False,
        has_retrieved_context: bool = False,
    ) -> SupervisorDecision:
        """Decide qué agente debe ejecutarse a continuación."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("La consulta del Supervisor no puede estar vacía.")

        state_context = {
            "retrieval_completed": retrieval_completed,
            "has_retrieved_context": has_retrieved_context,
        }

        prompt = (
            "Evalúa la siguiente solicitud y selecciona el próximo agente.\n\n"
            f"Consulta del usuario:\n{query.strip()}\n\n"
            f"Estado actual del flujo:\n"
            f"{json.dumps(state_context, ensure_ascii=False)}\n\n"
            "Responde únicamente con el JSON solicitado."
        )

        try:
            generation = await self._llm.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
            )

            data = self._extract_json(generation.content)

            decision = SupervisorDecision.model_validate(data)

        except SupervisorResponseError:
            raise
        except Exception as exc:
            raise SupervisorResponseError(
                "No se pudo obtener una decisión válida del Supervisor."
            ) from exc

        # No se permite volver a recuperar documentos indefinidamente.
        if retrieval_completed and decision.next_agent == "retrieval_agent":
            return SupervisorDecision(
                next_agent="answer_agent",
                reason=(
                    "La recuperación documental ya fue realizada. "
                    "Se delega la elaboración de la respuesta final."
                ),
            )

        return decision