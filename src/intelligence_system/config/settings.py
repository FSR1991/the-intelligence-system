from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    # Aplicación y API
    app_name: str = "The Intelligence System"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"
    api_prefix: str = "/api/v1"

    # LLM: las credenciales y el modelo se proporcionan por entorno.
    groq_api_key: SecretStr | None = None
    groq_model: str | None = None
    groq_timeout_seconds: float = Field(default=30.0, gt=0, le=180)
    groq_max_retries: int = Field(default=2, ge=0, le=10)

    # Redis: cola, estado de tareas y checkpoints.
    redis_url: str | None = None
    redis_task_queue_key: str = "intelligence:tasks:queue"
    task_ttl_seconds: int = Field(default=86400, ge=60)
    worker_max_concurrency: int = Field(default=3, ge=1, le=20)

    # Pinecone: recuperación semántica.
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str | None = None
    pinecone_namespace: str | None = None
    pinecone_embedding_model: str | None = None

    # LangSmith: observabilidad.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "the-intelligence-system"

    # Ingesta y recuperación RAG.
    rag_chunk_size: int = Field(default=800, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=120, ge=0, le=1000)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_candidate_k: int = Field(default=20, ge=1, le=100)
    rag_rrf_k: int = Field(default=60, ge=1, le=200)

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                "RAG_CHUNK_OVERLAP debe ser menor que RAG_CHUNK_SIZE."
            )
        return self

    def require_values(self, *names: str) -> None:
        """Exige configuración cuando se va a inicializar un servicio real."""
        missing: list[str] = []

        for name in names:
            value = getattr(self, name, None)

            if isinstance(value, SecretStr):
                value = value.get_secret_value()

            if value is None or (
                isinstance(value, str) and not value.strip()
            ):
                missing.append(name.upper())

        if missing:
            raise ValueError(
                "Faltan variables de entorno necesarias: "
                + ", ".join(missing)
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()