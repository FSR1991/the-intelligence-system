from __future__ import annotations

import asyncio
import hashlib
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from langchain_core.documents import Document
from pypdf import PdfReader


class DocumentIngestionError(RuntimeError):
    """Error controlado durante la ingesta de documentos."""


class PDFIngestionService:
    """Extrae texto y metadatos de documentos PDF de forma asíncrona."""

    async def extract_pdf(
        self,
        content: bytes,
        filename: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        """Extrae las páginas de un PDF y devuelve documentos LangChain."""

        if not isinstance(content, bytes) or not content:
            raise DocumentIngestionError(
                "El contenido del PDF está vacío o no es válido."
            )

        if not isinstance(filename, str) or not filename.strip():
            raise DocumentIngestionError(
                "El nombre del archivo no puede estar vacío."
            )

        if Path(filename).suffix.lower() != ".pdf":
            raise DocumentIngestionError(
                "El formato del archivo debe ser PDF."
            )

        return await asyncio.to_thread(
            self._extract_pdf_sync,
            content,
            Path(filename).name,
            dict(metadata or {}),
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normaliza espacios sin eliminar la separación entre párrafos."""

        text = text.replace("\x00", "").replace("\r\n", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _extract_pdf_sync(
        cls,
        content: bytes,
        filename: str,
        extra_metadata: dict[str, Any],
    ) -> list[Document]:
        """Realiza la lectura del PDF en un hilo de trabajo."""

        try:
            reader = PdfReader(BytesIO(content), strict=False)
        except Exception as exc:
            raise DocumentIngestionError(
                "No se pudo leer el PDF. El archivo puede estar dañado."
            ) from exc

        if reader.is_encrypted:
            raise DocumentIngestionError(
                "El PDF está cifrado y no puede procesarse sin contraseña."
            )

        if not reader.pages:
            raise DocumentIngestionError(
                "El PDF no contiene páginas."
            )

        document_id = hashlib.sha256(content).hexdigest()
        documents: list[Document] = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                extracted_text = page.extract_text() or ""
            except Exception as exc:
                raise DocumentIngestionError(
                    f"No se pudo extraer el texto de la página "
                    f"{page_number}."
                ) from exc

            text = cls._normalize_text(extracted_text)

            # Las páginas sin texto no aportan contenido recuperable.
            if not text:
                continue

            page_metadata = {
                **extra_metadata,
                "source": filename,
                "file_name": filename,
                "document_id": document_id,
                "page": page_number,
            }

            documents.append(
                Document(
                    page_content=text,
                    metadata=page_metadata,
                )
            )

        if not documents:
            raise DocumentIngestionError(
                "No se pudo extraer texto del PDF. "
                "Si es un documento escaneado, necesita OCR."
            )

        return documents