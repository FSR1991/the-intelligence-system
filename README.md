The Intelligence System

Sistema de inteligencia artificial multiagente con RAG, LangGraph, FastAPI, Redis y LangSmith.

1. Descripción del proyecto

The Intelligence System es una plataforma de inteligencia artificial diseñada para resolver consultas que requieren recuperación de conocimiento documental, razonamiento de varios pasos y coordinación de agentes especializados.

El sistema integra una arquitectura RAG (Retrieval-Augmented Generation), un Supervisor implementado con LangGraph, agentes especializados, una API REST asíncrona y persistencia mediante Redis.

Las solicitudes se procesan en segundo plano mediante un worker. LangGraph conserva el estado de las ejecuciones mediante un Checkpointer, mientras que el mecanismo Human-in-the-Loop (HITL) permite interrumpir tareas para solicitar aprobación o rechazo humano.

La solución utiliza Groq como proveedor de modelos de lenguaje, Pinecone para la recuperación documental y LangSmith para la observabilidad.

El proyecto integra y refactoriza las funcionalidades desarrolladas en las pre-entregas de los módulos 6 y 7.

2. Funcionalidades principales

- Ingesta de documentos PDF y extracción de texto.
- Fragmentación documental y conservación de metadatos.
- Recuperación híbrida semántica y léxica.
- Fusión de resultados mediante Reciprocal Rank Fusion (RRF).
- Arquitectura multiagente coordinada por un Supervisor.
- Agentes especializados de recuperación y generación de respuestas.
- Procesamiento asíncrono de tareas mediante FastAPI y un worker.
- Persistencia de estados y checkpoints mediante Redis.
- Interrupción, aprobación y rechazo de tareas mediante HITL.
- Validación de entradas y salidas con Pydantic.
- Trazabilidad de ejecuciones, consumo de tokens, latencia y estimación de costos mediante LangSmith.

3. Arquitectura del sistema

Componentes principales

Componente| Responsabilidad| Tecnología
API REST| Recepción y consulta de tareas| FastAPI
Task Service| Gestión de tareas y estados| Python async
Worker| Procesamiento en segundo plano| asyncio
Supervisor| Coordinación y delegación| LangGraph y Groq
Retrieval Agent| Recuperación de conocimiento| RAG y Pinecone
Answer Agent| Generación de respuestas| Groq
Agent Tools| Herramientas estructuradas de los agentes| Pydantic
PDF Ingestion| Extracción de texto documental| PyPDF
Chunking| Fragmentación y metadatos| LangChain Text Splitters
Hybrid Retriever| Recuperación semántica y léxica| Pinecone
Task Storage| Cola y estados de tareas| Redis
Checkpointer| Persistencia del estado del grafo| LangGraph Checkpoint Redis
Observability| Trazabilidad de ejecuciones| LangSmith

Diagrama del grafo multiagente

flowchart TD
    A["Usuario"] --> B["FastAPI REST API"]
    B --> C["Task Service"]
    C --> D["Redis: cola y estado de tareas"]
    D --> E["Worker asíncrono"]
    E --> F["LangGraph StateGraph"]
    F --> G["Supervisor"]

    G -->|"Requiere conocimiento documental"| H["Retrieval Agent"]
    G -->|"Responder directamente"| I["Answer Agent"]

    H --> J["Agent Tools"]
    J --> K["Hybrid RAG Retriever"]

    K --> L["Pinecone: búsqueda semántica"]
    K --> M["Pinecone: búsqueda léxica"]

    L --> N["Reciprocal Rank Fusion"]
    M --> N

    N --> O["Fuentes y metadatos"]
    O --> G

    G -->|"Delegar respuesta"| I
    I --> P["Groq LLM"]

    P --> Q{"¿Requiere aprobación?"}

    Q -->|"No"| R["Resultado final"]
    Q -->|"Sí"| S["HITL: interrupt"]
    S --> T["Checkpoint en Redis"]
    T --> U["Decisión humana mediante API"]
    U --> V["Reanudación mediante Command"]
    V --> W{"Decisión"}

    W -->|"Aprobada"| R
    W -->|"Rechazada"| X["Tarea rechazada"]

    R --> Y["Redis: persistencia del resultado"]
    R --> Z["LangSmith: trazabilidad"]

Flujo de ingesta documental

flowchart TD
    A["Documento PDF"] --> B["PDFIngestionService"]
    B --> C["Extracción de texto y páginas"]
    C --> D["Normalización y metadatos"]
    D --> E["RAGChunkingService"]
    E --> F["Fragmentos documentales"]
    F --> G["HybridRAGRetriever"]
    G --> H["Pinecone: indexación"]
    H --> I["Base de conocimiento"]

Recuperación híbrida

El recuperador utiliza dos estrategias complementarias:

1. Búsqueda semántica: utiliza "EmbedQuery" para recuperar contenido relacionado conceptualmente con la consulta.
2. Búsqueda léxica: utiliza "TextQuery" para recuperar coincidencias textuales.

Ambas búsquedas se ejecutan de manera asíncrona y concurrente. Los resultados se combinan mediante Reciprocal Rank Fusion (RRF), conservando los identificadores documentales y los metadatos relevantes.

4. Persistencia y aprobación humana

Redis y LangGraph Checkpointer

Redis cumple dos funciones principales:

- Persistir estados, resultados y cola de tareas.
- Almacenar checkpoints de LangGraph para conservar el estado de las conversaciones y permitir recuperar ejecuciones.

Cada conversación utiliza un identificador "thread_id" para asociar el estado persistido con su ejecución.

El Checkpointer utiliza "AsyncRedisSaver" mediante la integración "langgraph-checkpoint-redis".

La instancia Redis debe ofrecer compatibilidad con RedisJSON y RediSearch, requeridos por la integración RedisVL utilizada por el Checkpointer.

Human-in-the-Loop (HITL)

Las tareas que requieren aprobación humana pueden detener el flujo mediante "interrupt".

El estado queda persistido en Redis hasta recibir una decisión desde la API:

- "approve": aprueba la ejecución.
- "reject": rechaza la ejecución.

Después de recibir la decisión, el sistema reanuda el grafo mediante "Command(resume=...)", utilizando el checkpoint existente.

5. Tecnologías utilizadas

Área| Tecnología
Lenguaje| Python 3.12
API REST| FastAPI
Servidor ASGI| Uvicorn
Validación| Pydantic
Configuración| Pydantic Settings
Orquestación multiagente| LangGraph
Integración LLM| LangChain Groq
Modelo de lenguaje| Groq
Base documental| Pinecone
Persistencia| Redis
Checkpointer| LangGraph Checkpoint Redis
Fragmentación| LangChain Text Splitters
Extracción PDF| PyPDF
Recuperación léxica| rank-bm25
Observabilidad| LangSmith
Comunicación HTTP| HTTPX
Pruebas| Pytest y Pytest AsyncIO

6. Estructura del proyecto

the-intelligence-system/
│
├── data/
│   └── knowledge_base/
│       └── entrega_final.md
│
├── screenshots/
│   ├── r2-pruebas-asincronicas.png
│   ├── r3-persistencia-redis-checkpointer.png
│   ├── r4-validacion-pydantic.png
│   ├── r5-langsmith-traza-detalle.png
│   ├── r7-despliegue-local.png
│   ├── r7-tarea-real-done.png
│   └── suite-completa-44-tests.png
│
├── scripts/
│   └── ingest_knowledge.py
│
├── src/
│   ├── __init__.py
│   └── intelligence_system/
│       ├── __init__.py
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── graph.py
│       │   ├── supervisor.py
│       │   └── tools.py
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── routes/
│       │       ├── __init__.py
│       │       ├── approval.py
│       │       ├── health.py
│       │       └── tasks.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py
│       │
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── costs.py
│       │   └── tracing.py
│       │
│       ├── persistence/
│       │   ├── __init__.py
│       │   ├── checkpointer.py
│       │   └── redis_client.py
│       │
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── chunking.py
│       │   ├── ingestion.py
│       │   └── retrieval.py
│       │
│       └── services/
│           ├── __init__.py
│           ├── llm_client.py
│           └── task_service.py
│
├── tests/
│   ├── __init__.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_agents_redis.py
│   │   ├── test_api.py
│   │   ├── test_persistence.py
│   │   └── test_task_service_redis.py
│   │
│   └── unit/
│       ├── __init__.py
│       ├── test_agents.py
│       ├── test_agents_hitl.py
│       ├── test_llm_client.py
│       ├── test_rag.py
│       ├── test_redis_client.py
│       ├── test_retrieval.py
│       └── test_task_service.py
│
├── .env.example
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
├── requirements-dev.txt
└── run.sh

7. Requisitos previos

- Python 3.12 o superior.
- Git.
- Git Bash en Windows.
- Conexión a Internet.
- Credenciales de Groq.
- Una instancia Redis compatible con RedisJSON y RediSearch.
- Un índice y namespace configurados en Pinecone.
- Credenciales de LangSmith para utilizar observabilidad remota.

Para el mecanismo de ejecución local documentado en este proyecto no se requiere Docker Desktop.

8. Instalación y configuración

8.1. Clonar el repositorio

git clone https://github.com/FSR1991/the-intelligence-system.git
cd the-intelligence-system

8.2. Crear el entorno virtual

python -m venv .venv

8.3. Instalar dependencias

python -m pip install --upgrade pip
pip install -r requirements-dev.txt

8.4. Configurar variables de entorno

Crear el archivo local:

cp .env.example .env

Completar en ".env" las credenciales y parámetros correspondientes.

Variable| Descripción
"APP_NAME"| Nombre de la aplicación
"ENVIRONMENT"| Entorno de ejecución
"LOG_LEVEL"| Nivel de registros
"API_PREFIX"| Prefijo de la API
"GROQ_API_KEY"| Credencial de Groq
"GROQ_MODEL"| Modelo de lenguaje
"GROQ_TIMEOUT_SECONDS"| Tiempo máximo de espera del LLM
"GROQ_MAX_RETRIES"| Cantidad de reintentos
"REDIS_URL"| URL de conexión a Redis
"REDIS_TASK_QUEUE_KEY"| Identificador de la cola
"TASK_TTL_SECONDS"| Tiempo de vida de las tareas
"WORKER_MAX_CONCURRENCY"| Concurrencia máxima del worker
"PINECONE_API_KEY"| Credencial de Pinecone
"PINECONE_INDEX_NAME"| Nombre del índice
"PINECONE_NAMESPACE"| Namespace documental
"PINECONE_EMBEDDING_MODEL"| Modelo de embeddings configurado
"LANGSMITH_TRACING"| Activación de trazas
"LANGSMITH_API_KEY"| Credencial de LangSmith
"LANGSMITH_PROJECT"| Nombre del proyecto de trazabilidad
"RAG_CHUNK_SIZE"| Tamaño de los fragmentos
"RAG_CHUNK_OVERLAP"| Solapamiento entre fragmentos
"RAG_TOP_K"| Cantidad final de resultados
"RAG_CANDIDATE_K"| Cantidad de candidatos recuperados
"RAG_RRF_K"| Parámetro de fusión RRF

La estructura de configuración y los valores de ejemplo se encuentran en ".env.example".

Para habilitar la observabilidad remota:

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=tu_clave_de_langsmith
LANGSMITH_PROJECT=the-intelligence-system

Las credenciales reales deben permanecer únicamente en ".env". Nunca deben publicarse en GitHub.

9. Ingesta de conocimiento

El proyecto incluye un script para procesar e indexar el documento de conocimiento incluido en "data/knowledge_base/entrega_final.md".

Con el entorno virtual y las variables de entorno configurados, ejecutar desde la raíz:

python scripts/ingest_knowledge.py

El script:

1. Lee el documento de conocimiento.
2. Genera su identificador documental.
3. Fragmenta el contenido utilizando "RAGChunkingService".
4. Indexa los fragmentos en Pinecone.
5. Ejecuta una búsqueda híbrida de validación.
6. Muestra los resultados recuperados.

Se requiere que las credenciales, el índice y el namespace de Pinecone estén correctamente configurados.

10. Ejecución local

Con ".venv" creado, las dependencias instaladas y ".env" configurado, ejecutar desde Git Bash en la raíz del proyecto:

bash run.sh

El script verifica el entorno y las variables locales, configura "PYTHONPATH" e inicia Uvicorn con FastAPI.

Durante el ciclo de vida de la aplicación:

- Se establece la conexión con Redis.
- Se inicializa el servicio de tareas.
- Se inicia el worker asíncrono.
- El worker comienza a procesar las tareas encoladas.

Direcciones disponibles:

Servicio| Dirección
API| http://127.0.0.1:8000
Swagger UI| http://127.0.0.1:8000/docs
ReDoc| http://127.0.0.1:8000/redoc
Health Check| http://127.0.0.1:8000/api/v1/health

Para detener el servidor, utilizar "Ctrl + C" en la terminal.

11. API REST

La documentación interactiva se encuentra disponible en "/docs".

Método| Endpoint| Funcionalidad
GET| "/"| Información general de la API
GET| "/api/v1/health"| Estado de la aplicación y conexión Redis
POST| "/api/v1/tasks"| Crear y encolar una tarea
GET| "/api/v1/tasks/{task_id}"| Consultar estado y resultado
POST| "/api/v1/tasks/{task_id}/approval"| Aprobar o rechazar una tarea

Crear una tarea

POST /api/v1/tasks?requires_approval=false
Content-Type: application/json

Ejemplo del cuerpo:

{
  "query": "Explicá la arquitectura del sistema y cómo funciona el RAG.",
  "conversation_id": null,
  "metadata_filter": {}
}

La API devuelve "202 Accepted" porque la tarea se procesa de forma asíncrona.

Ejemplo de respuesta:

{
  "task_id": "identificador-generado",
  "status": "QUEUED"
}

Consultar una tarea

GET /api/v1/tasks/{task_id}

La respuesta incluye el estado, el identificador de conversación, el resultado generado y los posibles errores.

Cuando la tarea finaliza correctamente, su estado es "DONE".

Aprobar o rechazar una tarea

POST /api/v1/tasks/{task_id}/approval
Content-Type: application/json

Ejemplo de aprobación:

{
  "decision": "approve",
  "feedback": "Aprobado."
}

Ejemplo de rechazo:

{
  "decision": "reject",
  "feedback": "Requiere revisión."
}

Estados de las tareas

Estado| Descripción
"QUEUED"| Tarea recibida y encolada
"RUNNING"| Procesamiento en ejecución
"WAITING_APPROVAL"| Esperando una decisión humana
"DONE"| Ejecución completada
"FAILED"| Ejecución fallida
"REJECTED"| Tarea rechazada

12. Validación con Pydantic

Los contratos de entrada y salida se definen en:

"src/intelligence_system/models/schemas.py"

Incluyen validaciones para:

- Solicitudes de tareas.
- Identificadores de conversación.
- Filtros de metadatos.
- Resultados y citas documentales.
- Solicitudes de aprobación humana.
- Entradas y salidas de herramientas.
- Decisiones del Supervisor.
- Campos adicionales no permitidos.

Los modelos utilizan validación de tipos, longitudes y restricciones de valores.

13. Comunicación asíncrona

El sistema utiliza interfaces asíncronas para las comunicaciones externas:

- Generación del LLM mediante "ainvoke".
- Operaciones Redis mediante "redis.asyncio".
- Checkpoints mediante "AsyncRedisSaver".
- Operaciones de Pinecone mediante "AsyncPinecone".
- Búsquedas semánticas y léxicas concurrentes.
- Ejecución del grafo mediante "graph.ainvoke".
- Worker basado en "asyncio".
- Extracción PDF mediante "asyncio.to_thread" para evitar bloquear el event loop.

14. Observabilidad con LangSmith

LangSmith permite inspeccionar las ejecuciones del sistema y el flujo interno de los agentes.

La instrumentación permite registrar:

- Ejecuciones de tareas.
- Supervisor y agentes involucrados.
- Llamadas al modelo de lenguaje.
- Tokens de entrada y salida.
- Latencia.
- Identificadores de ejecución y conversación.
- Estimación de costos.

La configuración se realiza mediante:

- "LANGSMITH_TRACING"
- "LANGSMITH_API_KEY"
- "LANGSMITH_PROJECT"

Proyecto de trazabilidad:

"the-intelligence-system"

Plataforma:

https://smith.langchain.com/

15. Pruebas automatizadas

El proyecto utiliza Pytest y Pytest AsyncIO para validar componentes unitarios y flujos de integración.

Ejecutar la suite completa

Desde la raíz del proyecto:

python -m pytest -q

Pruebas de comunicación asíncrona

python -m pytest -q tests/unit/test_llm_client.py tests/unit/test_retrieval.py tests/unit/test_redis_client.py tests/unit/test_rag.py

Pruebas de persistencia e integración Redis

python -m pytest -v tests/integration/test_task_service_redis.py

Pruebas de API, agentes y HITL

python -m pytest -v tests/integration/test_api.py tests/unit/test_agents.py tests/unit/test_agents_hitl.py

Resultados de validación registrados

Validación| Resultado
Suite completa| 44 tests aprobados
Pruebas unitarias seleccionadas| 17 tests aprobados
Persistencia e integración Redis| 4 tests aprobados
API, agentes y HITL| 14 tests aprobados
Verificación de dependencias| Sin dependencias rotas
Compilación Python| Sin errores

La ejecución completa registrada finalizó con 44 pruebas aprobadas y 16 advertencias de deprecación provenientes de "langgraph-checkpoint-redis". Las advertencias no impidieron la aprobación de las pruebas.

16. Evidencias de validación

Las capturas correspondientes a las pruebas y al funcionamiento del sistema están organizadas en "screenshots/".

Requisito| Evidencia
R2 — Comunicación asíncrona| "r2-pruebas-asincronicas.png"
R3 — Persistencia Redis| "r3-persistencia-redis-checkpointer.png"
R4 — Validación Pydantic| "r4-validacion-pydantic.png"
R5 — Observabilidad LangSmith| "r5-langsmith-traza-detalle.png"
R7 — Despliegue local| "r7-despliegue-local.png"
R7 — Tarea real completada| "r7-tarea-real-done.png"
Regresión general| "suite-completa-44-tests.png"

17. Seguridad

- Las credenciales se administran mediante variables de entorno.
- ".env" está excluido del control de versiones.
- ".env.example" contiene la estructura de configuración sin credenciales reales.
- ".venv/" y los archivos temporales están excluidos mediante ".gitignore".
- Los identificadores del índice, namespace y modelo se configuran externamente.

Nunca deben publicarse claves de Groq, Redis, Pinecone ni LangSmith.

18. Cumplimiento de la consigna

Requisito| Descripción| Estado
R1| Integración y refactorización de módulos 6 y 7| Implementado y validado
R2| Comunicación asíncrona con LLMs y bases de datos| Implementado y probado
R3| Persistencia mediante LangGraph Checkpointer| Implementado y probado
R4| Validación Pydantic de API y herramientas| Implementado y probado
R5| Pruebas y trazas en LangSmith| Validado
R6| README y diagrama del grafo de agentes| Documentado
R7| Despliegue local mediante una instrucción| Validado con "bash run.sh"

El despliegue se realiza localmente mediante el script "run.sh". No se requiere CI/CD ni infraestructura de despliegue en la nube para cumplir el alcance solicitado.

---

The Intelligence System
Sistema de inteligencia artificial multiagente con RAG, persistencia, validación y observabilidad.

Repositorio: https://github.com/FSR1991/the-intelligence-system