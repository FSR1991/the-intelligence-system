# The Intelligence System
## Especificación oficial de la entrega final

### 1. Objetivo

The Intelligence System es un sistema de inteligencia artificial completo que integra recuperación aumentada mediante RAG, agentes coordinados por un Supervisor, API profesional y trazabilidad de las ejecuciones.

El sistema debe permitir resolver tareas complejas que requieran conocimiento específico, razonamiento multipaso y acceso a información documentada.

### 2. Arquitectura

La arquitectura integra:

- Python 3.12 o superior.
- FastAPI para la interfaz HTTP.
- LangGraph para la coordinación multiagente.
- Supervisor para delegar tareas.
- RAG para recuperar conocimiento específico.
- Pinecone para almacenamiento y recuperación vectorial.
- Redis para persistencia y administración de estados.
- Pydantic para validación.
- LangSmith o Arize Phoenix para observabilidad.

### 3. Comunicación asíncrona

Las comunicaciones con los modelos de lenguaje y las bases de datos deben utilizar async y await.

No deben mezclarse llamadas bloqueantes de tipo requests dentro de funciones async def.

Se deben utilizar clientes asíncronos y herramientas compatibles.

### 4. Recuperación aumentada mediante RAG

RAG significa Retrieval-Augmented Generation.

Permite que un modelo de lenguaje consulte una base de conocimiento específica antes de elaborar una respuesta.

La recuperación híbrida combina búsqueda semántica y búsqueda léxica.

La búsqueda semántica permite encontrar información relacionada por significado.

La búsqueda léxica identifica coincidencias entre términos y palabras.

Los resultados pueden combinarse mediante Reciprocal Rank Fusion (RRF).

La recuperación de información relevante reduce la necesidad de que el modelo responda exclusivamente a partir de su conocimiento previo.

### 5. Sistema multiagente

LangGraph permite coordinar agentes mediante un Supervisor.

El Supervisor analiza las solicitudes y determina qué agente debe intervenir.

Los agentes pueden ejecutar tareas especializadas, recuperar información y elaborar respuestas.

Las operaciones críticas deben incorporar Human-in-the-Loop (HITL), permitiendo solicitar aprobación humana antes de continuar.

### 6. Persistencia

LangGraph debe utilizar un Checkpointer para conservar el estado de las conversaciones.

Redis permite mantener información de las tareas y los checkpoints.

La persistencia debe permitir recuperar el estado después de reiniciar el servidor o los procesos correspondientes.

### 7. Validación

Pydantic debe validar los datos de entrada y salida de la API, además de los argumentos y resultados de las herramientas utilizadas por los agentes.

### 8. Observabilidad

El sistema debe registrar trazas de las ejecuciones.

La plataforma de observabilidad debe permitir inspeccionar las operaciones de los agentes, tiempos de ejecución y errores.

También deben registrarse los tokens utilizados y los costos estimados de las llamadas a los modelos.

### 9. Pruebas

La entrega debe contener al menos cinco pruebas del sistema.

Se debe comprobar el funcionamiento de los componentes y verificar que las trazas se visualicen correctamente en LangSmith o Arize Phoenix.

### 10. Documentación

El repositorio debe incluir un README que explique la arquitectura, configuración, ejecución y funcionamiento del proyecto.

Debe incluir un diagrama de flujo del grafo de agentes.

### 11. Despliegue local

El sistema debe poder levantarse con una sola instrucción.

Se permite Docker Compose o un script de ejecución local.

No se exige CI/CD ni infraestructura en la nube.

### 12. Errores que deben evitarse

- No incorporar identificadores de índices ni modelos directamente en el código.
- Utilizar variables de entorno para la configuración.
- No bloquear operaciones asíncronas.
- No entregar una arquitectura sin trazabilidad.
- No presentar resultados simulados como integraciones reales.
-