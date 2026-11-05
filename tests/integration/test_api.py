from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.intelligence_system.api.main import app
from src.intelligence_system.models.schemas import (
    ApprovalDecisionRequest,
    TaskAcceptedResponse,
    TaskCreateRequest,
    TaskResult,
    TaskStatus,
    TaskStatusResponse,
)
from src.intelligence_system.services.task_service import (
    TaskNotFoundError,
    TaskStateError,
)


class FakeRedis:
    async def ping(self):
        return True


class FakeTaskService:
    def __init__(self):
        self.tasks = {}

        now = datetime.now(timezone.utc)

        self.tasks["waiting-approve"] = TaskStatusResponse(
            task_id="waiting-approve",
            status=TaskStatus.WAITING_APPROVAL,
            conversation_id="conversation-approve",
            created_at=now,
            updated_at=now,
        )

        self.tasks["waiting-reject"] = TaskStatusResponse(
            task_id="waiting-reject",
            status=TaskStatus.WAITING_APPROVAL,
            conversation_id="conversation-reject",
            created_at=now,
            updated_at=now,
        )

    async def submit(
        self,
        request: TaskCreateRequest,
        requires_approval: bool = False,
    ):
        task_id = "api-generated-task"
        now = datetime.now(timezone.utc)

        self.tasks[task_id] = TaskStatusResponse(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            conversation_id="api-test-conversation",
            created_at=now,
            updated_at=now,
        )

        return TaskAcceptedResponse(
            task_id=task_id,
            status=TaskStatus.QUEUED,
        )

    async def get_status(self, task_id: str):
        task = self.tasks.get(task_id)

        if task is None:
            raise TaskNotFoundError(
                f"No existe la tarea {task_id}."
            )

        return task

    async def submit_approval(
        self,
        task_id: str,
        decision: ApprovalDecisionRequest,
    ):
        task = self.tasks.get(task_id)

        if task is None:
            raise TaskNotFoundError(
                f"No existe la tarea {task_id}."
            )

        if task.status != TaskStatus.WAITING_APPROVAL:
            raise TaskStateError(
                "La tarea no está esperando una aprobación humana."
            )

        new_status = (
            TaskStatus.DONE
            if decision.decision == "approve"
            else TaskStatus.REJECTED
        )

        now = datetime.now(timezone.utc)

        result = (
            TaskResult(
                answer="Respuesta de prueba aprobada.",
                sources=[],
                input_tokens=10,
                output_tokens=5,
            )
            if new_status == TaskStatus.DONE
            else None
        )

        updated = TaskStatusResponse(
            task_id=task_id,
            status=new_status,
            conversation_id=task.conversation_id,
            created_at=task.created_at,
            updated_at=now,
            result=result,
        )

        self.tasks[task_id] = updated

        return updated


@pytest.fixture
def client():
    app.state.settings = SimpleNamespace(
        app_name="The Intelligence System"
    )
    app.state.redis_client = FakeRedis()
    app.state.task_service = FakeTaskService()

    # No se ejecuta el lifespan: evita conectar Redis real
    # en estas pruebas aisladas de la capa HTTP.
    test_client = TestClient(app)

    yield test_client

    test_client.close()


def test_root_returns_api_information(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "The Intelligence System"


def test_health_returns_ok(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_submit_task_returns_202(client):
    response = client.post(
        "/api/v1/tasks",
        json={
            "query": "Prueba automatizada de creación.",
            "metadata_filter": {},
        },
    )

    assert response.status_code == 202
    assert response.json()["task_id"] == "api-generated-task"
    assert response.json()["status"] == "QUEUED"


def test_get_task_returns_persisted_status(client):
    response = client.get(
        "/api/v1/tasks/waiting-approve"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "WAITING_APPROVAL"


def test_approve_task_returns_done(client):
    response = client.post(
        "/api/v1/tasks/waiting-approve/approval",
        json={
            "decision": "approve",
            "feedback": "Prueba automatizada de aprobación.",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "DONE"
    assert response.json()["result"]["answer"] == (
        "Respuesta de prueba aprobada."
    )


def test_reject_task_returns_rejected(client):
    response = client.post(
        "/api/v1/tasks/waiting-reject/approval",
        json={
            "decision": "reject",
            "feedback": "Prueba automatizada de rechazo.",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "REJECTED"
    assert response.json()["result"] is None


def test_missing_task_returns_404(client):
    response = client.get(
        "/api/v1/tasks/task-that-does-not-exist"
    )

    assert response.status_code == 404


def test_approval_of_final_task_returns_409(client):
    response = client.post(
        "/api/v1/tasks/waiting-reject/approval",
        json={
            "decision": "reject",
            "feedback": "Primera decisión.",
        },
    )

    assert response.status_code == 200

    response = client.post(
        "/api/v1/tasks/waiting-reject/approval",
        json={
            "decision": "approve",
            "feedback": "Decisión posterior inválida.",
        },
    )

    assert response.status_code == 409


def test_invalid_task_payload_returns_422(client):
    response = client.post(
        "/api/v1/tasks",
        json={
            "query": "",
            "metadata_filter": {},
        },
    )

    assert response.status_code == 422


def test_invalid_approval_decision_returns_422(client):
    response = client.post(
        "/api/v1/tasks/waiting-approve/approval",
        json={
            "decision": "maybe",
            "feedback": "Decisión no permitida.",
        },
    )

    assert response.status_code == 422