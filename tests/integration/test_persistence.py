from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_conversation_and_task_survive_process_restart():
    thread_id = "process-restart-" + uuid.uuid4().hex
    expected_value = "conversation-restored-after-process-restart"
    task_key = f"intelligence:tasks:restart-test:{thread_id}"

    writer_script = """
import asyncio
import sys

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from src.intelligence_system.persistence.checkpointer import RedisCheckpointer
from src.intelligence_system.persistence.redis_client import AsyncRedisClient


thread_id = sys.argv[1]
expected = sys.argv[2]
task_key = sys.argv[3]


class State(TypedDict):
    value: str


async def persist_node(state: State):
    return {"value": state["value"]}


async def main():
    config = {"configurable": {"thread_id": thread_id}}

    builder = StateGraph(State)
    builder.add_node("persist", persist_node)
    builder.add_edge(START, "persist")
    builder.add_edge("persist", END)

    async with RedisCheckpointer() as saver:
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke({"value": expected}, config=config)

        saved = await saver.aget_tuple(config)
        assert saved is not None

    redis = AsyncRedisClient()

    try:
        await redis.connect()
        await redis.set_json(
            task_key,
            {
                "task_id": thread_id,
                "status": "WAITING_APPROVAL",
                "conversation_id": thread_id,
            },
            ex=300,
        )
    finally:
        await redis.close()

    print("PROCESS_WRITE_OK")


asyncio.run(main())
"""

    reader_script = """
import asyncio
import sys

from src.intelligence_system.persistence.checkpointer import RedisCheckpointer
from src.intelligence_system.persistence.redis_client import AsyncRedisClient


thread_id = sys.argv[1]
expected = sys.argv[2]
task_key = sys.argv[3]


async def main():
    config = {"configurable": {"thread_id": thread_id}}

    async with RedisCheckpointer() as saver:
        saved = await saver.aget_tuple(config)

        assert saved is not None

        actual = saved.checkpoint.get(
            "channel_values", {}
        ).get("value")

        assert actual == expected, (
            f"Conversación recuperada incorrectamente: {actual!r}"
        )

    redis = AsyncRedisClient()

    try:
        await redis.connect()

        task = await redis.get_json(task_key)

        assert task is not None
        assert task["task_id"] == thread_id
        assert task["status"] == "WAITING_APPROVAL"

        print("CONVERSATION_RECOVERY_OK")
        print("TASK_STATE_RECOVERY_OK")

        await redis.delete(task_key)

    finally:
        await redis.close()

        async with RedisCheckpointer() as saver:
            await saver.adelete_thread(thread_id)

    print("PROCESS_RESTART_PERSISTENCE_OK")


asyncio.run(main())
"""

    writer = subprocess.run(
        [
            sys.executable,
            "-c",
            writer_script,
            thread_id,
            expected_value,
            task_key,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert writer.returncode == 0, (
        "Falló el proceso que guardaba los datos.\n"
        f"Salida:\n{writer.stdout}\n"
        f"Error:\n{writer.stderr}"
    )

    assert "PROCESS_WRITE_OK" in writer.stdout

    reader = subprocess.run(
        [
            sys.executable,
            "-c",
            reader_script,
            thread_id,
            expected_value,
            task_key,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert reader.returncode == 0, (
        "Falló el proceso que recuperaba los datos.\n"
        f"Salida:\n{reader.stdout}\n"
        f"Error:\n{reader.stderr}"
    )

    assert "CONVERSATION_RECOVERY_OK" in reader.stdout
    assert "TASK_STATE_RECOVERY_OK" in reader.stdout
    assert "PROCESS_RESTART_PERSISTENCE_OK" in reader.stdout