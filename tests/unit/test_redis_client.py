import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from src.intelligence_system.config.settings import Settings
from src.intelligence_system.persistence.redis_client import (
    AsyncRedisClient,
    RedisConfigurationError,
    RedisConnectionFailure,
    RedisDataError,
)


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.closed = False

    async def ping(self):
        return True

    async def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        return int(self.data.pop(key, None) is not None)

    async def aclose(self):
        self.closed = True


class BrokenRedis(FakeRedis):
    async def ping(self):
        raise RedisConnectionError("connection refused")


@pytest.mark.asyncio
async def test_connect_without_url_fails_clearly():
    client = AsyncRedisClient(settings=Settings(redis_url=None))

    with pytest.raises(
        RedisConfigurationError,
        match="REDIS_URL",
    ):
        await client.connect()


@pytest.mark.asyncio
async def test_connect_and_ping():
    client = AsyncRedisClient(
        settings=Settings(redis_url=None),
        client=FakeRedis(),
    )

    await client.connect()
    assert await client.ping() is True


@pytest.mark.asyncio
async def test_json_round_trip_and_delete():
    client = AsyncRedisClient(
        settings=Settings(redis_url=None),
        client=FakeRedis(),
    )

    value = {"status": "QUEUED", "attempts": 2}

    assert await client.set_json("task:1", value, ex=60) is True
    assert await client.get_json("task:1") == value
    assert await client.delete("task:1") == 1
    assert await client.get_json("task:1") is None


@pytest.mark.asyncio
async def test_invalid_json_value_is_rejected():
    client = AsyncRedisClient(
        settings=Settings(redis_url=None),
        client=FakeRedis(),
    )

    with pytest.raises(RedisDataError):
        await client.set_json("task:2", {"invalid": object()})


@pytest.mark.asyncio
async def test_connection_error_is_translated():
    client = AsyncRedisClient(
        settings=Settings(redis_url=None),
        client=BrokenRedis(),
    )

    with pytest.raises(RedisConnectionFailure):
        await client.connect()


@pytest.mark.asyncio
async def test_close_releases_client():
    fake = FakeRedis()
    client = AsyncRedisClient(
        settings=Settings(redis_url=None),
        client=fake,
    )

    await client.close()

    assert fake.closed is True