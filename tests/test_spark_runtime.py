import asyncio
from threading import Event
from typing import Any, cast

import pytest

from src.agent.spark_runtime import SparkActionTimeoutError, run_spark_with_timeout


def test_spark_timeout_interrupts_only_its_own_action(monkeypatch):
    released = Event()
    tags = []
    interrupted = []
    removed = []
    deadlines = []
    wait_for = asyncio.wait_for

    class Spark:
        def addTag(self, tag):
            tags.append(tag)

        def removeTag(self, tag):
            removed.append(tag)

        def interruptTag(self, tag):
            interrupted.append(tag)
            released.set()

    async def short_deadline(action, timeout):
        deadlines.append(timeout)
        return await wait_for(action, timeout=0.02)

    monkeypatch.setattr("src.agent.spark_runtime.asyncio.wait_for", short_deadline)
    with pytest.raises(SparkActionTimeoutError, match="after 120 seconds"):
        asyncio.run(
            run_spark_with_timeout(cast(Any, Spark()), lambda: released.wait(5))
        )

    assert deadlines == [120]
    assert len(tags) == 1
    assert tags == interrupted == removed
