import re
from types import SimpleNamespace
from typing import Any, cast

import src.agent.main as main_module
from src.agent.config import AgentConfig
from src.agent.query_results import QueryResultRegistry


def test_query_result_registry_is_run_scoped() -> None:
    registry = QueryResultRegistry()
    first = object()
    second = object()

    first_ref = registry.create(cast(Any, first))
    second_ref = registry.create(cast(Any, second))

    assert re.fullmatch(r"df-[0-9a-f]{32}", first_ref)
    assert re.fullmatch(r"df-[0-9a-f]{32}", second_ref)
    assert first_ref != second_ref
    assert registry.resolve(first_ref) is first
    assert registry.resolve(second_ref) is second


def test_run_uses_run_scoped_result_registry(
    monkeypatch: Any,
) -> None:
    config = cast(
        AgentConfig,
        SimpleNamespace(
            catalog_url="https://catalog.example",
            catalog_token="token",
            catalog_name="analytics",
            schema_name="bronze",
            table_column_page_size=20,
            temperature=0.0,
            request_limit=3,
        ),
    )
    spark = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(main_module, "get_spark_session", lambda _config: spark)
    monkeypatch.setattr(main_module, "create_model", lambda _config: "model")

    def run_sync(prompt: str, **kwargs: Any) -> SimpleNamespace:
        captured["prompt"] = prompt
        captured.update(kwargs)
        return SimpleNamespace(output="answer")

    monkeypatch.setattr(main_module.agent, "run_sync", run_sync)

    assert main_module.run("question", config) == "answer"
    assert captured["prompt"] == "question"
    assert captured["deps"].spark is spark
    assert isinstance(
        captured["deps"].result_registry,
        QueryResultRegistry,
    )


def test_main_accepts_a_prompt_argument(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setattr(main_module, "run", lambda prompt: f"answer to {prompt}")

    main_module.main(["question"])

    assert capsys.readouterr().out == "answer to question\n"
