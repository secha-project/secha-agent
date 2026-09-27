"""Live smoke checks: python -m scripts.check_tuni_tools (uses .env)."""

import asyncio

from pydantic_ai import Agent, ModelSettings, UsageLimits
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from src.agent.config import AgentConfig


async def main() -> None:
    model = AgentConfig(
        model_provider="tuni", model_name="Qwen3.8-27B-AIC"
    ).create_model()
    agent = Agent(
        model,
        model_settings=ModelSettings(temperature=0, max_tokens=1024, timeout=90),
    )
    executed: list[tuple[int, int]] = []

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        """Add two integers."""
        executed.append((a, b))
        return a + b

    prompt = "Call add with a=2 and b=3. After getting the result, reply with only the result."
    for streaming in (False, True):
        executed.clear()
        if streaming:
            async with agent.run_stream(
                prompt, usage_limits=UsageLimits(request_limit=3)
            ) as result:
                output = await result.get_output()
                messages = result.all_messages()
        else:
            result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=3))
            output = result.output
            messages = result.all_messages()

        calls = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolCallPart)
        ]
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert executed == [(2, 3)], executed
        assert len(calls) == len(returns) == 1, (calls, returns)
        assert calls[0].tool_name == returns[0].tool_name == "add"
        assert calls[0].args_as_dict() == {"a": 2, "b": 3}
        assert calls[0].tool_call_id == returns[0].tool_call_id
        assert returns[0].content == 5
        assert output.strip() == "5", output
        print(f"PASS streaming={streaming}: add(2, 3) -> 5 -> {output!r}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
