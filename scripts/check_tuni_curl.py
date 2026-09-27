"""Live curl smoke check: python -m scripts.check_tuni_curl (uses .env)."""

import json
import subprocess
from pathlib import Path

from dotenv import dotenv_values

cfg = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
base = cfg["TUNI_INFERENCE_URL"].rstrip("/")


def request(path, payload=None):
    args = [
        "curl",
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--max-time",
        "90",
        "--header",
        "@-",
        base + path,
    ]
    headers = (
        "Authorization: Bearer "
        + cfg["TUNI_INFERENCE_API_KEY"]
        + "\nContent-Type: application/json\n"
    )
    if payload is not None:
        args += ["--data-raw", json.dumps(payload)]
    result = subprocess.run(
        args, input=headers, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return json.loads(result.stdout)


print("Models:", [m["id"] for m in request("/models")["data"]], flush=True)
payload = {
    "model": "Qwen3.8-27B-AIC",
    "temperature": 0,
    "max_tokens": 1024,
    "messages": [
        {
            "role": "user",
            "content": "Call add with a=2 and b=3. After getting the result, reply with only the result.",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "tool_choice": "auto",
}
response = request("/chat/completions", payload)
msg = response["choices"][0]["message"]
print("First response:", json.dumps(msg), flush=True)
calls = msg.get("tool_calls", [])
assert len(calls) == 1, calls
call = calls[0]
assert call["function"]["name"] == "add"
assert json.loads(call["function"]["arguments"]) == {"a": 2, "b": 3}
payload["messages"] += [
    msg,
    {"role": "tool", "tool_call_id": call["id"], "content": "5"},
]
response = request("/chat/completions", payload)
print("Final response:", json.dumps(response["choices"][0]["message"]), flush=True)
assert response["choices"][0]["message"]["content"].strip() == "5"
print("PASS: curl tool round trip")
