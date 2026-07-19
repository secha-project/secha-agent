# SECHA agent

## Local traces

Install the OpenTelemetry Desktop Viewer on macOS:

```bash
brew install --cask ctrlspice/tap/otel-desktop-viewer
```

Start it with a persistent local DuckDB database:

```bash
otel-desktop-viewer --db ./telemetry.duckdb
```

Then run the agent in another terminal:

```bash
uv run python src/agent/agent.py
```

Open <http://localhost:8000/traces> to inspect the agent run, model requests,
and tool calls. The agent sends OTLP/HTTP traces to `http://localhost:4318` and
does not send them to Pydantic Logfire. Metrics export is disabled because the
local trace viewer does not expose an OTLP metrics endpoint.
