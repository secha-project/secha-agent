# SECHA agent

SECHA-agent is an AI data assistant for SECHA's data platform. It consists of three main parts: an LLM agent, an evaluation suite and a user interface.

See the [C4 architecture diagrams](docs/c4-architecture.md) for the system
context, containers, and conversational request path.

## Run the app

Configure the required catalog, Spark Connect, selected model provider, and
OpenTelemetry values in `.env`, then start the Streamlit app:

```bash
uv run streamlit run
```

`streamlit_app.py` is the single ASGI entry point. Its conventional filename
lets Streamlit find it without a path argument. It mounts the Streamlit UI from
`src/app/application.py` and the authentication routes from
`src/app/routes/auth.py` on the same listener.

The app uses OpenAI as its model provider and stores chat history locally by
default. Users must create an account in the app. Login sessions are kept for
seven days in an HTTP-only browser cookie. If the app is served over HTTPS,
set `SECHA_AUTH_SECURE_COOKIE=true` so the cookie is only sent over HTTPS.

## Agent

The agent, built with Pydantic AI, can answer data analytics questions by interacting with the Unity Catalog through the REST API and SparkSQL using the following tools:

- `browse_tables`
- `find_columns`
- `describe_table`
- `query_data`
- `inspect_table_output`
- `display_table_output`

The Boring Semantic Layer capability adds governed semantic-model
discovery, metric queries, and period comparisons. It uses an Ibis PySpark
backend bound to the same authenticated Spark Connect session, and its results
enter the same table artifact pipeline as raw Spark SQL.

Define models in `src/agent/semantic_models`. Its model factory receives the
Ibis backend bound to the application Spark session, which remains the sole
execution and authorization boundary. Resolve physical tables through that
backend and return base or composed BSL models from the factory.

Configure the catalog, Spark Connect, model provider, and OpenTelemetry values
in `.env`. Keep credentials for every provider in this file; provider and
capability selection belong to the app or benchmark configuration:

```dotenv
OPENAI_API_KEY=...
TUNI_INFERENCE_URL=...
TUNI_INFERENCE_API_KEY=...
```

Optional model defaults use `OPENAI_MODEL_NAME` and
`TUNI_INFERENCE_MODEL_NAME`. Select a provider per benchmark with
`agent.model_provider`; changing benchmark providers never requires editing
`.env`.
The app and one-off agent use the `AgentConfig` capability default: catalog,
query, semantic layer, table outputs, and visualization. Pass `enabled_capabilities`
explicitly when constructing `AgentConfig`, or set it in a validation run
configuration, to use a different profile.

Then run a one-off prompt with:

```bash
uv run python -m src.agent.main "What units does the active energy column use?"
```

## Validation

The validation suite, built with Pydantic Evals, aims to validate the quality of the agent's answers accross different themes and knowledge domains. A run is configured with a yaml file such as `src/validation/run_configs/openai-mini.yaml`, that points to a dataset file (containing question-answer pairs). Select the agent profile under its `agent` mapping:

```yaml
agent:
  model_provider: openai
  model_name: gpt-5.4-mini
  enabled_capabilities: [catalog, query, table_outputs, visualization]
```

That example config file evaluates the agent in the small dataset in `src/validation/benchmark/power-quality-monitoring/dataset.yaml`, you can run it with:

```bash
uv run python -m src.validation.run_benchmark --config openai-mini
```

## Telemetry

All responses, tool use and requests can be tracked with OpenTelemetry locally, set up the telemetry backend with:

```bash
phoenix serve
```

You can access the dashboard at `http://localhost:6006`.
