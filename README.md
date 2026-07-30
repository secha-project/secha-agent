# SECHA agent

SECHA-agent is an AI data assistant for SECHA's data platform. It consists of three main parts: an LLM agent, an evaluation suite and a user interface.

## Agent

The agent, built with Pydantic AI, can answer data analytics questions by interacting with the Unity Catalog through the REST API and SparkSQL using the following tools:

- `browse_tables`
- `search_columns`
- `describe_table`
- `query_data`
- `inspect_query_result`
- `display_query_result`

Configure the catalog, Spark Connect, model provider, and OpenTelemetry values
in `.env`, then run a one-off prompt with:

```bash
uv run python -m src.agent.main "What units does the active energy column use?"
```

## Validation

The validation suite, built with Pydantic Evals, aims to validate the quality of the agent's answers accross different themes and knowledge domains. A run is configured with a yaml file such as `src/validation/run_configs/openai-mini.yaml`, that points to a dataset file (containing question-answer pairs). That example config file evaluates the agent in the small dataset in `src/validation/benchmark/power-quality-monitoring/dataset.yaml`, you can run it with:

```bash
uv run python -m src.validation.run_benchmark --config openai-mini
```

## Run the app

Configure the required catalog, Spark Connect, OpenAI, and OpenTelemetry values
in `.env`, then start the Streamlit app:

```bash
uv run python -m streamlit run src/app/main.py
```

The app uses OpenAI as its model provider and stores chat history locally by
default. Users must create an account in the app.

## Telemetry

All responses, tool use and requests can be tracked with OpenTelemetry locally, set up the telemetry backend with:

```bash
PHOENIX_SQL_DATABASE_URL=sqlite:///telemetry.db
phoenix serve
```

You can access the dashboard at `http://localhost:6006`.
