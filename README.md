# SECHA agent

SECHA-agent is an AI data assistant for SECHA's data platform. It consists of three main parts: an LLM agent, an evaluation suite and a user interface.

## Agent

The agent, built with Pydantic AI, can answer data analytics questions by interacting with the Unity Catalog through the REST API and SparkSQL using the following tools:

- `browse_tables`
- `find_columns`
- `describe_table`
- `query_data`
- `inspect_table_output`
- `display_table_output`
- `visualize_table_output`
- `display_chart_output`

Configure the catalog, Spark Connect, model provider, and OpenTelemetry values
in `.env`, then run a one-off prompt with:

```bash
uv run python -m src.agent.main "What units does the active energy column use?"
```

## Validation

The validation suite, built with Pydantic Evals, aims to validate the quality of the agent's answers across different themes and knowledge domains. A run is configured with a YAML file such as `src/validation/run_configs/gpt-6-sol__baseline.yaml`, that points to a dataset file (containing question-answer pairs). That example config file evaluates the agent in the dataset in `src/validation/benchmark/power-quality-monitoring/dataset.yaml`, you can run it with:

```bash
uv run python -m src.validation.run_benchmark --config gpt-6-sol__baseline
```

To run all configurations in order, use `bash scripts/run_benchmarks.sh`.
Rerun the same command to continue after an interruption, or add `--new` to start a fresh run.

## Run the app

Configure the required catalog (including `CATALOG_TOKEN_ADMIN`), Spark Connect,
selected model provider, and OpenTelemetry values in `.env`, then start the Streamlit app:

```bash
uv run --group app streamlit run
```

The app uses TUNI as its default model provider and stores chat history locally.
Accounts are provisioned by an administrator. Create an admin account or grant
admin access to an existing account with:

```bash
uv run --group app python -m src.app.admin YOUR_USERNAME
```

Sign in and open **Users** in the sidebar to create accounts.

## Telemetry

All responses, tool use and requests can be tracked with OpenTelemetry locally, set up the telemetry backend with:

```bash
uv run --group telemetry phoenix serve
```

You can access the dashboard at `http://localhost:6006`.
