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
default. Accounts are provisioned by an administrator. Login sessions are kept for
seven days in an HTTP-only browser cookie. If the app is served over HTTPS,
set `SECHA_AUTH_SECURE_COOKIE=true` so the cookie is only sent over HTTPS.

### Account administration

Public signup is disabled by default; the existing signup flow is retained.
To enable it later, start the server with `SECHA_ALLOW_SIGNUP=true` in its
environment (for example, `SECHA_ALLOW_SIGNUP=true uv run streamlit run`).

On the server, from the project directory, grant your account admin access:

```bash
uv run python -m src.app.admin YOUR_USERNAME
```

If that username does not exist, the command prompts for a password without
echoing it. Existing accounts keep their passwords. Use `--db PATH` or
`SECHA_CHAT_HISTORY_DB` when the app uses a custom database; the command must
target the same database as the running app.

Sign in and open **Users** in the sidebar to create accounts. Assign a unique
temporary password and share it privately with the recipient. They must choose
a different password before accessing the app. Changing the password revokes
their previous login sessions. Admin-created accounts are regular users;
granting admin access requires the local command above.

Existing accounts and conversations are preserved, and existing users remain
regular users until explicitly promoted. Disabling signup does not revoke
accounts that were already created. The database gains the account flags
automatically on startup; restart the server after upgrading.

## Agent

The agent, built with Pydantic AI, can answer data analytics questions by interacting with the Unity Catalog through the REST API and SparkSQL using the following tools:

- `browse_tables`
- `find_columns`
- `describe_table`
- `query_data`
- `inspect_table_output`
- `display_table_output`

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

Interrupted runs can be resumed in place from their results directory. To rerun
selected tasks after changing their wording or expected artifacts, add `--redo`.
This copies the checkpoint and artifacts to a new timestamped run directory,
leaves the original run untouched, replaces all repetitions of the named tasks,
and generates a fresh report:

```bash
uv run python -m src.validation.run_benchmark \
  --config openai-mini \
  --resume benchmark-results/power-quality-monitoring/openai-mini_YYYY-MM-DD_HH-MM \
  --redo task_6 task_8
```

## Telemetry

All responses, tool use and requests can be tracked with OpenTelemetry locally, set up the telemetry backend with:

```bash
phoenix serve
```

You can access the dashboard at `http://localhost:6006`.
