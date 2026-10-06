# SECHA agent

SECHA-agent is an AI data assistant for SECHA's data platform. It consists of three main parts: an LLM agent, an evaluation suite and a user interface.

See the [C4 architecture diagrams](docs/c4-architecture.md) for the system
context, containers, and conversational request path.

See [prompt optimization](docs/prompt-optimization.md) for the offline GEPA runner,
named prompt overrides, and experiment setup.

## Installation profiles

The base dependencies run the agent and benchmarks with a remote Spark Connect
server. They use `pyspark-client` without the full Spark distribution or JVM JARs,
and `pydantic-ai-slim` with OpenAI-compatible providers, tracing, and retries.
OpenAI, ZAI, and TUNI all use this provider interface.

For a benchmark-only VM, install from the repository root:

```bash
uv sync --locked --no-dev --no-cache
uv run --locked --no-dev --no-cache python -m src.validation.run_benchmark \
  --config gpt-6-sol__baseline --dry-run
```

Remove `--dry-run` to execute the benchmark. Keep `--no-dev` on subsequent
`uv run` commands to avoid installing development tools. `--no-cache` avoids
retaining package downloads and builds; it does not remove an existing cache.

To migrate an existing installation on a VM with limited disk space:

```bash
uv cache clean
uv sync --locked --no-dev --no-cache --reinstall-package pyspark-client
df -h .
du -sh .venv
```

Use these commands after copying the updated `pyproject.toml` and `uv.lock` to
the VM. Sync removes dependencies that are no longer selected. Reinstalling
`pyspark-client` restores its shared `pyspark` module files after full `pyspark`
is removed. Cache cleanup preserves installed environments with uv's normal
copy/clone/hardlink modes; custom symlink installs must be recreated. Check
`df` for actual space reclaimed because hard-linked files can share disk blocks.
See [uv cache documentation](https://docs.astral.sh/uv/concepts/cache/).

Optional dependency groups are selected with `--group`:

| Group | Purpose |
| --- | --- |
| `app` | Streamlit UI, authentication, and HTTP routes |
| `plots` | Publication figures using plotnine |
| `dev` | pytest, Ruff, and ty; installed by default unless `--no-dev` is set |
| `optimization` | GEPA prompt optimization |

For a complete development environment and test suite:

```bash
uv run --group app --group plots --group optimization pytest
```

## Run the app

Configure the required catalog, Spark Connect, selected model provider, and
OpenTelemetry values in `.env`, then start the Streamlit app:

```bash
uv run --group app streamlit run
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
environment (for example, `SECHA_ALLOW_SIGNUP=true uv run --group app streamlit run`).

On the server, from the project directory, grant your account admin access:

```bash
uv run --group app python -m src.app.admin YOUR_USERNAME
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
- `visualize_table_output`
- `display_chart_output`

Configure the catalog, Spark Connect, model provider, and OpenTelemetry values
in `.env`. Keep credentials for every provider in this file; provider and
capability selection belong to the app or benchmark configuration:

```dotenv
OPENAI_API_KEY=...
ZAI_API_KEY=...
TUNI_INFERENCE_URL=...
TUNI_INFERENCE_API_KEY=...
```

Optional model defaults use `OPENAI_MODEL_NAME`, `ZAI_MODEL_NAME`, and
`TUNI_INFERENCE_MODEL_NAME`. Select `openai`, `zai`, or `tuni` per benchmark with
`agent.model_provider`; changing benchmark providers never requires editing
`.env`.
The app and one-off agent use the `AgentConfig` capability default: catalog,
query, semantic layer, table outputs, and visualization. Pass `enabled_capabilities`
explicitly when constructing `AgentConfig`, or set it in a validation run
configuration, to use a different profile.

Visualization uses the same preview-before-publication pattern as tables.
`visualize_table_output` returns a private PNG preview and a chart `output_ref`.
The agent inspects that image, revises the chart if needed, then calls
`display_chart_output` to publish the selected draft. Drafts stay out of the
displayed artifacts and benchmark outputs. Chart references last for the current
agent run; a later turn can recreate a chart from its saved table output.

This workflow requires an image-capable model. The z.ai provider now defaults to
`glm-5.3-flash`; `ZAI_MODEL_NAME` or an explicit `model_name` still overrides it.
The app's default provider remains `tuni`. All four `glm-5.3__*` benchmark
configs now select `glm-5.3-flash` for image review. The PNG uses a fixed viewport;
the published interactive chart still adapts to the chat width.

The synthetic review probe renders a draft with deliberately incorrect axis
units. It needs no Spark or catalog connection. The live option uses configured
z.ai credentials and asks the agent to inspect, correct, and publish the chart:

```bash
uv run python -m scripts.probe_visualization_review
uv run python -m scripts.probe_visualization_review --live
```

The probe saves draft and published chart files plus a result summary under
`output/visualization-review-probe`. The live run is capped at eight model
requests, 40,000 total tokens, and 30 minutes. Its request timeout defaults to
600 seconds; use `--request-timeout` and `--run-timeout` to override these budgets.

Use `--provider openai --model gpt-6-sol` (or another configured provider/model)
to test the same flow elsewhere. The default `--case units` contains incorrect
axis units; `--case control` starts with a correct chart that should be published
unchanged. Each image contains a random reference code absent from the text
prompt. The probe verifies that the model reads that code and the original axis
label, receives the preview before publication, preserves the data and title,
and publishes exactly one appropriate chart. Results include image hashes and
tool-call order without storing model reasoning or base64 image payloads.

For one real visualization benchmark question, including Spark/catalog access
and the normal visualization evaluators, use a fresh output directory:

```bash
uv run python -m scripts.probe_visualization_benchmark \
  --config glm-5.3__baseline --task task_26 \
  --output output/visualization-benchmark-smoke/glm-flash
```

This runs one repetition of the selected case without editing the benchmark
YAMLs, with a 30-minute deadline for the agent run. It saves the checkpoint,
report, chart previews, final chart, and image handoff evidence. `result.json`
separates image-review verification from the benchmark assertions, so
successful publication does not hide a wrong chart.

Normal agent runs allow six HTTP attempts within a 1,800-second retry budget
per logical model request. This includes pacing, backoff, and non-streaming
response reads; it is separate from the 600-second HTTP inactivity timeout.
The longer budget gives slow Qwen requests room to complete and retry without
being cut off by the former 300-second deadline. Set
`model_request_retry_budget_seconds` and `model_request_max_attempts` explicitly
to choose a different policy. Benchmark configs retain their unlimited retry
settings and 50-request agent limit. Spark actions allow 120 seconds before
interrupting the tagged query. The retry policy also recognizes the TUNI/Open
WebUI gateway's exact HTTP 400 `Server Connection Error` response as transient;
other invalid requests still fail immediately.

Then run a one-off prompt with:

```bash
uv run python -m src.agent.main "What units does the active energy column use?"
```

## Validation

The complete 16-config experiment matrix, naming conventions, and dry-run loop
are documented in [run_configs/README.md](src/validation/run_configs/README.md).
The optional `few_shot` capability adds examples to tool descriptions;
it can be combined independently with `semantic_layer`.

The validation suite, built with Pydantic Evals, validates the agent's answers
across different themes and knowledge domains. A run is configured with a YAML
file such as `src/validation/run_configs/gpt-6-sol__baseline.yaml`, which points
to a dataset containing question-answer pairs. Select the agent profile under
its `agent` mapping:

```yaml
agent:
  model_provider: openai
  model_name: gpt-6-sol
  enabled_capabilities: [catalog, query, table_outputs, visualization]
```

That config evaluates `src/validation/benchmark/power-quality-monitoring/dataset.yaml`:

```bash
uv run --locked --no-dev python -m src.validation.run_benchmark --config gpt-6-sol__baseline
```

Check a configuration locally before running it:

```bash
uv run --locked --no-dev python -m src.validation.run_benchmark --config gpt-6-sol__baseline --dry-run
```

This validates provider settings, prompt construction, the dataset, and referenced
artifact files without calling models, Spark, or evaluators, or writing benchmark
results. It does not verify remote credentials or model availability. Dry runs
cannot be combined with `--resume` or `--redo`.

Interrupted runs can be resumed in place from their results directory. To rerun
selected tasks after changing their wording or expected artifacts, add `--redo`.
This copies the checkpoint and artifacts to a new timestamped run directory,
leaves the original run untouched, replaces all repetitions of the named tasks,
and generates a fresh report:

```bash
uv run --locked --no-dev python -m src.validation.run_benchmark \
  --config gpt-6-sol__baseline \
  --resume benchmark-results/power-quality-monitoring/gpt-6-sol__baseline_YYYY-MM-DD_HH-MM \
  --redo task_6 task_8
```

To run every configuration in order, use the bash launcher:

```bash
bash scripts/run_benchmarks.sh --dry-run  # Validate all configs locally
bash scripts/run_benchmarks.sh            # Start, or continue after interruption
```

Each config keeps its configured tasks and repetitions (currently 55 × 3 = 165
attempts). The launcher saves the config order, YAML snapshots, per-config run
state, and progress logs under `benchmark-results/batch-logs/`. After a crash or
Ctrl-C, repeat the second command: completed configs are skipped, unfinished
attempts resume in the same run directory, and the remaining configs follow.
No separate Python resume command or manual list of completed configs is needed.
Recorded execution failures count as completed attempts, as in the standalone
runner; resuming does not silently redo them.

The first invocation starts fresh, regardless of older benchmark reports. Once
its queue is finished, repeating the command does nothing. Use `--new` for a new
full benchmark, or `--new --skip CONFIG` to omit selected configs. To continue an
older saved queue, pass `--resume benchmark-results/batch-logs/SAVED_DIR`.
After starting with `--new`, omit that flag when resuming. Changed YAML files are
rejected on resume; restore them or start a new queue with `--new`. Dry runs
validate the current configs without changing saved progress or creating logs.

Start the log viewer separately, before or during the benchmark:

```bash
python3 scripts/serve_benchmark_logs.py
```

Open `http://localhost:8501`. It follows the launcher's saved queue, shows the
number of completed configs and the latest config's progress, and switches when
a config starts or resumes. Keep it running across interruptions and `--new`
runs; no server restart or new log path is needed. The page refreshes every five
seconds; checkpoint counts are logged once per minute. A positional log path
pins the viewer to one file instead.

Report cost and usage fields retain the same schema and represent **subtotals of
recorded usage**. In `results.overall.usage` and each category's `summary`,
`input_tokens`, `output_tokens`, `total_tokens`, and `estimated_cost_usd` cover
attempts with recorded input or output token usage. A completed attempt counts
even if its correctness assertions fail. Execution failures without metrics and
other attempts missing token usage are excluded from these subtotals.

Compare `results.overall.usage.attempts_with_usage` with
`results.overall.attempts`, or a category's `summary.attempts_with_usage` with its
`summary.attempts`, to determine coverage. A smaller usage count means the cost
excludes some attempts; it does not mean those attempts were free. If no attempts
have token usage, the input/output/total token fields and cost are `null`. Cost
also remains `null` if pricing is unavailable; `pricing_source` explains the
pricing basis. Requests, tool calls, and additional token metrics sum their
available recorded values and may likewise be incomplete. Use Phoenix traces
to recover usage absent from the saved case results when a full-run cost is needed.

To generate the grouped ablation pass-rate figure, name related run configs
`ROOT__ABLATION.yaml`, for example `gpt-6-sol__baseline.yaml` and
`gpt-6-sol__semantic_layer.yaml`. The suffixes are discovered from completed
reports, so additional ablations need no plotting-code changes. All roots must
have the same suffixes, dataset, and tasks. Generate the PDF with:

```bash
uv run --group plots python -m scripts.generate_ablation_figure -m 20
```

Use `--root gpt-6-sol` to limit the figure to one run family. The default
output is `output/pdf/ablation_pass_rates.pdf`.

For the two stacked pass@k and pass^k plots, select one ablation suffix; the
figure includes every completed run with that suffix and keeps scores on their
reported 0-1 scale. Each plot has its own y-axis label, while they share the
bottom k axis and one model legend:

```bash
uv run --group plots python -m scripts.generate_pass_k_figures --ablation baseline
```

The default output is `output/pdf/pass_k_baseline.pdf`. These curves require
at least two benchmark repetitions.

## Telemetry

All responses, tool use and requests can be tracked with OpenTelemetry locally, set up the telemetry backend with:

```bash
phoenix serve
```

You can access the dashboard at `http://localhost:6006`.

Logfire scrubbing is disabled for app and benchmark runs, preserving original
identifiers, SQL, tool arguments/results, and model messages in new traces.
Previously redacted trace content cannot be recovered from the saved traces.
