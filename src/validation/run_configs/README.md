# Benchmark ablations

The 16 experiment configs cover four models and every combination of few-shot
tool examples and semantic-layer access. Each config uses the complete
`power-quality-monitoring` dataset and concurrency **4**. Every configuration
uses **3 repetitions**: 55 tasks × 3 = **165 attempts per config**, or **2,640
attempts** across all 16 configs.

Transient model failures are retried on the individual HTTP request, preserving
the current conversation and successful tool calls. Matrix configs keep retrying
until the request succeeds or the run is interrupted: both attempt and retry-time
limits are `null`. Waits grow through 15, 30, 60, 120, 240, then 300 seconds,
plus up to 5 seconds of jitter. A valid `Retry-After` is a minimum, even when it
exceeds 300 seconds. SDK retries are disabled so they do not multiply attempts.
Authentication/invalid-request errors and recognized exhausted-quota errors
are not retried. Invalid tool arguments retain the agent's existing correction
limit; incorrect benchmark answers are not retried to obtain a passing score.

Requests from each run's model client start at least 5 seconds apart (at most
12 starts per minute), including first attempts and retries. Workers using that
client share failure cooldowns. The visualization judge also retries indefinitely
and shares a separate 5-second pacer across judge calls in the process. These
are conservative starting values, not provider quota guarantees: token limits
and other processes can still cause throttling.

Concurrency lets four cases overlap model latency and tool execution. The request
pacer controls their combined traffic: the 5-second interval applies across the
four cases, not independently to each case. During a shared cooldown, in-flight
work can finish, but new requests wait. Provider limits on tokens or simultaneous
requests can still require more spacing or fewer concurrent cases.

Configure this with `agent.model_request_max_attempts`,
`agent.model_request_retry_budget_seconds`, and
`agent.model_request_min_interval_seconds`; all three are recorded in metadata.
For example, a minimum interval of 10 seconds limits that client to 6 request
starts per minute. Outside the matrix, agent defaults remain 6 attempts and a
1,800-second retry budget, with no minimum interval. The HTTP inactivity timeout
is 600 seconds. The retry budget includes pacing, backoff, and non-streaming
response reads, so a shorter budget can end a request before its HTTP timeout.
The exact Open WebUI HTTP 400 `Server Connection Error` response is treated as
transient; other HTTP 400 errors remain permanent. HTTP timeouts are separate
from the retry budget. All matrix configs use `task_max_attempts: 1`; retries
happen within the request, so successful tool work is retained while waiting.

An outage can leave a run waiting indefinitely, with retry warnings in the log.
Ctrl-C interrupts model backoff, leaves the current case pending, and preserves
completed checkpoint entries. Resume using the same config and run directory.
An interrupted case restarts on resume; its in-memory conversation is not saved.
See the [retry guidance](https://developers.openai.com/api/docs/guides/rate-limits).

Text evaluation checks whether every expected value occurs in the answer,
allowing surrounding explanation. Matches preserve identifier boundaries,
ignore case/whitespace/basic Markdown, and consume repeated values separately.
It does not enforce answer-only formatting or interpret negation.

Time boundaries are stated in individual benchmark questions, following the
reference queries. Ordinary clock-time ranges include the start and exclude the
end; date ranges include both named calendar dates. Rolling windows explicitly
state whether the current measurement is included. For aggregates labeled by
their window end, the final window can have an end label equal to the range end:
tasks 46 and 55 explicitly select end labels greater than the starting boundary
and less than or equal to the ending boundary. Task 48 assigns windows ending at
midnight to the preceding UTC date. These wording clarifications preserve the
reference queries and expected answers.

Keep earlier interrupted runs as separate results: their scores use the old
text policy and their metadata uses the old retry/concurrency settings.
`--resume` rejects changed configuration metadata and does not rescore completed
cases. New runs use a new timestamped results directory.

| Config root | Provider | Exact model ID |
| --- | --- | --- |
| `gpt-6-sol` | `openai` | `gpt-6-sol` |
| `gpt-6-luna` | `openai` | `gpt-6-luna` |
| `glm-5.3` | `zai` | `glm-5.3-flash` |
| `qwen3.8-27b` | `tuni` | `Qwen3.8-27B-AIC` |

The `glm-5.3` config root now selects `glm-5.3-flash` for image review, verified
with live visualization probes on 2026-10-06. Earlier GLM-5.3 runs remain separate;
the model change prevents resuming their checkpoints with these configs.
GPT-6 IDs were checked on 2026-09-26 against the official
[GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol) and
[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) documentation.
Qwen retains the existing TUNI deployment ID. These selections supersede the
older model names in `docs/3-methodology.tex`.

Every root has the same four suffixes:

| Suffix | Few-shot tool examples | Semantic layer |
| --- | --- | --- |
| `baseline` | Off | Off |
| `few_shot` | On | Off |
| `semantic_layer` | Off | On |
| `few_shot_semantic_layer` | On | On |

The full filename is `ROOT__SUFFIX.yaml`, for example
`gpt-6-sol__few_shot_semantic_layer.yaml`. The generators split at the final
`__`; keep both factors in one suffix. Matching roots must keep the same model,
dataset, tasks, and suffix set for comparisons. Repetition counts are recorded
per run; use k ≤ 3 when comparing pass@k across this matrix.

All variants enable `catalog`, `query`, `table_outputs`, and `visualization`.
The `few_shot` capability adds two inventory examples to each active tool's
description, using the same examples across all models and semantic settings.
It adds no tools or domain metric definitions. Examples live in
`src/agent/prompts/capabilities/*/tools/*.md`; their instruction header is in
`src/agent/prompts/capabilities/few_shot/instructions.md`. This capability is
recorded in report metadata, so the baseline table excludes few-shot variants.

All configs omit temperature and reasoning effort, using each provider's
defaults. `AgentConfig.temperature` defaults to `None`, so an omitted setting
is not sent in model requests. Explicit temperature overrides remain supported.
The existing model adapter uses the Responses API for GPT-6. Request limit 50
is shared across all configs.

GLM uses Pydantic AI's native `ZaiModel` and Z.ai provider profile, with preserved
thinking across tool calls. Nested tool schemas are inlined with
`InlineDefsJsonSchemaTransformer` so table references are presented directly as
objects. This changes the schema sent to GLM; tool argument validation remains
unchanged. The adapter identifier `zai_inline_defs_v1` is recorded in GLM run
metadata, so earlier GLM checkpoints cannot be resumed under this configuration.

TUNI Qwen's default reasoning was confirmed with small live probes on 2026-09-26;
see [the reasoning check](../../../docs/tuni-qwen-setup.md#reasoning-check-2026-09-26).

From the repository root, validate one config without running the benchmark:

```bash
uv sync --locked --no-dev --no-cache
uv run --locked --no-dev --no-cache python -m src.validation.run_benchmark \
  --config gpt-6-sol__baseline --dry-run
```

Validate all 16 configs without external calls or benchmark result files:

```bash
bash scripts/run_benchmarks.sh --dry-run
```

Run every YAML config sequentially with `bash scripts/run_benchmarks.sh`.
The first invocation starts fresh; after an interruption, repeat the same command
without flags. It skips finished configs, resumes unfinished attempts in the
current config's saved run directory, then continues with the remaining configs.
Each YAML's repetitions and concurrency are preserved (currently 55 tasks × 3
repetitions = 165 attempts per config).

```bash
bash scripts/run_benchmarks.sh                    # Start or continue
bash scripts/run_benchmarks.sh --new              # Start all configs from scratch
bash scripts/run_benchmarks.sh --new --skip glm-5.3__baseline
bash scripts/run_benchmarks.sh --resume benchmark-results/batch-logs/SAVED_DIR
```

The launcher remembers only runs it started, using a saved config order, YAML
snapshots and per-config state beside the logs. Old standalone reports are not
used to infer completion. A finished queue stays finished until `--new` is given;
after starting with `--new`, omit that flag when continuing. `--resume DIR`
selects an older queue and makes it the default for subsequent invocations.
Repeat `--skip CONFIG` to omit configs when creating a queue; saved queues retain
their selection even if new YAML files appear. Changed or missing saved YAMLs
stop the launcher before any benchmark starts. Restore them or use `--new`.

`--dry-run` always validates the current config selection, with optional skips,
without changing saved progress or creating logs. Recorded execution failures
remain completed attempts; use the standalone runner's explicit redo options
when you intend to repeat them. Per-config locks prevent two launchers from
executing the same saved run concurrently and are released when a process exits.

Numbered banners show the active config, start time and log path. Each run keeps
its live progress display when launched in a terminal; completion banners include
elapsed time. Separate per-config logs are saved in `benchmark-results/batch-logs/`.
Logs contain run names and result directories, checkpoint counts once per minute,
and finish/failure/interruption status. Completed counts include execution failures,
which are also counted separately; they are not passing-score counts. Terminal
redraws, full model output, report tables, and repeated warnings are not copied into
these logs. The terminal still displays live progress and diagnostic output.
Standalone runs can opt in with `--progress-log PATH`.
The script stops if a command fails and can be invoked from any directory.
Rerun it to continue; even setup failures before the first attempt have a saved
empty checkpoint. No Python resume command is needed.

To view the latest status log in a browser, run in a separate terminal:

```bash
python3 scripts/serve_benchmark_logs.py
```

Open `http://localhost:8501`. The page refreshes every five seconds and follows
the launcher's `latest-queue` pointer, showing completed-config counts and the
most recently updated log in that queue. It switches as configs start or resume,
and when `--new` or `--resume DIR` selects another queue. Unrelated runs cannot
take over the page. Keep the server running independently across benchmark
interruptions; the launcher does not start or stop it.

Without a saved queue, the viewer discovers the latest standalone benchmark or
optimizer log. Pass a positional log path to pin it to one file, including an
optimizer's `progress.log`. Checkpoint counts update once per minute. Add
`--host 0.0.0.0` to make the viewer reachable from another machine; the viewer has
no authentication. It uses only Python's standard library, shows up to 200 recent
log lines, and does not write HTTP access logs.

The base install includes benchmark dependencies. Streamlit and publication
plotting are optional (`--group app` and `--group plots` respectively). See the
[installation profiles](../../../README.md#installation-profiles) for cache
cleanup and migration from an existing full installation.

Dry runs validate local settings, prompt construction, dataset loading, and
reference-file existence. They do not verify remote model availability,
credentials, or database connectivity. Remove `--dry-run` when ready to execute
a selected config. The plotting scripts accept the suffix as `--ablation`
(for example, `baseline`); the grouped ablation figure discovers all suffixes.

TUNI Qwen has no configured pricing reference. Its scores and token counts are
recorded, but its monetary cost may be unavailable. The cost figure requires a
finite cost for every selected model. Set an explicitly chosen
`agent.pricing: {provider: ..., model_name: ...}` reference consistently across
all four Qwen configs if using a hosted-model cost proxy; that proxy would not
represent measured on-premises cost.

## Usage accounting

Use the locked Pydantic AI/Evals 2.17.0 dependencies or newer compatible versions.
Older Pydantic AI versions reject the `output_reasoning_tokens` field extracted
by genai-prices 0.1.x and can silently replace all token counts with zeros.
The dependency floor prevents that incompatibility; offline regression tests
cover reasoning usage, cache reads/writes, and the API-to-report path.

After syncing dependencies, restart running benchmark processes to load the fix.
Reports show unavailable (`null`) cost and token totals when an attempt lacks
input/output usage, rather than treating missing data as free. Other measured
metrics and benchmark scores remain available.

Existing checkpoints retain their original measurements on resume. Their usage
can be repaired from saved raw provider responses or another authoritative usage
record; zero-only traces cannot reconstruct the lost token counts. The Sol task
24 cost pilot retains its raw responses and has been repaired with original
report/checkpoint backups.
