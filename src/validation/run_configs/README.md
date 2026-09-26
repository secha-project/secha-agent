# Benchmark ablations

The 16 experiment configs cover four models and every combination of few-shot
tool examples and semantic-layer access. Each config uses the complete
`power-quality-monitoring` dataset, **5 repetitions**, and concurrency **4**.
With 55 tasks, that
is 275 scored attempts per config and 4,400 across the matrix.

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
300-second retry budget, with no minimum interval. HTTP timeouts are separate
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
| `glm-5.3` | `zai` | `glm-5.3` |
| `qwen3.8-27b` | `tuni` | `Qwen3.8-27B-AIC` |

Model IDs were checked on 2026-09-26 against the official
[GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol),
[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), and
[GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3) documentation. Qwen retains the
existing TUNI deployment ID. These selections supersede the older model names
in `docs/3-methodology.tex`.

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
dataset, tasks, repetitions, and suffix set for comparisons.

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
The script stops if a command fails and can be invoked from any directory.

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
