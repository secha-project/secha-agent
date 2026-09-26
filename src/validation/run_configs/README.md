# Benchmark ablations

The 16 experiment configs cover four models and every combination of few-shot
tool examples and semantic-layer access. Each config uses the complete
`power-quality-monitoring` dataset, **5 repetitions**, concurrency 4, and up to
3 attempts for transient task failures. With 55 tasks, that is 275 scored
attempts per config and 4,400 across the matrix, before transient retries.

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
for root in gpt-6-sol gpt-6-luna glm-5.3 qwen3.8-27b; do
  for variant in baseline few_shot semantic_layer few_shot_semantic_layer; do
    uv run --locked --no-dev --no-cache python -m src.validation.run_benchmark \
      --config "${root}__${variant}" --dry-run || exit 1
  done
done
```

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
