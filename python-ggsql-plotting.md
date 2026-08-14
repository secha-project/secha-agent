# Engine-independent plotting with ggsql in Python

This document describes the Python plotting pipeline used by querychat. The
important architectural choice is that ggsql does not execute the user's SQL
against the application's database. querychat first executes the SQL through a
small query-engine interface, materializes the result, and then gives that
result plus the `VISUALISE` portion to ggsql. This preserves query pushdown and
keeps plotting independent of whether the source is DuckDB, SQLAlchemy, Ibis,
Polars SQL, or another implementation of `QueryExecutor`.

The pipeline is:

```text
user request
    -> LLM emits {ggsql, title}
    -> ggsql validates and splits SQL from VISUALISE
    -> QueryExecutor executes the SQL against the source engine
    -> result is converted to a Polars DataFrame
    -> DataFrame is registered with an in-memory ggsql DuckDBReader
    -> ggsql executes only the visualization against that result
    -> ggsql Spec -> VegaLiteWriter -> Altair chart
    -> Altair JupyterChart -> shinywidgets -> inline chat result
```

## Libraries installed and why

Visualization support is optional. Install it with:

```bash
pip install "querychat[viz]"
```

The `viz` extra contains:

| Library | Minimum version | Purpose |
|---|---:|---|
| `ggsql` | 0.3.2 | Parses and validates the combined SQL/`VISUALISE` query, produces a writer-independent `Spec`, and renders that spec through `VegaLiteWriter`. |
| `altair` | 6.0 | Represents the Vega-Lite chart and supplies `JupyterChart` plus static image export. |
| `shinywidgets` | 0.8.0 | Registers and displays the Altair `JupyterChart` as an interactive widget in Shiny. |
| `vl-convert-python` | 1.9.0 | Converts the Vega-Lite/Altair chart to PNG for model feedback and download/preview behavior. |

`duckdb` and `polars` also participate in the pipeline. `duckdb` is a core
querychat dependency and is used by ggsql's local reader. The engine result is
normalized to a Polars DataFrame before registration with that reader. Polars
is not pinned directly by the `viz` extra, so it must be supplied by ggsql's
dependency chain or by the selected querychat data-frame/development extra.

The same four optional visualization packages are included in the development
dependency group so that tests and type checks can exercise the complete
pipeline. On macOS CI, `unixODBC` is installed because ggsql may require it when
compiled from source.

querychat does not import ggsql during an ordinary package import. Imports of
`validate`, `DuckDBReader`, and `VegaLiteWriter` are deliberately inside the
visualization functions. When `"visualize"` is requested, querychat first uses
`importlib.util.find_spec()` to check that `ggsql`, `altair`, `shinywidgets`,
and `vl_convert` are all available. If not, construction fails early with the
`pip install querychat[viz]` instruction.

## Prompt supplied to the LLM

There are two prompt surfaces: visualization instructions in the system prompt
and the description of the `querychat_visualize` tool. The following blocks are
copied word for word from the Python prompt templates. Mustache expressions are
resolved at runtime. In particular, `{{> ggsql-syntax}}` is replaced verbatim
with `pkg-py/src/querychat/prompts/ggsql-syntax.md`, and `{{db_type}}` is
replaced with the source engine's SQL dialect name.

### System-prompt visualization section, verbatim

````markdown
{{#has_tool_visualize}}
### Visualizing Data

You can create visualizations using the `querychat_visualize` tool, which uses ggsql — a SQL extension for declarative data visualization. Write a ggsql query (SQL with a `VISUALISE` clause), and the tool executes the SQL, renders the `VISUALISE` clause as an Altair chart, and displays it inline in the chat.

#### Visualization best practices

Use the `querychat_get_schema` tool to retrieve column names, types, and summary statistics for a table before writing visualization queries. {{#has_tool_query}}If that context isn't sufficient for a confident visualization — e.g., you're unsure about value distributions, need to check for NULLs, or want to gauge row counts before choosing a chart type — use the `querychat_query` tool to inspect the data before visualizing. Always pass `collapsed=true` for these preparatory queries so the chart remains the focal point of the response.{{/has_tool_query}}

Follow the principles below to produce clear, interpretable charts.

#### Axis labels must be readable

When the x-axis contains categorical labels (names, categories, long strings), prefer flipping axes with `PROJECT y, x TO cartesian` so labels read naturally left-to-right. Short numeric or date labels on the x-axis are fine horizontal — this applies specifically to text categories.

#### Always include axis labels with units

Charts should be interpretable without reading the surrounding prose. Always include axis labels that describe what is shown, including units when applicable (e.g., `LABEL y => 'Revenue ($M)'`, not just `LABEL y => 'Revenue'`).

#### Maximize data-ink ratio

Every visual element should serve a purpose:

- Don't map columns to aesthetics (color, size, shape) unless the distinction is meaningful to the user's question. A single-series bar chart doesn't need color.
- When using color for categories, keep to 7 or fewer distinct values. Beyond that, consider filtering to the most important categories or using facets instead.
- Avoid dual-encoding the same variable (e.g., mapping the same column to both x-position and color) unless it genuinely aids interpretation.

#### Avoid overplotting

When a dataset has many rows, plotting one mark per row creates clutter that obscures patterns. Before generating a query, consider the row count and data characteristics visible in the schema.

**For large datasets (hundreds+ rows):**

- **Aggregate first**: Use `GROUP BY` with `COUNT`, `AVG`, `SUM`, or other aggregates to reduce to meaningful summaries before visualizing.
- **Choose chart types that summarize naturally**: histograms for distributions, boxplots for group comparisons, line charts for trends over time.

**For two numeric variables with many rows:**

Bin in SQL and use `DRAW tile` to create a heatmap:

```sql
WITH binned AS (
    SELECT ROUND(x_col / 5) * 5 AS x_bin,
           ROUND(y_col / 5) * 5 AS y_bin,
           COUNT(*) AS n
    FROM large_table
    GROUP BY x_bin, y_bin
)
SELECT * FROM binned
VISUALISE x_bin AS x, y_bin AS y, n AS fill
DRAW tile
SCALE fill TO viridis
```

**If individual points matter** (e.g., outlier detection): use `SETTING opacity` to reveal density through overlap.

#### Choose chart types based on the data relationship

Match the chart type to what the user is trying to understand:

- **Comparison across categories**: bar chart (`DRAW bar`, with `PROJECT y, x TO cartesian` for long labels). Order bars by value, not alphabetically.
- **Trend over time**: line chart (`DRAW line`). Use `SCALE x VIA date` for date columns.
- **Distribution of a single variable**: histogram (`DRAW histogram`) or density (`DRAW density`).
- **Relationship between two numeric variables**: scatter plot (`DRAW point`), but prefer aggregation or heatmap if the dataset is large.
- **Part-of-whole**: stacked bar chart (map subcategory to `fill`). Avoid pie charts — position along a common scale is easier to decode than angle.

#### ggsql syntax reference

<ggsql-syntax-reference>
{{> ggsql-syntax}}
</ggsql-syntax-reference>
{{#has_tool_query}}

**Avoid redundant expanded results.** If you run a preparatory query before visualizing, or if both a table and chart would show the same data, always pass `collapsed=true` on the query so the user sees the chart prominently, not a duplicate table above it. The user can still expand the table if they want the exact values.
{{/has_tool_query}}
{{/has_tool_visualize}}
````

The system prompt also tells the model to call `querychat_get_schema` before
writing SQL for a table whose schema has not yet been retrieved, and supplies
general read-only SQL rules. Those rules are shared by query, filtering, and
visualization tools rather than being plotting-specific.

### Tool prompt, verbatim

````markdown
Create a data visualization

Render a ggsql query (SQL with a VISUALISE clause) as an Altair chart displayed inline in the chat.

**When to use:** Call this tool when the user's question involves comparisons, distributions, or trends — even for small result sets, a chart is often clearer than a table.{{#has_tool_query}} For single-value answers (averages, counts, totals, specific lookups) or when the user needs exact values, use `querychat_query` instead.{{/has_tool_query}}

**Key constraints:**

- All data transformations must happen in the `SELECT` clause. `VISUALISE` and `MAPPING` accept column names only, not SQL expressions or functions.
- Do NOT include `LABEL title => ...` in the query — use the `title` parameter instead.
- If a visualization fails, read the error message carefully and retry with a corrected query. Common fixes: correcting column names, adding `SCALE DISCRETE` for integer categories, moving SQL expressions out of `VISUALISE` into the `SELECT` clause, and using `DRAW range` for interval-style marks instead of deprecated `errorbar`.{{#has_tool_query}} If the error persists, fall back to `querychat_query` for a tabular answer.{{/has_tool_query}}

{{#multi_table}}

**Multi-table queries:** The SELECT portion of your ggsql query can reference any table from the schema and use JOINs.

{{/multi_table}}
Parameters
----------
ggsql :
    A full ggsql query. Must include a VISUALISE clause and at least one DRAW clause. The SELECT portion uses {{db_type}} SQL; VISUALISE and MAPPING accept column names only, not expressions. Do NOT include `LABEL title => ...` in the query — use the `title` parameter instead.
title :
    A brief, user-friendly title for this visualization. This is displayed as the card header above the chart.

Returns
-------
:
    If successful, a static image of the rendered plot. If not, an error message.
````

At runtime, `Tool.from_func()` exposes this as a tool named
`querychat_visualize` with two string arguments:

```text
ggsql: the complete SQL + VISUALISE statement
title: a short title displayed in the chart card
```

## Validation of SQL and ggsql produced by the LLM

Validation and execution happen in a deliberate order.

### 1. Parse and validate the combined ggsql statement

The tool calls:

```python
validated = ggsql.validate(ggsql_text)
```

The resulting `Validated` object separates the input into:

```python
validated.sql()  # source-engine SQL
validated.visual()  # VISUALISE/DRAW/SCALE/etc.
```

The tool then requires `validated.has_visual()` to be true. A plain SQL query
is rejected with guidance to use the ordinary query tool. It also requires
`validated.valid()` to be true. If ggsql reports multiple errors, their
`message` fields are joined with newlines and returned to the model so it can
correct and retry the request.

This validation catches ggsql grammar errors, including invalid visual clauses
and expressions in places where ggsql accepts column names only. It also
performs the split between SQL and visualization syntax.

### 2. Apply the querychat single-result restriction

querychat currently supports visual layers that can all be replayed from one
final SQL result. Before running SQL, it scans the visual portion and rejects a
`DRAW` clause containing its own `MAPPING ... FROM source`. The model must
instead construct one final SQL relation—using CTEs and `UNION ALL` if
necessary—and select layers from it with `FILTER`.

Top-level `VISUALISE ... FROM table` remains supported. Because the Python
ggsql bindings do not expose that source directly, querychat extracts it from
the top-level `VISUALISE` clause. Quoted table names are preserved.

### 3. Validate and execute SQL in the source engine

The SQL part is passed only through this interface:

```python
class QueryExecutor(ABC):
    def execute_query(self, query: str) -> Any: ...
```

This is the query-engine independence boundary. The visualization layer knows
nothing about the source connection or SQL implementation.

Before execution, querychat's concrete executors call `check_query()`. It
rejects SQL beginning with destructive, schema, administrative, or—unless an
explicit environment escape hatch is enabled—data-modification keywords. The
configured source engine then parses, validates, and executes its own SQL
dialect. Consequently, `ggsql.validate()` is not a replacement for SQL-engine
validation: ggsql validates the combined grammar, while the source engine is
authoritative for SQL names, types, functions, joins, and syntax.

Any SQL safety, parse, bind, or execution error propagates into the visualize
tool's error handler. The user-visible/model-visible message is truncated to a
useful maximum length so that large Vega schemas or backend diagnostics do not
overwhelm the next model turn.

### 4. Materialize only the final result

The source engine performs filtering, joining, aggregation, CTE evaluation,
and other SQL work. Only its final result is converted to a Polars DataFrame:

```python
pl_df = to_polars(executor.execute_query(validated.sql()))
```

This preserves database pushdown and avoids teaching ggsql how to connect to
every source engine. The tradeoff is that the final plotting data must fit in
the Python process's memory.

## Rendering

Rendering has two stages: produce a writer-independent ggsql specification,
then turn it into an Altair/Vega-Lite chart.

### Produce the ggsql `Spec`

querychat creates a fresh local reader for each visualization:

```python
reader = ggsql.DuckDBReader("duckdb://memory")
```

For the common `SELECT ... VISUALISE ...` form, it registers the final frame as
`_data` and gives ggsql a synthetic local query:

```python
reader.register("_data", pl_df)
spec = reader.execute(f"SELECT * FROM _data {validated.visual()}")
```

For `VISUALISE ... FROM table`, it registers the frame under that table name
and executes the visual portion directly:

```python
reader.register(table_name, pl_df)
spec = reader.execute(validated.visual())
```

In both cases, the application's SQL has already run. The local DuckDB reader
exists only to let ggsql evaluate its visualization grammar against the
materialized final relation.

### Render the `Spec` as Altair/Vega-Lite

The writer converts the engine-independent spec to an Altair chart:

```python
chart = ggsql.VegaLiteWriter().render_chart(spec)
```

querychat deep-copies the chart and wraps it in `alt.JupyterChart`. Simple
charts use responsive `width="container"` and `height="container"`. Facet and
concatenation charts receive calculated pixel dimensions because container
sizing is not currently reliable for compound Vega-Lite specifications; a
Shiny reactive effect recalculates those dimensions when the output container
changes size.

`shinywidgets.register_widget()` registers the `JupyterChart`, and
`output_widget()` embeds it in the tool-result card. The card adds fullscreen,
PNG/SVG save controls, and a read-only view of the original ggsql query.

querychat also attempts to render a fixed-size PNG with Altair. This image is
returned to the model as visual feedback. PNG generation is best-effort: if it
fails, the interactive chart still succeeds and the tool returns a textual
confirmation.

On bookmark or chat-history restoration, querychat saves only each chart's
original ggsql string and widget ID. Restore revalidates the ggsql, reruns the
SQL against current source data, regenerates the spec and Altair chart, and
registers it under the original widget ID. A failed restored query is skipped
with a warning so one stale chart does not prevent the rest of the conversation
from loading.

## Minimal reusable implementation shape

The engine-independent core can be expressed as:

```python
from typing import Any, Protocol

import ggsql


class QueryEngine(Protocol):
    def execute_query(self, sql: str) -> Any: ...


def render_ggsql(engine: QueryEngine, query: str):
    validated = ggsql.validate(query)
    if not validated.has_visual():
        raise ValueError("Query must include a VISUALISE clause")
    if not validated.valid():
        raise ValueError("\n".join(error["message"] for error in validated.errors()))

    # `to_polars` is an application adapter for pandas, Polars, Arrow, Ibis,
    # or another frame returned by the selected engine.
    frame = to_polars(engine.execute_query(validated.sql()))

    reader = ggsql.DuckDBReader("duckdb://memory")
    reader.register("_data", frame)
    spec = reader.execute(f"SELECT * FROM _data {validated.visual()}")

    return ggsql.VegaLiteWriter().render_chart(spec)
```

Production code should additionally enforce read-only SQL, handle
`VISUALISE ... FROM table`, reject unsupported layer-level sources, constrain
error output, and decide how large a result may be materialized for plotting.

# Benchmark Annotator implementation notes

The implementation in Benchmark Annotator uses pandas as the complete
application-facing handoff between Spark and ggsql and Streamlit as the
renderer. Spark executes the application SQL once, and the application calls
`ggsql.render_altair(pandas_frame, visual)` with the resulting bounded frame.
No reader, table registration, or database setup is needed in the application.
The ggsql text cannot query Spark or Unity Catalog.

### Dependencies

The required packages are:

```toml
dependencies = [
    "altair>=6.0",
    "ggsql>=0.3.2",
    "pandas>=2.2.0",
    "streamlit>=1.58.0",
    "vl-convert-python>=1.9.0",
]
```

With UV, add or synchronize them with:

```bash
uv add "ggsql>=0.3.2" "altair>=6.0" "vl-convert-python>=1.9.0"
uv sync
```

`ggsql` parses and evaluates the visualization grammar. Its
`render_altair()` helper accepts pandas directly and produces an Altair chart.
Altair supplies the Vega-Lite representation, while `vl-convert-python` lets
`Chart.save()` export PNG without a browser or Selenium. pandas and Streamlit
are already core application dependencies. Do not add `duckdb` to the
application dependencies just to use this path.

### Materialize Spark results before plotting

Run the SQL through Spark, probe one row beyond the application limit, and
convert only the final result to pandas:

```python
MAX_ROWS = 1_000

spark_frame = spark.sql(sql).limit(MAX_ROWS + 1)
pandas_frame = spark_frame.toPandas()
truncated = len(pandas_frame.index) > MAX_ROWS
pandas_frame = pandas_frame.iloc[:MAX_ROWS].reset_index(drop=True)
pandas_frame.attrs = {}
```

This keeps joins, filters, windows, aggregation, and ordering in Spark. ggsql
never sees source credentials or tables; it sees only the capped pandas
result. Do not cache explicit SQL executions. Keep the materialized result
with the corresponding SQL text so a result is not plotted after its SQL has
changed.

### Parse ggsql but ignore its SQL portion

The visualization input may be either visual-only ggsql:

```sql
VISUALISE time_hour AS x, power AS y
DRAW line
```

or a combined SQL/visual statement. Validate the whole string, require a
visual section and at least one `DRAW`, then use only `validated.visual()`.
Any SQL returned by `validated.sql()` is deliberately ignored because the
Spark result is the only permitted data source.

Pass the pandas frame and visual portion directly to ggsql:

```python
validated = ggsql.validate(ggsql_text)
if not validated.has_visual():
    raise ValueError("ggsql must include a VISUALISE clause")
if not validated.valid():
    raise ValueError("\n".join(error["message"] for error in validated.errors()))

visual = validated.visual()
if not re.search(r"\bDRAW\b", visual, re.IGNORECASE):
    raise ValueError("ggsql must include at least one DRAW clause")

chart = ggsql.render_altair(pandas_frame, visual)
```

Do not write `FROM _data`; there is no application-managed table name. The
pandas frame passed to `render_altair()` is always the source. Combined
SQL/visual input is safe because the application ignores `validated.sql()` and
passes only `validated.visual()` onward.

### Normalize pandas timestamps to microseconds

Spark's `toPandas()` commonly produces `datetime64[ns]`. ggsql's numeric
datetime representation uses microseconds since the Unix epoch. Passing a
nanosecond pandas column unchanged makes the values 1,000 times too large; in
Vega-Lite this appears as dates around year `+57140` instead of 2025.

Normalize every datetime column before passing it to ggsql:

```python
def normalize_datetime_precision(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame
    for column in frame.columns:
        series = frame[column]
        dtype = series.dtype

        if isinstance(dtype, pd.DatetimeTZDtype):
            if normalized is frame:
                normalized = frame.copy()
            normalized[column] = (
                series.dt.tz_convert("UTC")
                .dt.tz_localize(None)
                .astype("datetime64[us]")
            )
        elif pd.api.types.is_datetime64_dtype(dtype):
            if normalized is frame:
                normalized = frame.copy()
            normalized[column] = series.astype("datetime64[us]")

    return normalized
```

This converts timezone-aware values to naive UTC before changing precision.
It preserves sub-second information down to microseconds and does not mutate
the SQL result unless a datetime column actually needs normalization.

For a timestamp aesthetic, let ggsql infer the scale or state it explicitly:

```sql
VISUALISE time_hour AS x, power AS y
DRAW line
SCALE x VIA datetime
```

Use `VIA date` only for date-only columns. Applying it to a timestamp produces
an error such as `Cannot coerce ... Timestamp(Nanosecond, None) to date`.
Axis labels can be formatted without changing the data type:

```sql
SCALE x VIA datetime
RENAMING * => '{:time %H:%M}'
```

### Render through Vega-Lite in Streamlit

Create the chart directly from pandas:

```python
chart = ggsql.render_altair(pandas_frame, visual)
```

In this application, passing that chart directly to `st.altair_chart()`
produced axes but no marks. The generated specification referred to a named
dataset, but the corresponding rows were not reliably delivered to the
frontend. The layer then had no rows on which to apply ggsql's generated
`__ggsql_source__` filter.

Split the named dataset from the Vega-Lite spec and bind its rows explicitly:

```python
streamlit_spec = chart.to_dict()
data_reference = streamlit_spec.pop("data")
datasets = streamlit_spec.pop("datasets")
rows = datasets[data_reference["name"]]

if streamlit_spec.get("height") == "container":
    streamlit_spec["height"] = 420

st.vega_lite_chart(
    pd.DataFrame(rows),
    streamlit_spec,
    width="stretch",
    height=500,
    theme=None,
)
```

Binding the data explicitly preserves ggsql's internal fields, including
`__ggsql_source__`, `__ggsql_row_index__`, and its mapped aesthetic columns.
Do not remove those fields: the generated layers and filters depend on them.

The explicit heights solve a separate sizing issue. ggsql may emit
`height: "container"`, while Streamlit's chart element defaults to
content-sized height. Those two responsive constraints can form a loop that
collapses the plot to a very short strip. Use a concrete Vega plot height and
a slightly larger Streamlit element height for axes and padding. Use
`width="stretch"`; `use_container_width` is deprecated.

Set `theme=None` when the saved/rendered Vega-Lite appearance should be kept
exactly. Allowing Streamlit's default chart theme can override parts of the
ggsql-generated configuration.

### Save reproducible artifacts

Persist three visualization artifacts beneath the task directory:

```text
artifacts/task_1/
├── visualization.ggsql
├── visualization.json
└── visualization.png
```

- `visualization.ggsql` is source metadata, analogous to `query.sql`.
- `visualization.json` is the expected visualization output. It must be a
  self-contained Vega-Lite document with its named dataset and transformed
  rows still embedded.
- `visualization.png` is a high-resolution preview and comparison artifact.

The eval_model is
```python
class VisualizationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec_path: str
    image_path: str
```

The corresponding YAML shape is:

```yaml
metadata:
  query_path: artifacts/task_1/query.sql
  ggsql_path: artifacts/task_1/visualization.ggsql
expected_output:
  - spec_path: artifacts/task_1/visualization.json
    image_path: artifacts/task_1/visualization.png
```

Build the standalone JSON before removing data for Streamlit transport:

```python
standalone_spec = chart.to_dict()
if standalone_spec.get("height") == "container":
    standalone_spec["height"] = 420

spec_json = (json.dumps(standalone_spec, indent=2, ensure_ascii=False) + "\n").encode(
    "utf-8"
)
```

Because `chart.to_dict()` contains both the top-level named-data reference and
the matching `datasets` mapping, the JSON can be rendered without rerunning
Spark, SQL, or ggsql.

Render PNG from a separate fixed-size chart. A fixed 1,200 by 700 chart at 2x
pixel density produces an image of at least 2,400 by 1,400 pixels, plus any
outer chart padding:

```python
export_chart = chart.properties(width=1_200, height=700)
buffer = BytesIO()
export_chart.save(buffer, format="png", scale_factor=2)
png = buffer.getvalue()
```

The interactive chart can remain responsive; export dimensions should not be
reused for the page preview.

Write the ggsql source, Vega-Lite JSON, PNG, and dataset YAML as one artifact
transaction. Resolve every relative path against the dataset directory,
reject paths that escape it, write each artifact to a temporary file, `fsync`
it, and replace the destination atomically. Before overwriting, compare the
opened dataset version with the file on disk. On failure, restore previous
artifact bytes as well as the previous dataset model so YAML and artifacts
cannot disagree.

Task copy, reorder, delete, and undo operations must treat all three files as
referenced artifacts. When positional task names change, rewrite paths to the
new `artifacts/task_N/` directory while preserving the case's stable ID.

### End-to-end rendering shape

The core sequence, omitting application forms and editor state, is:

```python
def render_materialized_ggsql(
    ggsql_text: str,
    spark_result: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, bytes, bytes]:
    validated = ggsql.validate(ggsql_text.strip())
    if not validated.has_visual() or not validated.valid():
        raise ValueError(validated.errors())

    visual = validated.visual()
    if not re.search(r"\bDRAW\b", visual, re.IGNORECASE):
        raise ValueError("At least one DRAW clause is required")

    frame = normalize_datetime_precision(spark_result)
    chart = ggsql.render_altair(frame, visual)

    standalone = chart.to_dict()
    if standalone.get("height") == "container":
        standalone["height"] = 420
    spec_json = (json.dumps(standalone, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )

    streamlit_spec = chart.to_dict()
    data_reference = streamlit_spec.pop("data")
    datasets = streamlit_spec.pop("datasets")
    streamlit_rows = pd.DataFrame(datasets[data_reference["name"]])
    if streamlit_spec.get("height") == "container":
        streamlit_spec["height"] = 420

    export_chart = chart.properties(width=1_200, height=700)
    png_buffer = BytesIO()
    export_chart.save(png_buffer, format="png", scale_factor=2)

    return streamlit_spec, streamlit_rows, spec_json, png_buffer.getvalue()
```
