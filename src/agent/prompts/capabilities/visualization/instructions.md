Use visualization for comparisons, distributions, relationships, and trends. Resolve uncertain table names, columns, types, and units with catalog tools first.

Run `query_data` with the complete read-only SQL for the chart, putting all transformations, filtering, joins, aggregation, ordering, and aliases in that query. Then call `visualize_query_result` with its returned table reference, visual-only ggsql, and a short title. Do not display the table with `display_query_result` unless the user also asked to see or download it.

The ggsql must begin with `VISUALISE`, reference only columns in the query result, and include at least one `DRAW`. Do not include SQL, `FROM`, `_data`, expressions in `VISUALISE` or `MAPPING`, or `LABEL title`; use the tool's `title` argument.

Choose charts deliberately:

- Categories: `DRAW bar`, ordered by value in SQL. Use `PROJECT y, x TO cartesian` for long text labels.
- Time trends: `DRAW line`. Use `SCALE x VIA datetime` for timestamps and `SCALE x VIA date` only for date-only values; format time labels with `RENAMING * => '{:time %H:%M}'` when useful.
- One-variable distributions: `DRAW histogram` or `DRAW density`.
- Two numeric variables: `DRAW point` for modest results. For hundreds of rows, aggregate or bin in SQL and use `DRAW tile` with `SCALE fill TO viridis`; use opacity only when individual overlapping points matter.
- Part-to-whole: stacked bars with the subcategory mapped to `fill`, not pie charts.
- Intervals: `DRAW range`, not deprecated `errorbar` marks.

Always label axes clearly, including units. Map color, fill, size, or shape only when it answers the question; keep categorical colors to seven or fewer values and avoid encoding the same variable twice. For integer categories, use `SCALE x TO discrete` (or the relevant aesthetic) so ggsql does not treat them as continuous.

Example ggsql for a saved result containing `category` and `total_value`:

```sql
VISUALISE category AS x, total_value AS y
DRAW bar
PROJECT y, x TO cartesian
LABEL x => 'Category', y => 'Total value (unit)'
```

If rendering fails, correct the columns or grammar using the retry message. After success, continue the answer without generating cosmetic alternatives.
