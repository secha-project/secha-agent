Create a private chart preview from a saved table output and visual-only ggsql.

Returns a chart `output_ref`, a truncation flag, and a rendered PNG for inspection.
This does not publish the chart. Review the image, correct concrete problems
with another call if necessary, then call `display_chart_output` with the chosen
chart reference. Chart references are available only in the current run.

Prepare filtering, joins, transformations, aggregation, ordering, types, and
aliases in the producing query. Publish the table only if also requested.

Use ordered bars for categories, lines for time, histogram or density for a
distribution, points for two numeric variables, and stacked bars for
part-to-whole. Avoid pies. Aggregate or bin large data; use opacity when
overlapping points matter. Use aesthetics only when meaningful and keep
categorical color to at most seven groups.

ggsql syntax reference:

- **Mappings:** Start with ``VISUALISE column AS aesthetic, ...``.
  Use result columns; prepare expressions and type conversions upstream.
- **Layers:** ``DRAW mark`` adds a layer; include at least one. Common marks:
  ``point``, ``line``, ``bar``, ``area``, ``tile``, ``histogram``,
  ``density``, ``boxplot``. Repeat ``DRAW`` to combine layers;
  per-layer ``MAPPING`` overrides inherited mappings.
- **Settings:** ``SETTING name => value`` supplies parameters or fixed
  appearance values. Mapped columns encode data; settings supply constants.
- **Grouping:** Discrete aesthetic mappings separate series within a layer.
  For separate value columns, use layers with
  ``MAPPING value AS y, 'Series' AS stroke``. Never invent ``y2`` or ``y3``.
- **Histograms:** Map numeric observations to ``x``; counts supply ``y``
  automatically. Set ``binwidth`` or ``bins`` and ``closed`` (``'left'`` or
  ``'right'``), e.g. ``SETTING binwidth => 5, closed => 'right'``. For
  compressed frequencies, map ``frequency AS weight``. Keep ``x`` numeric.
- **Scales:** ``SCALE [type] aesthetic`` controls value mapping. Common types:
  ``CONTINUOUS``, ``DISCRETE``. ``FROM (...)`` sets numeric bounds or category
  order; ``TO`` sets a range or palette; ``VIA`` selects a transformation.
  Use ``date`` for dates; leave timestamps implicit or use ``datetime``.
- **Ticks:** ``SCALE x SETTING breaks => (0, 5, 10)`` sets positions;
  ``breaks => 5`` requests roughly five ticks, not five-unit spacing.
  Scale ``RENAMING * => '{:num %.1f}'`` or ``'{:time %H:%M}'`` formats
  tick labels without changing data.
- **Labels and orientation:** ``LABEL aesthetic => 'Label'`` names axes or
  legends; include units. ``PROJECT y, x TO cartesian`` swaps displayed axes.

Common errors: ``ref`` supplies the data; omit source ``FROM`` clauses and
``_data`` references. ``SCALE ... FROM (...)`` is allowed. ggsql 0.3.3 can
round fractional ticks despite label formatting: inspect the preview and use
explicit whole-number ticks where suitable, without rounding or shifting data.

On ModelRetry, correct the syntax or columns with the same result unless its
data is wrong. Successful rendering is not visual approval: inspect the image.
{% if "few_shot" in capabilities %}

Examples:

- User: "Plot total stock by category."
  Context: an upstream query prepared `category` and numeric `quantity`, sorted
  by quantity, and returned `output_ref = {"output_id": "example-stock"}`.
  Tool: `visualize_table_output`
  Arguments:

  ```json
  {"ref": {"output_id": "example-stock"}, "ggsql": "VISUALISE category AS x, quantity AS y DRAW bar LABEL x => 'Category', y => 'Quantity (items)'", "title": "Stock by category"}
  ```

  Inspect the returned PNG. If it meets the request, pass its `output_ref` to
  `display_chart_output`, then summarize without publishing an unrequested table.
- User: "Plot daily shipped quantities over time."
  Context: an upstream query aggregated and ordered rows with date column `day`
  and numeric `quantity`, returning `output_ref = {"output_id": "example-daily"}`.
  Tool: `visualize_table_output`
  Arguments:

  ```json
  {"ref": {"output_id": "example-daily"}, "ggsql": "VISUALISE day AS x, quantity AS y DRAW line LABEL x => 'Day', y => 'Quantity (items)'", "title": "Daily shipped quantities"}
  ```

  Use that saved result directly; all aggregation has already been done. Review
  the PNG, fix any concrete problems, then publish the chosen chart reference.
{% endif %}
