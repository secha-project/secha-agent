Create a chart from a saved table output and visual-only ggsql.

Prepare filtering, joins, transformations, aggregation, ordering, types, and
aliases in the producing query. Publish the table only if also requested.

Use ordered bars for categories, lines for time, histogram or density for a
distribution, points for two numeric variables, and stacked bars for
part-to-whole. Avoid pies. Aggregate or bin large data; use opacity when
overlapping points matter. Use aesthetics only when meaningful and keep
categorical color to at most seven groups.

ggsql syntax and common-error rules:

- Start with ``VISUALISE`` mappings such as ``column AS x`` and add at least
  one ``DRAW mark``. Common marks: ``bar``, ``line``, ``point``,
  ``histogram``, ``density``, and ``tile``.
- Use ``LABEL aesthetic => 'Label'`` with units. For long categories, use
  ``PROJECT y, x TO cartesian``.
- For multiple values, use one ``DRAW`` per value with
  ``MAPPING value AS y, 'Series' AS stroke``. Never invent ``y2`` or ``y3``.
- ``SCALE x VIA date`` accepts dates, never timestamps. Leave timestamp
  scales implicit or use ``datetime`` when needed.
- Map result columns only: no expressions, functions, or ``RENAMING`` for
  coercion or date formatting. Prepare typed display fields upstream.
- Do not add ``FROM`` or ``_data``; ``ref`` supplies the data.

On ModelRetry, correct the syntax or columns with the same result unless its
data is wrong. Do not generate cosmetic alternatives after success.
