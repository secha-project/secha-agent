Run read-only Spark SQL and save a table output with a small preview.

Use this for row-level data, actual values, aggregates, and calculations.
Resolve uncertain identifiers with catalog tools first; do not use
schema-search tools for row values. Follow the active capability policy when
choosing between governed objects and source-table calculations.

Answer from the preview when it is sufficient. Otherwise pass the returned
reference to ``inspect_table_output``. This tool does not publish data in
the UI: when the user asks to see or download rows, records, a table, sample
data, or a sample time series, pass the reference to
``display_table_output`` instead of reproducing the result in Markdown.
