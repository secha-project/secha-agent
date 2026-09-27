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
{% if "few_shot" in capabilities %}

Examples:

- User: "How many item records are stored?"
  Context: `demo.inventory.items` is a discovered source table.
  Tool: `query_data`
  Arguments:

  ```json
  {"sql": "SELECT COUNT(*) AS item_count FROM demo.inventory.items"}
  ```

  If the preview contains `item_count = 12`, report that count in the user's
  requested final-answer format.
  A scalar answer does not need a displayed table.
- User: "Show five item records ordered by item ID."
  Context: `demo.inventory.items` and its `item_id` and `quantity` columns are known.
  Tool: `query_data`
  Arguments:

  ```json
  {"sql": "SELECT item_id, quantity FROM demo.inventory.items ORDER BY item_id LIMIT 5"}
  ```

  Pass the returned `output_ref` to `display_table_output` to publish the rows.
{% endif %}
