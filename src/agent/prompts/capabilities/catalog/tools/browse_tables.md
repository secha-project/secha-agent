List tables in the configured schema with canonical names and short descriptions.

Use this first when a table name is unknown, ambiguous, or possibly
misspelled, including when locating a metadata table that may map a human
device name to its stored identifier. This tool does not return columns or
search row values; after selecting a table, use ``find_columns`` or
``describe_table`` to inspect its schema.
{% if "few_shot" in capabilities %}

Examples:

- User: "Which inventory tables are available?"
  Tool: `browse_tables`
  Arguments:

  ```json
  {}
  ```

  If the result lists `demo.inventory.items` and `demo.inventory.shipments`,
  summarize those returned names and descriptions.
- User: "Find the table containing item categories."
  Tool: `browse_tables`
  Arguments:

  ```json
  {}
  ```

  If `demo.inventory.items` is a relevant returned table, inspect its columns
  next with `find_columns`; do not infer columns from its table name alone.
{% endif %}
