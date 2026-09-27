Show the description and selected column metadata for one known table.

Use this only after the canonical table identifier is known. By default it
returns the first configured page of columns. Page through wide schemas
with ``start_column`` and ``end_column``, or request known exact identifiers
with ``column_names``; exact names take precedence over the range. Use
``find_columns`` instead for human-language concepts or unknown names.
{% if "few_shot" in capabilities %}

Examples:

- User: "Describe the item table."
  Context: catalog discovery returned `demo.inventory.items`.
  Tool: `describe_table`
  Arguments:

  ```json
  {"table_name": "demo.inventory.items"}
  ```

  Summarize the returned table description and column metadata.
- User: "What do item_id and quantity mean?"
  Context: both exact column names were returned for `demo.inventory.items`.
  Tool: `describe_table`
  Arguments:

  ```json
  {"table_name": "demo.inventory.items", "column_names": ["item_id", "quantity"]}
  ```

  Explain the returned definitions; if a description is missing, say so.
{% endif %}
