Find columns in one known table using human-language metadata keywords.

Use this when the exact column identifier is unknown or the user describes
a field in natural language. This searches column names and comments only;
it cannot find device IDs or other row values. Use ``query_data`` to search
or filter actual data. Returned identifiers are canonical.
{% if "few_shot" in capabilities %}

Examples:

- User: "Which field records the number of items in stock?"
  Context: catalog discovery returned `demo.inventory.items`.
  Tool: `find_columns`
  Arguments:

  ```json
  {"table_name": "demo.inventory.items", "keywords": "quantity stock"}
  ```

  If the result identifies `quantity`, report that exact name and its metadata.
- User: "Find the shipment date column."
  Context: `demo.inventory.shipments` is a known canonical table.
  Tool: `find_columns`
  Arguments:

  ```json
  {"table_name": "demo.inventory.shipments", "keywords": "shipment date"}
  ```

  Use the returned column identifier in subsequent queries; do not search for
  a particular date value with this tool.
{% endif %}
