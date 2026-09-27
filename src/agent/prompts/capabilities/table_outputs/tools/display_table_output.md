Publish a saved table output in the user interface for viewing or download.

Use this when the user asks to see, get, or download rows, records, tables,
sample data, or a sample time series. Do not use it for a scalar answer or
explanation alone, and do not manually reproduce the output as a Markdown
or text table. Publishing does not replace requested explanation,
interpretation, or caveats.
{% if "few_shot" in capabilities %}

Examples:

- User: "Show me the five item records."
  Context: the matching query returned `output_ref = {"output_id": "example-items"}`.
  Tool: `display_table_output`
  Arguments:

  ```json
  {"ref": {"output_id": "example-items"}}
  ```

  After publication succeeds, briefly confirm that the table is available.
- User: "Make that shipment result available for download and explain it."
  Context: the saved shipment result has reference `{"output_id": "example-shipments"}`.
  Tool: `display_table_output`
  Arguments:

  ```json
  {"ref": {"output_id": "example-shipments"}}
  ```

  Then give the requested explanation using the observed result.
{% endif %}
