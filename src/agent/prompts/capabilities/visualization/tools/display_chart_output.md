Publish a reviewed chart preview in the user interface.

Use the chart `output_ref` returned by `visualize_table_output`, not the source
table reference. Inspect the rendered image before calling this tool. If the
chart needs correction, create and inspect a revised preview first. Publish
only the selected version of each requested chart. Publication uses the saved
specification without rerunning the query or rebuilding the chart.
{% if "few_shot" in capabilities %}

Examples:

- User: "Plot total stock by category."
  Context: `visualize_table_output` returned a readable bar chart with correct
  values and units and `output_ref = {"output_id": "example-chart"}`.
  Tool: `display_chart_output`
  Arguments:

  ```json
  {"ref": {"output_id": "example-chart"}}
  ```

  After publication succeeds, give any requested interpretation.
{% endif %}
