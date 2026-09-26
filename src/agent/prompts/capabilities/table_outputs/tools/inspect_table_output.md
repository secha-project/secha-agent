Inspect rows and columns from a saved table output.

Use this when the producer's preview does not contain enough rows or columns
to answer reliably. The row range is 1-based and inclusive. By default this
shows 10 rows and the first 10 columns; ``column_names`` selects specific
columns instead. Row positions are deterministic only when the producing
query defines an order. Inspection does not publish the table to the user.
{% if "few_shot" in capabilities %}

Examples:

- User: "What is the quantity on the twelfth row of that result?"
  Context: an ordered query returned `output_ref = {"output_id": "example-items"}`;
  its preview shows only the first ten of twenty rows and includes `quantity`.
  Call: `inspect_table_output({"ref": {"output_id": "example-items"}, "start_row": 12, "end_row": 12, "column_names": ["quantity"]})`.
  Answer from the returned row.
- User: "Explain the item IDs and quantities in the saved result."
  Context: a saved output has reference `{"output_id": "example-stock"}` and
  columns `item_id` and `quantity`, but the current preview omitted `quantity`.
  Call: `inspect_table_output({"ref": {"output_id": "example-stock"}, "column_names": ["item_id", "quantity"]})`.
  Use the returned preview to explain the values and state any preview limit.
{% endif %}
