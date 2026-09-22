Inspect rows and columns from a saved table output.

Use this when the producer's preview does not contain enough rows or columns
to answer reliably. The row range is 1-based and inclusive. By default this
shows 10 rows and the first 10 columns; ``column_names`` selects specific
columns instead. Row positions are deterministic only when the producing
query defines an order. Inspection does not publish the table to the user.
