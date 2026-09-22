Show the description and selected column metadata for one known table.

Use this only after the canonical table identifier is known. By default it
returns the first configured page of columns. Page through wide schemas
with ``start_column`` and ``end_column``, or request known exact identifiers
with ``column_names``; exact names take precedence over the range. Use
``find_columns`` instead for human-language concepts or unknown names.
