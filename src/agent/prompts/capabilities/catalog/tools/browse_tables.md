List tables in the configured schema with canonical names and short descriptions.

Use this first when a table name is unknown, ambiguous, or possibly
misspelled, including when locating a metadata table that may map a human
device name to its stored identifier. This tool does not return columns or
search row values; after selecting a table, use ``find_columns`` or
``describe_table`` to inspect its schema.
