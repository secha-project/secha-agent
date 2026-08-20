## Catalog capability

Use the catalog capability to discover and inspect schema metadata.

Call `browse_tables` when a table name is unknown, ambiguous, or possibly misspelled. Call `find_columns` when the user describes a column in human language or its exact identifier is unknown. Use `describe_table` only after the table identifier is known, either to inspect a page of its columns or to retrieve exact columns by name.

When device identifiers are uncertain, use `browse_tables` to locate an available metadata table and `find_columns` or `describe_table` to inspect its identifier columns. Catalog tools search schema metadata only; they cannot search or filter table data or row values. Use `query_data` when data values are needed.

Treat table and column names returned by `browse_tables`, `find_columns`, and `describe_table` as authoritative. Do not invent identifiers.
