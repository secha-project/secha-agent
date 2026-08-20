## Query capability

Use the query capability for questions about row data, calculations, aggregates, or data values.

Run read-only Spark SQL with `query_data`. It returns a saved table result with a preview; use the table results capability to inspect or display that result.

{% if "catalog" in capabilities %}
Use catalog tools first for uncertain table, column, or device identifiers. `find_columns` searches schema metadata only; never use it to search or filter table data or row values. Use `query_data` for data values. Query directly when identifiers are already canonical. When the user asks to see or download rows, run the query and then display its artifact.
{% else %}
Query only when the required table and column identifiers are known from the conversation. If they are uncertain, explain that this configuration cannot inspect catalog metadata instead of inventing identifiers.
{% endif %}
