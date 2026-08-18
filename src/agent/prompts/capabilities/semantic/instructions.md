The semantic capability provides governed business or domain metrics, calculated measures, and questions that combine declared measures and dimensions. Always search for the requested metrics in the semantic models and prefer it over reconstructing a declared measure in raw SQL.

Call `list_semantic_models` to know which models you have access to. Call `describe_semantic_model` before querying a model unless its exact dimensions and measures are already known from a tool result in this conversation. Use only names returned by these tools.

Run governed queries with `query_semantic_model`. Use `compare_semantic_periods` for period-over-period questions. The returned reference is an ordinary table result: use `inspect_query_result` when its preview is insufficient, and `display_query_result` when the user asks to see or download the data. It can also be passed to the visualization capability.

Use raw `query_data` only when the requested calculation is not represented by the semantic models or when row-level analysis is required. Do not silently substitute a hand-written SQL approximation for an available governed measure.
