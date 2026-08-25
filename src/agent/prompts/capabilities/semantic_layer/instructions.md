## Semantic layer capability

Use the semantic layer for governed business or domain metrics, declared measures, and combinations of declared measures and dimensions. Use only model, measure, and dimension names returned by its discovery tools.

{% if "query" in capabilities %}
Fall back to raw SQL only when the requested calculation is not represented in a semantic model or row-level analysis is required. Never silently replace an available governed measure with a hand-written approximation.
{% else %}
If the request is not represented by a semantic model or requires row-level analysis, explain that raw SQL is unavailable rather than approximating a governed measure.
{% endif %}
