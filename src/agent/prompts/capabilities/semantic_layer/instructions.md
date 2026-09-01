## Semantic layer capability

Use catalog metadata to discover native metric views for governed business or domain metrics. A metric view has `table_type` `METRIC_VIEW`; its fields are classified as dimensions or measures. Reference dimensions directly and measures only through `MEASURE(measure_name)` in Spark SQL.

`MEASURE(name)` is an aggregate. Selected dimensions define the result grain: put every dimension projected by `SELECT` in `GROUP BY`. Filter-only dimensions need not be grouped. For a time series, select and group by its time dimension. Any number of measures can share that grain.

Prefer an available governed measure over reconstructing it from a source table. Fall back to raw SQL calculations only when no metric view represents the request or row-level analysis is required.
