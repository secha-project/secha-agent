# Agent run diagnostics

The local OpenTelemetry/Phoenix backend runs at `http://localhost:6006` and
persists traces in `phoenix.db`. For a quick read-only investigation, query the
SQLite database directly; SQLite also reads committed data from the adjacent WAL
file while the backend is running.

Run IDs reported by users may be span IDs rather than trace IDs. Resolve either
form first:

```bash
sqlite3 -header -column phoenix.db "
SELECT s.span_id, t.trace_id, s.trace_rowid, s.name, s.status_code,
       s.status_message
FROM spans AS s
JOIN traces AS t ON t.id = s.trace_rowid
WHERE s.span_id = '<run-id>' OR t.trace_id = '<run-id>';"
```

Then inspect the full span sequence using the returned `trace_rowid`:

```bash
sqlite3 -header -column phoenix.db "
SELECT span_id, parent_id, name, status_code, status_message, start_time, end_time
FROM spans
WHERE trace_rowid = <trace-rowid>
ORDER BY start_time;"
```

Tool arguments and model messages are stored as JSON in `spans.attributes`.
Prefer targeted `json_extract(...)` queries instead of printing all attributes,
which may contain user data or other sensitive values.
