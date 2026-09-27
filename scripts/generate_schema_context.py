from __future__ import annotations

import csv
import html
import json
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.agent.catalog import UnityCatalogClient
from src.agent.config import AgentConfig
from src.agent.spark_runtime import get_spark_session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "schema-context"
RAW_DIR = OUTPUT_DIR / "raw"
TABLES_DIR = OUTPUT_DIR / "tables"
INCLUDED_TABLES = (
    "device_metadata",
    "electricity_prices_procem",
    "power_quality_measurements_electrix",
    "power_quality_measurements_procem",
    "weather_measurements_procem",
)
SAMPLING_START = "2025-03-01 12:00:00"
SAMPLING_END = "2025-03-01 12:05:00"


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC)
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def json_text(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        value,
        default=json_default,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )


def markdown_cell(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, (dict, list)):
        value = json_text(value, indent=None)
    text = html.escape(str(value), quote=False).replace("|", "&#124;")
    return text.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")


def identifier(name: str) -> str:
    return f"`{name.replace('`', '``')}`"


def collect_rows(spark: Any, sql: str) -> list[dict[str, Any]]:
    return spark.sql(sql).toArrow().to_pylist()


def column_profile_sql(full_name: str, columns: list[str]) -> str:
    expressions = ["COUNT(*) AS row_count"]
    for column in columns:
        quoted = identifier(column)
        prefix = re.sub(r"[^a-zA-Z0-9]+", "_", column).strip("_").lower()
        expressions.extend(
            (
                f"COUNT({quoted}) AS {prefix}_non_null",
                f"COUNT(*) - COUNT({quoted}) AS {prefix}_null",
                f"COUNT(DISTINCT {quoted}) AS {prefix}_distinct",
            )
        )
    return f"SELECT {', '.join(expressions)} FROM {full_name}"


def cadence_sql(
    full_name: str,
    device_column: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> str:
    quoted_device = identifier(device_column)
    where = ""
    if start and end:
        where = (
            f" WHERE timestamp >= TIMESTAMP '{start}' AND timestamp < TIMESTAMP '{end}'"
        )
    return f"""WITH ordered AS (
  SELECT
    {quoted_device} AS device_identifier,
    timestamp,
    LAG(timestamp) OVER (
      PARTITION BY {quoted_device}
      ORDER BY timestamp
    ) AS previous_timestamp
  FROM {full_name}{where}
), gaps AS (
  SELECT
    device_identifier,
    unix_micros(timestamp) - unix_micros(previous_timestamp) AS gap_microseconds
  FROM ordered
  WHERE previous_timestamp IS NOT NULL
)
SELECT
  device_identifier,
  COUNT(*) AS observed_gaps,
  MIN(gap_microseconds) / 1000000.0 AS minimum_seconds,
  percentile_approx(gap_microseconds, 0.5) / 1000000.0 AS median_seconds,
  MAX(gap_microseconds) / 1000000.0 AS maximum_seconds
FROM gaps
GROUP BY device_identifier
ORDER BY device_identifier"""


def write_table_markdown(table: dict[str, Any]) -> None:
    name = table["name"]
    full_name = f"{table['catalog_name']}.{table['schema_name']}.{name}"
    columns = table.get("columns") or []
    lines = [
        f"# `{full_name}`",
        "",
        f"- Object type: `{table.get('table_type')}`",
        f"- Column count: **{len(columns)}**",
        "- Canonical source: [`../raw/catalog-metadata.json`](../raw/catalog-metadata.json)",
        "",
        "## Table metadata",
        "",
        "| Field | Catalog value |",
        "|---|---|",
    ]
    for key in sorted(key for key in table if key != "columns"):
        lines.append(f"| `{key}` | {markdown_cell(table[key])} |")
    lines.extend(
        (
            "",
            "## Ordered column dictionary",
            "",
            "Raw catalog positions are zero-based. Empty values are represented as `null`.",
            "",
            "| Position | Name | Type name | Type text | Type JSON | Precision | Scale | Interval type | Nullable | Partition index | Comment |",
            "|---:|---|---|---|---|---:|---:|---|---|---:|---|",
        )
    )
    fields = (
        "position",
        "name",
        "type_name",
        "type_text",
        "type_json",
        "type_precision",
        "type_scale",
        "type_interval_type",
        "nullable",
        "partition_index",
        "comment",
    )
    for column in columns:
        lines.append(
            "| "
            + " | ".join(markdown_cell(column.get(field)) for field in fields)
            + " |"
        )
    (TABLES_DIR / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_units(table: dict[str, Any]) -> dict[str, list[str]]:
    units: dict[str, list[str]] = {}
    for column in table.get("columns") or []:
        match = re.match(r"Unit:\s*([^,]*),", column.get("comment") or "")
        if match:
            units.setdefault(match.group(1).strip() or "unspecified", []).append(
                column["name"]
            )
    return units


def main() -> None:
    generated_at = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    config = AgentConfig()
    client = UnityCatalogClient(
        config.catalog_url, config.catalog_token, config.catalog_name
    )
    catalog_metadata = client._get(f"/catalogs/{config.catalog_name}")
    schema_metadata = client._get(
        f"/schemas/{config.catalog_name}.{config.schema_name}"
    )
    listed_names = {table["name"] for table in client.list_tables(config.schema_name)}
    missing = set(INCLUDED_TABLES) - listed_names
    if missing:
        raise RuntimeError(f"Catalog is missing expected tables: {sorted(missing)}")

    tables = []
    for name in INCLUDED_TABLES:
        table = client.get_table(f"{config.catalog_name}.{config.schema_name}.{name}")
        table["columns"] = sorted(
            table.get("columns") or [], key=lambda column: column.get("position", 0)
        )
        tables.append(table)
    tables.sort(key=lambda table: table["name"])
    tables_by_name = {table["name"]: table for table in tables}
    column_counts = {
        name: len(table["columns"]) for name, table in tables_by_name.items()
    }
    expected_counts = {
        "device_metadata": 4,
        "electricity_prices_procem": 4,
        "power_quality_measurements_electrix": 244,
        "power_quality_measurements_procem": 182,
        "weather_measurements_procem": 18,
    }
    if column_counts != expected_counts:
        raise RuntimeError(f"Unexpected column counts: {column_counts}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    catalog_snapshot = {
        "snapshot": {
            "generated_at_utc": generated_at,
            "catalog_name": config.catalog_name,
            "schema_name": config.schema_name,
            "included_table_count": len(tables),
            "included_tables": list(INCLUDED_TABLES),
            "scope_note": "Metric views are intentionally excluded by request.",
        },
        "catalog": catalog_metadata,
        "schema": schema_metadata,
        "tables": tables,
    }
    (RAW_DIR / "catalog-metadata.json").write_text(
        json_text(catalog_snapshot) + "\n", encoding="utf-8"
    )

    spark = get_spark_session(config, reuse_existing=False)
    profile: dict[str, Any] = {
        "snapshot": {
            "generated_at_utc": generated_at,
            "timezone": "UTC",
            "sampling_window_start_inclusive": f"{SAMPLING_START.replace(' ', 'T')}Z",
            "sampling_window_end_exclusive": f"{SAMPLING_END.replace(' ', 'T')}Z",
            "scope_note": "Metric views are intentionally excluded by request.",
        },
        "tables": {},
    }

    def run(table_name: str, query_name: str, sql: str) -> list[dict[str, Any]]:
        rows = collect_rows(spark, sql)
        profile["tables"].setdefault(table_name, {"queries": {}})["queries"][
            query_name
        ] = {
            "sql": sql,
            "rows": rows,
        }
        return rows

    device_full_name = "analytics.bronze.device_metadata"
    device_rows = run(
        "device_metadata",
        "verbatim_rows",
        f"SELECT description, device_identifier, notes, table_name FROM {device_full_name} "
        "ORDER BY table_name, device_identifier",
    )
    run(
        "device_metadata",
        "column_profile",
        column_profile_sql(
            device_full_name,
            ["description", "device_identifier", "notes", "table_name"],
        ),
    )
    run(
        "device_metadata",
        "referenced_tables",
        f"SELECT table_name, COUNT(*) AS device_count FROM {device_full_name} "
        "GROUP BY table_name ORDER BY table_name",
    )

    prices_full_name = "analytics.bronze.electricity_prices_procem"
    run(
        "electricity_prices_procem",
        "overview",
        f"SELECT COUNT(*) AS row_count, MIN(timestamp) AS minimum_timestamp, "
        f"MAX(timestamp) AS maximum_timestamp, COUNT(DISTINCT device_name) AS device_count, "
        f"MIN(price_FI) AS minimum_price_eur_per_mwh, "
        f"MAX(price_FI) AS maximum_price_eur_per_mwh, "
        f"SUM(CASE WHEN complete THEN 1 ELSE 0 END) AS complete_true, "
        f"SUM(CASE WHEN NOT complete THEN 1 ELSE 0 END) AS complete_false, "
        f"SUM(CASE WHEN complete IS NULL THEN 1 ELSE 0 END) AS complete_null "
        f"FROM {prices_full_name}",
    )
    run(
        "electricity_prices_procem",
        "per_device_coverage",
        f"SELECT device_name AS device_identifier, COUNT(*) AS row_count, "
        f"MIN(timestamp) AS minimum_timestamp, MAX(timestamp) AS maximum_timestamp "
        f"FROM {prices_full_name} GROUP BY device_name ORDER BY device_name",
    )
    run(
        "electricity_prices_procem",
        "column_profile",
        column_profile_sql(
            prices_full_name, ["device_name", "timestamp", "price_FI", "complete"]
        ),
    )
    run(
        "electricity_prices_procem",
        "observed_cadence",
        cadence_sql(prices_full_name, "device_name"),
    )

    weather_name = "weather_measurements_procem"
    weather_full_name = f"analytics.bronze.{weather_name}"
    weather_columns = [
        column["name"] for column in tables_by_name[weather_name]["columns"]
    ]
    run(
        weather_name,
        "overview",
        f"SELECT COUNT(*) AS row_count, MIN(timestamp) AS minimum_timestamp, "
        f"MAX(timestamp) AS maximum_timestamp, "
        f"COUNT(DISTINCT device_name) AS device_count FROM {weather_full_name}",
    )
    run(
        weather_name,
        "per_device_coverage",
        f"SELECT device_name AS device_identifier, COUNT(*) AS row_count, "
        f"MIN(timestamp) AS minimum_timestamp, MAX(timestamp) AS maximum_timestamp "
        f"FROM {weather_full_name} GROUP BY device_name ORDER BY device_name",
    )
    run(
        weather_name,
        "column_profile",
        column_profile_sql(weather_full_name, weather_columns),
    )
    run(
        weather_name,
        "observed_cadence",
        cadence_sql(
            weather_full_name,
            "device_name",
            start=SAMPLING_START,
            end=SAMPLING_END,
        ),
    )

    wide_tables = {
        "power_quality_measurements_electrix": "device_id",
        "power_quality_measurements_procem": "device_name",
    }
    for table_name, device_column in wide_tables.items():
        full_name = f"analytics.bronze.{table_name}"
        run(
            table_name,
            "overview",
            f"SELECT COUNT(*) AS row_count, MIN(timestamp) AS minimum_timestamp, "
            f"MAX(timestamp) AS maximum_timestamp, "
            f"COUNT(DISTINCT {identifier(device_column)}) AS device_count FROM {full_name}",
        )
        run(
            table_name,
            "per_device_coverage",
            f"SELECT {identifier(device_column)} AS device_identifier, COUNT(*) AS row_count, "
            f"MIN(timestamp) AS minimum_timestamp, MAX(timestamp) AS maximum_timestamp "
            f"FROM {full_name} GROUP BY {identifier(device_column)} ORDER BY device_identifier",
        )
        run(
            table_name,
            "identifier_and_timestamp_profile",
            column_profile_sql(full_name, [device_column, "timestamp"]),
        )
        run(
            table_name,
            "observed_cadence",
            cadence_sql(
                full_name,
                device_column,
                start=SAMPLING_START,
                end=SAMPLING_END,
            ),
        )

    (RAW_DIR / "data-profile.json").write_text(
        json_text(profile) + "\n", encoding="utf-8"
    )

    csv_path = RAW_DIR / "device_metadata.csv"
    fieldnames = ["description", "device_identifier", "notes", "table_name"]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(device_rows)

    for table in tables:
        write_table_markdown(table)

    profiles = profile["tables"]
    summary = [
        "# `analytics.bronze` schema summary",
        "",
        f"Snapshot generated: **{generated_at}**",
        "",
        f"This catalog-centric context covers {len(tables)} physical tables. Metric views are intentionally excluded.",
        "Raw catalog metadata is canonical; Markdown values are escaped only for readability.",
        "",
        "## Inventory",
        "",
        "| Table | Type | Columns | Rows at snapshot | Description |",
        "|---|---|---:|---:|---|",
    ]
    for table in tables:
        name = table["name"]
        query_name = "column_profile" if name == "device_metadata" else "overview"
        row_count = profiles[name]["queries"][query_name]["rows"][0]["row_count"]
        summary.append(
            f"| [`{name}`](tables/{name}.md) | `{table.get('table_type')}` | "
            f"{len(table['columns'])} | {row_count} | {markdown_cell(table.get('comment'))} |"
        )

    summary.extend(
        (
            "",
            "## Relationships and identifiers",
            "",
            "- `device_metadata.table_name` names one of the two power-quality tables.",
            "- `device_metadata.device_identifier` maps to `power_quality_measurements_electrix.device_id` or `power_quality_measurements_procem.device_name`, according to `table_name`.",
            "- The catalog does not declare primary-key or foreign-key constraints; these relationships are inferred from the metadata table content.",
            "- `electricity_prices_procem` is independent of the device metadata mapping and uses `device_name = ElectricityPrice`.",
            "- `weather_measurements_procem` is independent of the device metadata mapping and contains observations from one weather station.",
            "- All timestamp profiles and SQL execution use UTC.",
            "",
            "## Device metadata (verbatim values)",
            "",
            "The canonical CSV is [`raw/device_metadata.csv`](raw/device_metadata.csv). The same exact values are shown here with Markdown escaping.",
            "",
            "| Device identifier | Table | Description | Notes |",
            "|---|---|---|---|",
        )
    )
    for row in device_rows:
        summary.append(
            f"| {markdown_cell(row['device_identifier'])} | {markdown_cell(row['table_name'])} | "
            f"{markdown_cell(row['description'])} | {markdown_cell(row['notes'])} |"
        )

    summary.extend(("", "## Temporal coverage and cadence", ""))
    for name in (
        "electricity_prices_procem",
        "power_quality_measurements_electrix",
        "power_quality_measurements_procem",
        "weather_measurements_procem",
    ):
        overview = profiles[name]["queries"]["overview"]["rows"][0]
        cadence = profiles[name]["queries"]["observed_cadence"]["rows"]
        summary.extend((f"### `{name}`", ""))
        summary.append(
            f"- Coverage: `{json_default(overview['minimum_timestamp'])}` through "
            f"`{json_default(overview['maximum_timestamp'])}`; {overview['row_count']} rows across "
            f"{overview['device_count']} device identifier(s)."
        )
        if name == "electricity_prices_procem":
            summary.append(
                f"- Price range: {overview['minimum_price_eur_per_mwh']} to "
                f"{overview['maximum_price_eur_per_mwh']} EUR/MWh; completeness flags: "
                f"true={overview['complete_true']}, false={overview['complete_false']}, "
                f"null={overview['complete_null']}."
            )
            summary.append("- Observed cadence uses the full table range.")
        else:
            comment = tables_by_name[name].get("comment") or ""
            advertised = re.search(r"aggregated at ([^.]+? intervals?)", comment)
            if advertised:
                summary.append(f"- Catalog-advertised cadence: {advertised.group(1)}.")
            summary.append(
                f"- Observed cadence uses `{SAMPLING_START}Z` through `{SAMPLING_END}Z` (end exclusive)."
            )
        summary.extend(
            (
                "",
                "| Device | Observed gaps | Minimum seconds | Median seconds | Maximum seconds |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for row in cadence:
            summary.append(
                f"| {markdown_cell(row['device_identifier'])} | {row['observed_gaps']} | "
                f"{row['minimum_seconds']} | {row['median_seconds']} | {row['maximum_seconds']} |"
            )
        summary.append("")

    summary.extend(("## Units represented in column metadata", ""))
    for table in tables:
        units = extract_units(table)
        if not units:
            continue
        summary.extend(
            (
                f"### `{table['name']}`",
                "",
                "| Unit | Column count | Example columns |",
                "|---|---:|---|",
            )
        )
        for unit, columns in sorted(units.items()):
            examples = ", ".join(f"`{column}`" for column in columns[:8])
            if len(columns) > 8:
                examples += f", … (+{len(columns) - 8})"
            summary.append(f"| {markdown_cell(unit)} | {len(columns)} | {examples} |")
        summary.append("")

    notes = sorted({row["notes"] for row in device_rows if row.get("notes")})
    summary.extend(("## Data-quality and interpretation caveats", ""))
    summary.extend(f"- {markdown_cell(note)}" for note in notes)
    summary.extend(
        (
            "- The metadata table contains 3 Electrix identifiers while the live Electrix table contains additional identifiers; do not assume every observed device has a metadata row.",
            "- ProCem timestamps contain sub-second jitter. Use exact timestamps for filtering and ordering instead of assuming integer-second alignment.",
            "",
            "## Validation-question dimensions",
            "",
            "- Table discovery and source selection from catalog descriptions.",
            "- Natural-language-to-column resolution, including units, phases, harmonics, and measurement families.",
            "- Device-to-table mapping, location, equipment type, and deployment notes.",
            "- Reporting cadence, UTC boundaries, date coverage, completeness, and ordered time windows.",
            "- Cross-source naming differences such as `device_id` versus `device_name` and source-specific quantity names.",
            "- Weather conditions and solar irradiance alongside electrical measurements and electricity prices.",
            "- Nullability, missing metadata mappings, price completeness, and documented measurement defects.",
            "- Counter, instantaneous, average, minimum, maximum, phase, and total measurement semantics.",
            "",
            "For exact profile values and SQL, use [`raw/data-profile.json`](raw/data-profile.json).",
        )
    )
    (OUTPUT_DIR / "schema-summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    readme = [
        "# Schema context snapshot",
        "",
        f"Generated at **{generated_at}** from Unity Catalog schema `analytics.bronze`.",
        "This catalog-centric snapshot is intended as context for designing validation questions.",
        "Metric views are intentionally excluded.",
        "",
        "## Start here",
        "",
        "- [`schema-summary.md`](schema-summary.md): inventory, relationships, profiles, caveats, units, and question-design dimensions.",
        "- [`tables/`](tables/): exhaustive table and ordered column dictionaries.",
        "- [`raw/catalog-metadata.json`](raw/catalog-metadata.json): canonical catalog, schema, table, and column payloads.",
        f"- [`raw/device_metadata.csv`](raw/device_metadata.csv): all {len(device_rows)} metadata rows with exact stored values.",
        "- [`raw/data-profile.json`](raw/data-profile.json): aggregate results and exact read-only SQL.",
        "",
        "## Included physical tables",
        "",
    ]
    readme.extend(f"- `{table['name']}`" for table in tables)
    readme.extend(
        (
            "",
            "Every catalog field returned for the catalog, schema, included tables, and columns is retained. Owner identities, object IDs, timestamps, properties, and storage locations are included. The configured API URL, catalog token, Spark URL, and provider credentials are never serialized.",
            "",
            "## Extraction conventions",
            "",
            "- Tables are sorted by name; columns use the catalog's zero-based `position`.",
            "- `catalog-metadata.json` is canonical. Markdown only escapes values for safe rendering.",
            "- `device_metadata.csv` is UTF-8 with standard CSV quoting, LF line endings, and deterministic `table_name, device_identifier` ordering. Sorting changes presentation order only.",
            "- Spark uses UTC. JSON timestamps are ISO-8601 values ending in `Z`.",
            "- Time-series rows are not exported. Profiles contain only aggregates and per-device coverage.",
            "",
            "## Profiling definitions",
            "",
            "- All physical tables: row count; narrow tables: non-null, null, and distinct counts for every column.",
            "- Time-series tables: overall and per-device time coverage and observed cadence.",
            "- Wide power-quality tables: identifier and timestamp completeness only; every measurement column remains exhaustively documented through catalog metadata.",
            f"- Measurement and weather cadence use microsecond gaps from `{SAMPLING_START}Z` to `{SAMPLING_END}Z` (end exclusive). Price cadence uses its full range.",
            "- All profile SQL is preserved verbatim in `raw/data-profile.json` and is read-only.",
            "",
            "## Refresh",
            "",
            "Run `uv run python -m scripts.generate_schema_context` from the project root. The command replaces this point-in-time snapshot with current catalog metadata and aggregate profiles.",
            "",
            "## Snapshot limitation",
            "",
            "This is a point-in-time capture. Refresh it before treating row counts, coverage, ownership, comments, or storage details as current production truth.",
        )
    )
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        round_trip_rows = list(csv.DictReader(csv_file))
    if round_trip_rows != device_rows:
        raise RuntimeError("device_metadata CSV did not round-trip exactly")
    if len(round_trip_rows) != 16:
        raise RuntimeError(
            f"Expected 16 device metadata rows, got {len(round_trip_rows)}"
        )
    json.loads((RAW_DIR / "catalog-metadata.json").read_text(encoding="utf-8"))
    json.loads((RAW_DIR / "data-profile.json").read_text(encoding="utf-8"))
    print(
        json_text(
            {
                "generated_at_utc": generated_at,
                "column_counts": column_counts,
                "device_metadata_rows": len(device_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
