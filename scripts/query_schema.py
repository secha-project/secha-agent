"""Small, standalone helpers for Unity Catalog and Spark SQL experiments.

Examples:
    uv run python -m scripts.query_schema tables --full
    uv run python -m scripts.query_schema table weather_measurements_procem
    uv run python -m scripts.query_schema sql \
        "SELECT COUNT(*) AS rows FROM analytics.bronze.weather_measurements_procem"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from pyspark.sql import SparkSession


@dataclass(frozen=True)
class Settings:
    catalog_url: str
    catalog_token: str
    spark_url: str
    catalog_name: str = "analytics"
    schema_name: str = "bronze"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")

        def required(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise RuntimeError(f"Missing required environment variable: {name}")
            return value

        return cls(
            catalog_url=required("CATALOG_URL"),
            catalog_token=required("CATALOG_TOKEN"),
            spark_url=required("SPARK_URL"),
            catalog_name=os.getenv("CATALOG_NAME", "analytics"),
            schema_name=os.getenv("SCHEMA_NAME", "bronze"),
        )


def catalog_get(
    settings: Settings,
    path: str,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Make one authenticated GET request to Unity Catalog."""
    url = f"{settings.catalog_url.rstrip('/')}/api/2.1/unity-catalog{path}"
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {settings.catalog_token}",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def list_tables(
    settings: Settings,
    *,
    include_metric_views: bool = False,
) -> list[dict[str, Any]]:
    """List tables in the configured schema, following catalog pagination."""
    params = {
        "catalog_name": settings.catalog_name,
        "schema_name": settings.schema_name,
        "omit_columns": "true",
        "omit_properties": "true",
    }
    tables: list[dict[str, Any]] = []
    while True:
        page = catalog_get(settings, "/tables", params)
        tables.extend(page.get("tables") or [])
        page_token = page.get("next_page_token")
        if not page_token:
            break
        params["page_token"] = page_token

    if not include_metric_views:
        tables = [
            table
            for table in tables
            if str(table.get("table_type", "")).upper() != "METRIC_VIEW"
        ]
    return sorted(tables, key=lambda table: table["name"])


def get_table(settings: Settings, name: str) -> dict[str, Any]:
    """Return full metadata, including columns, for one table."""
    full_name = (
        name
        if name.count(".") == 2
        else (f"{settings.catalog_name}.{settings.schema_name}.{name}")
    )
    return catalog_get(settings, f"/tables/{quote(full_name, safe='.')}")


def get_spark(settings: Settings) -> SparkSession:
    """Create a UTC Spark Connect session configured for Unity Catalog."""
    spark = (
        SparkSession.builder.config(
            f"spark.sql.catalog.{settings.catalog_name}",
            "io.unitycatalog.spark.UCSingleCatalog",
        )
        .config(
            f"spark.sql.catalog.{settings.catalog_name}.uri",
            settings.catalog_url,
        )
        .config(
            f"spark.sql.catalog.{settings.catalog_name}.token",
            settings.catalog_token,
        )
        .config("spark.sql.defaultCatalog", settings.catalog_name)
        .config("spark.sql.session.timeZone", "UTC")
        .remote(settings.spark_url)
        .create()
    )
    spark.catalog.setCurrentDatabase(settings.schema_name)
    return spark


def query(settings: Settings, sql: str) -> list[dict[str, Any]]:
    """Execute SQL and return dictionaries with timestamp timezones preserved."""
    return get_spark(settings).sql(sql).toArrow().to_pylist()


def json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def print_json(value: Any) -> None:
    print(json.dumps(value, default=json_default, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tables_parser = subparsers.add_parser("tables", help="List schema tables")
    tables_parser.add_argument(
        "--full", action="store_true", help="Fetch full metadata for every table"
    )
    tables_parser.add_argument(
        "--include-metric-views",
        action="store_true",
        help="Include metric views (excluded by default)",
    )

    table_parser = subparsers.add_parser("table", help="Fetch full table metadata")
    table_parser.add_argument("name", help="Table name or catalog.schema.table")

    sql_parser = subparsers.add_parser("sql", help="Execute Spark SQL")
    sql_parser.add_argument("query", nargs="?", help="SQL text; reads stdin if omitted")

    args = parser.parse_args()
    settings = Settings.from_env()

    if args.command == "tables":
        tables = list_tables(settings, include_metric_views=args.include_metric_views)
        if args.full:
            tables = [get_table(settings, table["name"]) for table in tables]
        print_json(tables)
    elif args.command == "table":
        print_json(get_table(settings, args.name))
    else:
        sql = args.query if args.query is not None else sys.stdin.read()
        if not sql.strip():
            parser.error("SQL must be passed as an argument or through stdin")
        print_json(query(settings, sql))


if __name__ == "__main__":
    main()
