from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import SparkSession

from src.agent.catalog import UnityCatalogClient

METRIC_VIEWS_DIR = Path(__file__).resolve().parents[2] / "metric_views"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MetricField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    expr: str = Field(min_length=1, max_length=1024)


class MetricViewDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["0.1"] = "0.1"
    source: str = Field(min_length=1)
    filter: str | None = None
    dimensions: list[MetricField] = Field(default_factory=list)
    measures: list[MetricField] = Field(default_factory=list)

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, value: str) -> MetricViewDefinition:
        return cls.model_validate(yaml.safe_load(value))


class MetricView(BaseModel):
    full_name: str
    name: str
    comment: str = ""
    properties: dict[str, str] = Field(default_factory=dict)
    definition: MetricViewDefinition
    field_comments: dict[str, str] = Field(default_factory=dict)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quoted_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"Invalid identifier {value!r}; use letters, numbers, and underscores."
        )
    return f"`{value}`"


def build_create_metric_view_sql(
    full_name: str,
    definition: MetricViewDefinition,
    *,
    comment: str = "",
    properties: dict[str, str] | None = None,
    field_comments: dict[str, str] | None = None,
) -> str:
    identifier = ".".join(_quoted_identifier(part) for part in full_name.split("."))
    fields = definition.dimensions + definition.measures
    comments = field_comments or {}
    column_ddl = ""
    if any(comments.get(field.name) for field in fields):
        columns = [
            _quoted_identifier(field.name)
            + (
                f" COMMENT {_sql_string(comments[field.name])}"
                if comments.get(field.name)
                else ""
            )
            for field in fields
        ]
        column_ddl = " (\n  " + ",\n  ".join(columns) + "\n)"

    clauses = [f"CREATE OR REPLACE VIEW {identifier}{column_ddl}"]
    if comment:
        clauses.append(f"COMMENT {_sql_string(comment)}")
    if properties:
        values = ", ".join(
            f"{_sql_string(key)} = {_sql_string(value)}"
            for key, value in sorted(properties.items())
        )
        clauses.append(f"TBLPROPERTIES ({values})")
    clauses.extend(("WITH METRICS", "LANGUAGE YAML"))

    yaml_definition = definition.to_yaml()
    tag = "metric_view"
    while f"${tag}$" in yaml_definition:
        tag += "_definition"
    clauses.append(f"AS ${tag}$\n{yaml_definition}${tag}$")
    return "\n".join(clauses)


def build_documentation_yaml(
    full_name: str,
    definition: MetricViewDefinition,
    *,
    comment: str = "",
    properties: dict[str, str] | None = None,
    field_comments: dict[str, str] | None = None,
) -> str:
    metadata = yaml.safe_dump(
        {
            "catalog_metadata": {
                "full_name": full_name,
                "description": comment or None,
                "properties": properties or {},
                "field_descriptions": field_comments or {},
            }
        },
        sort_keys=False,
        allow_unicode=True,
    )
    commented_metadata = "\n".join(f"# {line}" for line in metadata.splitlines())
    return f"{commented_metadata}\n{definition.to_yaml()}"


class MetricViewService:
    def __init__(
        self,
        spark: SparkSession,
        catalog: UnityCatalogClient,
        schema_name: str,
        documentation_dir: Path = METRIC_VIEWS_DIR,
    ):
        self.spark = spark
        self.catalog = catalog
        self.schema_name = schema_name
        self.documentation_dir = documentation_dir

    def list_metric_views(self) -> list[dict]:
        return [
            table
            for table in self.catalog.list_tables(self.schema_name)
            if str(table.get("table_type", "")).upper() == "METRIC_VIEW"
        ]

    def load(self, full_name: str) -> MetricView:
        table = self.catalog.get_table(full_name)
        view_definition = table.get("view_definition")
        if not isinstance(view_definition, str):
            raise TypeError(f"{full_name} has no YAML view_definition.")
        field_comments = {
            str(column["name"]): str(column["comment"])
            for column in table.get("columns") or []
            if column.get("name") and column.get("comment")
        }
        return MetricView(
            full_name=full_name,
            name=str(table.get("name") or full_name.rsplit(".", 1)[-1]),
            comment=str(table.get("comment") or ""),
            properties={
                str(key): str(value)
                for key, value in (table.get("properties") or {}).items()
            },
            definition=MetricViewDefinition.from_yaml(view_definition),
            field_comments=field_comments,
        )

    def save(
        self,
        name: str,
        definition: MetricViewDefinition,
        *,
        comment: str = "",
        properties: dict[str, str] | None = None,
        field_comments: dict[str, str] | None = None,
    ) -> str:
        full_name = f"{self.catalog.name}.{self.schema_name}.{name}"
        sql = build_create_metric_view_sql(
            full_name,
            definition,
            comment=comment,
            properties=properties,
            field_comments=field_comments,
        )
        self.spark.sql(sql).collect()

        self.documentation_dir.mkdir(parents=True, exist_ok=True)
        path = self.documentation_dir / f"{full_name}.yaml"
        path.write_text(
            build_documentation_yaml(
                full_name,
                definition,
                comment=comment,
                properties=properties,
                field_comments=field_comments,
            ),
            encoding="utf-8",
        )
        return full_name
