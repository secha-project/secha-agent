from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import ModelRetry

from src.agent.capabilities.catalog import browse_tables, describe_table, find_columns
from src.agent.catalog import parse_columns


class Catalog:
    name = "catalog"

    def list_tables(self, schema_name: str) -> list[dict[str, Any]]:
        assert schema_name == "schema"
        return [
            {
                "name": "measurements",
                "table_type": "TABLE",
                "comment": "Raw measurements",
            },
            {
                "name": "energy_metrics",
                "table_type": "METRIC_VIEW",
                "comment": "Governed energy metrics",
            },
        ]

    def get_table(self, full_table_name: str) -> dict[str, Any]:
        assert full_table_name in {
            "catalog.schema.measurements",
            "catalog.schema.energy_metrics",
        }
        return {
            "table_type": (
                "METRIC_VIEW" if full_table_name.endswith("energy_metrics") else "TABLE"
            ),
            "columns": [
                {
                    "name": "active_energy_voltage",
                    "type_text": "DOUBLE",
                    "comment": "Combined measurement",
                },
                {
                    "name": "voltage_l1",
                    "type_text": "DOUBLE",
                    "comment": "Phase measurement",
                },
                {
                    "name": "consumption_total",
                    "type_text": "DOUBLE",
                    "comment": "Cumulative active energy",
                },
                {
                    "name": "device_id",
                    "type_text": "STRING",
                    "comment": "Meter identifier",
                },
            ]
        }


def _context(*, semantic_layer_enabled: bool = False) -> Any:
    return SimpleNamespace(
        deps=SimpleNamespace(
            catalog=Catalog(),
            schema_name="schema",
            table_column_page_size=10,
            semantic_layer_enabled=semantic_layer_enabled,
        )
    )


def test_find_columns_matches_multiple_keywords_independently() -> None:
    result = find_columns(
        _context(),
        table_name="measurements",
        keywords="energy, voltage",
    )

    assert result.total_matches == 3
    assert [match.name for match in result.matches] == [
        "active_energy_voltage",
        "voltage_l1",
        "consumption_total",
    ]
    assert result.matches[2].matched_in == ["comment"]


def test_catalog_hides_metric_views_without_semantic_capability() -> None:
    result = browse_tables(_context())

    assert [(table.name, table.table_type) for table in result.tables] == [
        ("catalog.schema.measurements", "TABLE")
    ]

    with pytest.raises(ModelRetry, match="not available"):
        describe_table(_context(), "energy_metrics")


def test_catalog_exposes_metric_views_with_semantic_capability() -> None:
    result = browse_tables(_context(semantic_layer_enabled=True))

    assert [(table.name, table.table_type) for table in result.tables] == [
        ("catalog.schema.measurements", "TABLE"),
        ("catalog.schema.energy_metrics", "METRIC_VIEW"),
    ]


def test_parse_columns_classifies_native_metric_fields() -> None:
    table = {
        "table_type": "METRIC_VIEW",
        "columns": [
            {
                "name": "site",
                "type_text": "STRING",
                "type_json": '{"metadata":{"metric_view.type":"dimension"}}',
            },
            {
                "name": "total_energy",
                "type_text": "DOUBLE",
                "type_json": '{"metadata":{"metric_view.type":"measure"}}',
            },
        ],
    }

    columns = parse_columns(table)

    assert [column.metric_role for column in columns] == ["dimension", "measure"]
