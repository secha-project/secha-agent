from types import SimpleNamespace
from typing import Any

from src.agent.capabilities.catalog import catalog_toolset, find_columns


class Catalog:
    name = "catalog"

    def get_table(self, full_table_name: str) -> dict[str, Any]:
        assert full_table_name == "catalog.schema.measurements"
        return {
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


def _context() -> Any:
    return SimpleNamespace(
        deps=SimpleNamespace(
            catalog=Catalog(),
            schema_name="schema",
            table_column_page_size=10,
        )
    )


def test_find_columns_exposes_requested_name_and_description() -> None:
    tool = catalog_toolset.tools["find_columns"]

    assert "search_columns" not in catalog_toolset.tools
    assert tool.description == (
        "Search for keywords in column names and metadata from a specific table.\n\n"
        "This searches schema metadata only. It does not search table data or row\n"
        "values; use ``query_data`` to filter or search data."
    )
    assert tool.function_schema.json_schema["required"] == [
        "table_name",
        "keywords",
    ]


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
