from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from src.app.metric_views import (
    MetricField,
    MetricViewDefinition,
    MetricViewService,
    build_create_metric_view_sql,
)
from src.app.views import metric_views as metric_views_view
from src.app.views.metric_views import (
    METRIC_VIEW_LIST_COMPONENT_KEY,
    _field_drafts,
    _insert_field,
    _load_metric_view_yaml,
    _metric_fields,
    _open_metric_view_from_table,
)


def test_metric_field_drafts_round_trip_editor_values() -> None:
    drafts = _field_drafts(
        "dimension",
        [MetricField(name="region", expr="region_name")],
        {"region": "Sales region"},
    )

    assert drafts[0]["description"] == "Sales region"
    assert _metric_fields(drafts) == [
        MetricField(name="region", expr="region_name")
    ]


def test_inserting_field_keeps_its_tab_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {
        "metric-view-draft": {
            "dimensions": [],
            "measures": [],
            "next_id": 0,
        },
        "metric-view-new-dimensions": True,
        "new-dimensions-name": "region",
        "new-dimensions-expression": "region_name",
        "new-dimensions-description": "Sales region",
    }
    monkeypatch.setattr(metric_views_view.st, "session_state", state)

    _insert_field("dimensions")

    assert state["metric-view-draft"]["dimensions"][0]["name"] == "region"
    assert state["metric-view-tabs"] == "Dimensions (1)"
    assert "metric-view-new-dimensions" not in state


def test_metric_view_table_event_opens_without_url_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {METRIC_VIEW_LIST_COMPONENT_KEY: {"selected": "analytics.bronze.mv"}}
    monkeypatch.setattr(metric_views_view.st, "session_state", state)

    _open_metric_view_from_table()

    assert state["metric-view-mode"] == "edit"
    assert state["metric-view-selection"] == "analytics.bronze.mv"


def test_loading_yaml_replaces_editor_values_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "metric-view-yaml-new": {
            "value": (
                "version: '0.1'\n"
                "source: analytics.bronze.orders\n"
                "filter: status = 'paid'\n"
                "dimensions:\n  - name: region\n    expr: region_name\n"
                "measures:\n  - name: revenue\n    expr: sum(amount)\n"
            )
        },
        "metric-view-draft": {
            "scope": "new",
            "dimensions": [
                {"id": "dimension-0", "name": "old", "expr": "old"}
            ],
            "measures": [],
            "next_id": 1,
        },
        "dimensions-dimension-0-name": "old",
    }
    monkeypatch.setattr(metric_views_view.st, "session_state", state)

    _load_metric_view_yaml("new")

    assert state["metric-view-source-new"] == "analytics.bronze.orders"
    assert state["metric-view-filter-new"] == "status = 'paid'"
    assert state["metric-view-draft"]["dimensions"][0]["name"] == "region"
    assert state["metric-view-draft"]["measures"][0]["name"] == "revenue"
    assert state["metric-view-tabs"] == "Source"
    assert "dimensions-dimension-0-name" not in state


def test_invalid_yaml_keeps_the_current_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = {"scope": "new", "dimensions": [], "measures": [], "next_id": 0}
    state = {"metric-view-yaml-new": "source: [", "metric-view-draft": draft}
    monkeypatch.setattr(metric_views_view.st, "session_state", state)

    _load_metric_view_yaml("new")

    assert state["metric-view-draft"] is draft
    assert state["metric-view-yaml-error"].startswith("Unable to load YAML:")


def test_metric_definition_round_trips_the_oss_yaml_contract() -> None:
    definition = MetricViewDefinition(
        source="analytics.bronze.orders",
        filter="status = 'paid'",
        dimensions=[MetricField(name="region", expr="region")],
        measures=[MetricField(name="revenue", expr="sum(amount)")],
    )

    assert MetricViewDefinition.from_yaml(definition.to_yaml()) == definition
    with pytest.raises(ValidationError):
        MetricViewDefinition.from_yaml(
            "version: '0.1'\nsource: orders\nwhere: status = 'paid'\n"
        )


def test_metric_definition_exports_multiline_sql_as_a_literal_block() -> None:
    definition = MetricViewDefinition(
        source="analytics.bronze.orders",
        measures=[
            MetricField(
                name="revenue",
                expr="SUM(\n  price * quantity\n)",
            )
        ],
    )

    exported = definition.to_yaml()

    assert "expr: |-\n    SUM(\n      price * quantity\n    )" in exported
    assert "\\n" not in exported
    assert MetricViewDefinition.from_yaml(exported) == definition


def test_create_sql_keeps_metadata_outside_yaml() -> None:
    sql = build_create_metric_view_sql(
        "analytics.bronze.sales_metrics",
        MetricViewDefinition(
            source="analytics.bronze.orders",
            measures=[MetricField(name="revenue", expr="sum(amount)")],
        ),
        comment="Sales team's metrics",
        properties={"owner": "analytics"},
        field_comments={"revenue": "Paid revenue"},
    )

    assert sql.startswith("CREATE VIEW `analytics`.`bronze`.`sales_metrics`")
    assert "`revenue` COMMENT 'Paid revenue'" in sql
    assert "COMMENT 'Sales team''s metrics'" in sql
    assert "TBLPROPERTIES ('owner' = 'analytics')" in sql
    assert "WITH METRICS\nLANGUAGE YAML\nAS $$" in sql
    yaml_body = sql.split("$$", 2)[1]
    assert "comment" not in yaml_body
    assert "properties" not in yaml_body


def test_create_sql_uses_a_valid_tag_when_yaml_contains_double_dollars() -> None:
    sql = build_create_metric_view_sql(
        "analytics.bronze.quoted_source",
        MetricViewDefinition(source="SELECT '$$' AS marker"),
    )

    assert "AS $metricview$" in sql
    assert sql.endswith("$metricview$")


def test_service_filters_catalog_loads_definition_and_saves_documentation(
    tmp_path: Path,
) -> None:
    executions: list[str] = []
    spark = SimpleNamespace(
        sql=lambda sql: SimpleNamespace(
            collect=lambda: executions.append(sql),
        )
    )
    catalog = SimpleNamespace(
        name="analytics",
        list_tables=lambda _schema: [
            {"full_name": "analytics.bronze.orders", "table_type": "TABLE"},
            {
                "name": "sales_metrics",
                "table_type": "METRIC_VIEW",
            },
        ],
        get_table=lambda _name: {
            "name": "sales_metrics",
            "comment": "Sales metrics",
            "properties": {"owner": "analytics"},
            "view_definition": (
                "version: '0.1'\nsource: analytics.bronze.orders\n"
                "measures:\n  - name: revenue\n    expr: sum(amount)\n"
            ),
            "columns": [{"name": "revenue", "comment": "Paid revenue"}],
        },
    )
    service = MetricViewService(
        cast(Any, spark), cast(Any, catalog), "bronze", tmp_path
    )

    assert service.list_metric_views() == [
        {
            "name": "sales_metrics",
            "full_name": "analytics.bronze.sales_metrics",
            "table_type": "METRIC_VIEW",
        }
    ]
    loaded = service.load("analytics.bronze.sales_metrics")
    assert loaded.definition.measures[0].name == "revenue"
    assert loaded.field_comments == {"revenue": "Paid revenue"}

    full_name = service.save(
        "sales_metrics",
        loaded.definition,
        comment=loaded.comment,
        properties=loaded.properties,
        field_comments=loaded.field_comments,
    )
    assert full_name == "analytics.bronze.sales_metrics"
    assert len(executions) == 1
    documentation = (
        tmp_path / "analytics.bronze.sales_metrics.yaml"
    ).read_text()
    assert documentation.startswith("# catalog_metadata:")
    assert "#   description: Sales metrics" in documentation
    assert MetricViewDefinition.from_yaml(documentation) == loaded.definition
