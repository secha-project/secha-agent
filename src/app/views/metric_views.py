from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from src.agent.spark_runtime import concise_spark_error

from ..metric_views import MetricField, MetricView, MetricViewDefinition
from ..models import User
from ..resources import AppResources
from .sidebar import render_sidebar

METRIC_VIEW_MODE_KEY = "metric-view-mode"
METRIC_VIEW_DRAFT_KEY = "metric-view-draft"
METRIC_VIEW_ERROR_KEY = "metric-view-error"
METRIC_VIEW_SELECTION_KEY = "metric-view-selection"
METRIC_VIEW_TABS_KEY = "metric-view-tabs"


def _field_drafts(
    kind: str,
    fields: list[MetricField],
    comments: dict[str, str],
) -> list[dict[str, str]]:
    return [
        {
            "id": f"{kind}-{index}",
            "name": field.name,
            "expr": field.expr,
            "description": comments.get(field.name, ""),
        }
        for index, field in enumerate(fields)
    ]


def _metric_fields(rows: list[dict[str, str]]) -> list[MetricField]:
    return [
        MetricField(name=row["name"].strip(), expr=row["expr"].strip())
        for row in rows
    ]


def _new_draft(view: MetricView, scope: str) -> dict:
    return {
        "scope": scope,
        "dimensions": _field_drafts(
            "dimension", view.definition.dimensions, view.field_comments
        ),
        "measures": _field_drafts(
            "measure", view.definition.measures, view.field_comments
        ),
        "next_id": len(view.definition.dimensions) + len(view.definition.measures),
    }


def _clear_metric_view_error() -> None:
    st.session_state.pop(METRIC_VIEW_ERROR_KEY, None)


def _render_metric_view_error() -> None:
    message = st.session_state.get(METRIC_VIEW_ERROR_KEY)
    if not message:
        return
    with st.container(key="metric-view-error-alert"):
        st.error(message)
        st.button(
            "Dismiss error",
            icon=":material/close:",
            type="tertiary",
            key="dismiss-metric-view-error",
            on_click=_clear_metric_view_error,
        )


def _show_metric_view_list() -> None:
    st.session_state[METRIC_VIEW_MODE_KEY] = "list"
    st.session_state.pop(METRIC_VIEW_DRAFT_KEY, None)
    _cancel_field("dimensions")
    _cancel_field("measures")
    _clear_metric_view_error()
    st.session_state.pop(METRIC_VIEW_TABS_KEY, None)


def _open_metric_view(full_name: str | None) -> None:
    st.session_state[METRIC_VIEW_MODE_KEY] = "create" if full_name is None else "edit"
    st.session_state[METRIC_VIEW_SELECTION_KEY] = full_name
    st.session_state.pop(METRIC_VIEW_DRAFT_KEY, None)
    _cancel_field("dimensions")
    _cancel_field("measures")
    _clear_metric_view_error()
    st.session_state.pop(METRIC_VIEW_TABS_KEY, None)


def _delete_field(kind: str, row_id: str) -> None:
    draft = st.session_state[METRIC_VIEW_DRAFT_KEY]
    draft[kind] = [row for row in draft[kind] if row["id"] != row_id]
    st.session_state[METRIC_VIEW_TABS_KEY] = f"{kind.title()} ({len(draft[kind])})"


def _insert_field(kind: str) -> None:
    draft = st.session_state[METRIC_VIEW_DRAFT_KEY]
    name = st.session_state.get(f"new-{kind}-name", "").strip()
    expr = st.session_state.get(f"new-{kind}-expression", "").strip()
    if not name or not expr:
        st.session_state[f"metric-view-new-{kind}-error"] = (
            "Name and SQL expression are required."
        )
        return

    row_id = f"{kind}-{draft['next_id']}"
    draft["next_id"] += 1
    draft[kind].append(
        {
            "id": row_id,
            "name": name,
            "expr": expr,
            "description": st.session_state.get(
                f"new-{kind}-description", ""
            ).strip(),
        }
    )
    _cancel_field(kind)
    st.session_state[METRIC_VIEW_TABS_KEY] = (
        f"{kind.title()} ({len(draft[kind])})"
    )


def _begin_field(kind: str) -> None:
    st.session_state[f"metric-view-new-{kind}"] = True
    st.session_state.pop(f"metric-view-new-{kind}-error", None)


def _cancel_field(kind: str) -> None:
    st.session_state.pop(f"metric-view-new-{kind}", None)
    st.session_state.pop(f"metric-view-new-{kind}-error", None)


def _render_field_editor(kind: str, singular: str) -> list[dict[str, str]]:
    draft = st.session_state[METRIC_VIEW_DRAFT_KEY]
    rows = draft[kind]

    with st.container(horizontal=True, horizontal_alignment="right"):
        st.button(
            f"Add {singular}",
            icon=":material/add:",
            key=f"add-{kind}",
            on_click=_begin_field,
            args=(kind,),
        )

    if st.session_state.get(f"metric-view-new-{kind}"):
        with st.container(border=True):
            st.markdown(f"**New {singular}**")
            st.text_input(
                "Name",
                placeholder=f"{singular}_name",
                key=f"new-{kind}-name",
            )
            st.text_area(
                "SQL expression",
                placeholder="Spark SQL expression",
                key=f"new-{kind}-expression",
            )
            st.text_input(
                "Description",
                placeholder="Optional description",
                key=f"new-{kind}-description",
            )
            with st.container(horizontal=True):
                st.button(
                    f"Insert {singular}",
                    type="primary",
                    icon=":material/add:",
                    key=f"insert-{kind}",
                    on_click=_insert_field,
                    args=(kind,),
                )
                st.button(
                    "Cancel",
                    type="tertiary",
                    key=f"cancel-{kind}",
                    on_click=_cancel_field,
                    args=(kind,),
                )
            if error := st.session_state.get(f"metric-view-new-{kind}-error"):
                st.error(error)

    if not rows:
        st.caption(f"No {kind} yet.")

    for row in rows:
        with st.container(key=f"metric-field-row-{kind}-{row['id']}"):
            label = row["name"] or f"New {singular}"
            with st.expander(label):
                row["name"] = st.text_input(
                    "Name",
                    value=row["name"],
                    key=f"{kind}-{row['id']}-name",
                )
                row["expr"] = st.text_area(
                    "SQL expression",
                    value=row["expr"],
                    key=f"{kind}-{row['id']}-expression",
                )
                row["description"] = st.text_input(
                    "Description",
                    value=row["description"],
                    key=f"{kind}-{row['id']}-description",
                )
            st.button(
                f"Delete {singular}",
                icon=":material/delete:",
                type="tertiary",
                key=f"delete-metric-field-{kind}-{row['id']}",
                on_click=_delete_field,
                args=(kind, row["id"]),
            )
    return rows


def _render_metric_view_list(tables: list[dict]) -> None:
    with st.container(
        horizontal=True,
        horizontal_alignment="right",
        vertical_alignment="center",
    ):
        st.button(
            "Create metric view",
            type="primary",
            icon=":material/add:",
            on_click=_open_metric_view,
            args=(None,),
        )

    if not tables:
        st.caption("No metric views in the catalog.")
        return

    for table in tables:
        full_name = str(table.get("full_name") or "")
        if not full_name:
            continue
        name = str(table.get("name") or full_name.rsplit(".", 1)[-1])
        with st.container(border=True):
            st.button(
                name,
                icon=":material/analytics:",
                help=full_name,
                key=f"open-metric-view-{full_name}",
                on_click=_open_metric_view,
                args=(full_name,),
                width="stretch",
            )
            st.caption(full_name)


def _empty_view() -> MetricView:
    return MetricView(
        full_name="",
        name="",
        definition=MetricViewDefinition.model_construct(source=""),
    )


def render_metric_views(resources: AppResources, user: User) -> None:
    render_sidebar(resources, user)
    st.title("Metric views")

    try:
        tables = resources.metric_views.list_metric_views()
    except Exception as exc:  # noqa: BLE001 - UI boundary reports REST failures.
        st.error(f"Unable to load metric views from Unity Catalog: {exc}")
        return

    if st.session_state.get(METRIC_VIEW_MODE_KEY, "list") == "list":
        _render_metric_view_list(tables)
        return

    selected = st.session_state.get(METRIC_VIEW_SELECTION_KEY)
    try:
        view = (
            _empty_view()
            if selected is None
            else resources.metric_views.load(selected)
        )
    except (ValueError, ValidationError) as exc:
        st.error(f"Unable to read {selected}: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - UI boundary reports REST failures.
        st.error(f"Unable to load {selected} from Unity Catalog: {exc}")
        return

    widget_scope = view.full_name or "new"
    draft = st.session_state.get(METRIC_VIEW_DRAFT_KEY)
    if not isinstance(draft, dict) or draft.get("scope") != widget_scope:
        draft = _new_draft(view, widget_scope)
        st.session_state[METRIC_VIEW_DRAFT_KEY] = draft

    st.button(
        "All metric views",
        icon=":material/arrow_back:",
        type="tertiary",
        on_click=_show_metric_view_list,
    )
    with st.container():
        source_tab, dimensions_tab, measures_tab, settings_tab = st.tabs(
            [
                "Source",
                f"Dimensions ({len(draft['dimensions'])})",
                f"Measures ({len(draft['measures'])})",
                "Settings",
            ],
            key=METRIC_VIEW_TABS_KEY,
            on_change="rerun",
        )
        with source_tab:
            source = st.text_area(
                "Source table or SQL query",
                value=view.definition.source,
                placeholder="analytics.bronze.source_table",
            )
            filter_expression = st.text_area(
                "Filter",
                value=view.definition.filter or "",
                placeholder="Optional Spark SQL predicate",
            )
        with dimensions_tab:
            dimension_rows = _render_field_editor("dimensions", "dimension")
        with measures_tab:
            measure_rows = _render_field_editor("measures", "measure")
        with settings_tab:
            name = st.text_input(
                "View name",
                value=view.name,
                disabled=bool(view.full_name),
                placeholder="sales_metrics",
            )
            comment = st.text_area("Description", value=view.comment)
        submitted = st.button(
            "Save metric view",
            type="primary",
            icon=":material/save:",
            key=f"save-metric-view-{widget_scope}",
            on_click=_clear_metric_view_error,
        )

    if not submitted:
        _render_metric_view_error()
        return

    try:
        definition = MetricViewDefinition(
            source=source.strip(),
            filter=filter_expression.strip() or None,
            dimensions=_metric_fields(dimension_rows),
            measures=_metric_fields(measure_rows),
        )
        saved_name = view.name or name.strip()
        full_name = resources.metric_views.save(
            saved_name,
            definition,
            comment=comment.strip(),
            properties=view.properties,
            field_comments={
                row["name"].strip(): row["description"].strip()
                for row in [*dimension_rows, *measure_rows]
                if row["description"].strip()
            },
        )
    except (ValueError, ValidationError) as exc:
        st.session_state[METRIC_VIEW_ERROR_KEY] = (
            f"Check the metric view definition: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary reports Spark failures.
        st.session_state[METRIC_VIEW_ERROR_KEY] = (
            f"Spark could not save the metric view: {concise_spark_error(exc)}"
        )
    else:
        st.success(f"Saved {full_name} to Unity Catalog and its YAML documentation.")
        return
    _render_metric_view_error()
