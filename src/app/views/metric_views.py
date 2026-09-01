from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st
import yaml
from pydantic import ValidationError

from src.agent.spark_runtime import concise_spark_error

from ..metric_views import MetricField, MetricView, MetricViewDefinition
from ..models import User
from ..resources import AppResources
from .sidebar import render_sidebar

METRIC_VIEW_MODE_KEY = "metric-view-mode"
METRIC_VIEW_DRAFT_KEY = "metric-view-draft"
METRIC_VIEW_ERROR_KEY = "metric-view-error"
METRIC_VIEW_LIST_COMPONENT_KEY = "metric-view-list-component"
METRIC_VIEW_SELECTION_KEY = "metric-view-selection"
METRIC_VIEW_TABS_KEY = "metric-view-tabs"
METRIC_VIEW_YAML_ERROR_KEY = "metric-view-yaml-error"

_METRIC_VIEW_TABLE_HTML = """
<div class="metric-view-table-wrap">
  <table class="metric-view-table">
    <thead><tr><th>Name</th><th>Created At</th></tr></thead>
    <tbody></tbody>
  </table>
</div>
"""

_METRIC_VIEW_TABLE_CSS = """
.metric-view-table-wrap {
  border: 1px solid var(--st-border-color);
  border-radius: var(--st-base-radius);
  overflow-x: auto;
}
.metric-view-table { border-collapse: collapse; color: var(--st-text-color); width: 100%; }
.metric-view-table th {
  background: var(--st-dataframe-header-background-color);
  font-weight: 600;
  text-align: left;
}
.metric-view-table th:first-child { width: 75%; }
.metric-view-table th,
.metric-view-table td {
  border-bottom: 1px solid var(--st-border-color-light);
  padding: 1rem 1.25rem;
}
.metric-view-table tbody tr:last-child td { border-bottom: 0; }
.metric-view-table a {
  align-items: center;
  color: inherit;
  display: inline-flex;
  gap: 0.5rem;
  text-decoration: none;
}
.metric-view-table a:hover { text-decoration: underline; }
"""

_METRIC_VIEW_TABLE_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component
  const tbody = parentElement.querySelector("tbody")
  if (!tbody) return

  tbody.replaceChildren()
  for (const row of data.rows ?? []) {
    const tr = document.createElement("tr")
    const nameCell = document.createElement("td")
    const link = document.createElement("a")
    const icon = document.createElement("span")
    const createdCell = document.createElement("td")

    link.href = "#"
    link.title = row.full_name
    link.onclick = event => {
      event.preventDefault()
      setTriggerValue("selected", row.full_name)
    }
    icon.ariaHidden = "true"
    icon.textContent = "▦"
    link.append(icon, document.createTextNode(row.name))
    nameCell.append(link)
    createdCell.textContent = row.created_at
    tr.append(nameCell, createdCell)
    tbody.append(tr)
  }
}
"""

_METRIC_VIEW_TABLE = st.components.v2.component(
    "metric_view_table",
    html=_METRIC_VIEW_TABLE_HTML,
    css=_METRIC_VIEW_TABLE_CSS,
    js=_METRIC_VIEW_TABLE_JS,
)

_YAML_EDITOR_HTML = """
<label for="yaml-editor">Metric view YAML</label>
<textarea id="yaml-editor" aria-describedby="yaml-editor-help" spellcheck="false"></textarea>
<div id="yaml-editor-help">Tab inserts two spaces. Press Escape, then Tab to leave the editor.</div>
"""

_YAML_EDITOR_CSS = """
label { display: block; margin-bottom: 0.5rem; }
textarea {
  background: var(--st-secondary-background-color);
  border: 1px solid var(--st-widget-border-color);
  border-radius: var(--st-base-radius);
  box-sizing: border-box;
  color: var(--st-text-color);
  font-family: var(--st-code-font);
  font-size: var(--st-code-font-size);
  height: 320px;
  padding: 0.75rem;
  resize: vertical;
  width: 100%;
}
textarea:focus { outline: 2px solid var(--st-primary-color); outline-offset: -1px; }
#yaml-editor-help { font-size: 0.875rem; margin-top: 0.25rem; opacity: 0.7; }
"""

_YAML_EDITOR_JS = """
export default function(component) {
  const { data, parentElement, setStateValue } = component
  const textarea = parentElement.querySelector("textarea")
  if (!textarea) return

  const nextValue = data.value ?? ""
  if (textarea.value !== nextValue) textarea.value = nextValue
  textarea.placeholder = data.placeholder ?? ""

  let allowTabExit = false
  textarea.oninput = event => setStateValue("value", event.target.value)
  textarea.onblur = () => { allowTabExit = false }
  textarea.onkeydown = event => {
    if (event.key === "Escape") {
      allowTabExit = true
      return
    }
    if (event.key !== "Tab") {
      allowTabExit = false
      return
    }
    if (allowTabExit) {
      allowTabExit = false
      return
    }

    event.preventDefault()
    const start = textarea.selectionStart
    const end = textarea.selectionEnd
    const lineStart = textarea.value.lastIndexOf("\\n", start - 1) + 1

    if (event.shiftKey) {
      const leadingSpaces = textarea.value.slice(lineStart).match(/^ {1,2}/)?.[0].length ?? 0
      textarea.setRangeText("", lineStart, lineStart + leadingSpaces, "preserve")
      textarea.setSelectionRange(
        Math.max(lineStart, start - leadingSpaces),
        Math.max(lineStart, end - leadingSpaces),
      )
    } else {
      textarea.setRangeText("  ", start, end, "end")
    }
    setStateValue("value", textarea.value)
  }
}
"""

_YAML_EDITOR = st.components.v2.component(
    "metric_view_yaml_editor",
    html=_YAML_EDITOR_HTML,
    css=_YAML_EDITOR_CSS,
    js=_YAML_EDITOR_JS,
)


def _yaml_editor(*, key: str, placeholder: str) -> None:
    state = st.session_state.get(key, {})
    value = (
        state.get("value", "")
        if isinstance(state, dict)
        else state
        if isinstance(state, str)
        else ""
    )
    _YAML_EDITOR(
        data={"value": value, "placeholder": placeholder},
        default={"value": value},
        key=key,
        on_value_change=lambda: None,
    )


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
    st.session_state.pop(METRIC_VIEW_YAML_ERROR_KEY, None)
    st.session_state.pop(METRIC_VIEW_TABS_KEY, None)


def _open_metric_view(full_name: str | None) -> None:
    st.session_state[METRIC_VIEW_MODE_KEY] = (
        "create" if full_name is None else "edit"
    )
    st.session_state[METRIC_VIEW_SELECTION_KEY] = full_name
    st.session_state.pop(METRIC_VIEW_DRAFT_KEY, None)
    _cancel_field("dimensions")
    _cancel_field("measures")
    _clear_metric_view_error()
    st.session_state.pop(METRIC_VIEW_YAML_ERROR_KEY, None)
    st.session_state.pop(METRIC_VIEW_TABS_KEY, None)


def _open_metric_view_from_table() -> None:
    event = st.session_state.get(METRIC_VIEW_LIST_COMPONENT_KEY)
    if event and event.get("selected"):
        _open_metric_view(str(event["selected"]))


def _load_metric_view_yaml(scope: str) -> None:
    state = st.session_state.get(f"metric-view-yaml-{scope}", "")
    value = state.get("value", "") if isinstance(state, dict) else state
    try:
        definition = MetricViewDefinition.from_yaml(value)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        st.session_state[METRIC_VIEW_YAML_ERROR_KEY] = f"Unable to load YAML: {exc}"
        return

    old_draft = st.session_state.get(METRIC_VIEW_DRAFT_KEY)
    if isinstance(old_draft, dict):
        for kind in ("dimensions", "measures"):
            for row in old_draft.get(kind, []):
                for field in ("name", "expression", "description"):
                    st.session_state.pop(f"{kind}-{row['id']}-{field}", None)

    imported_view = MetricView(
        full_name="",
        name="",
        definition=definition,
    )
    st.session_state[METRIC_VIEW_DRAFT_KEY] = _new_draft(imported_view, scope)
    st.session_state[f"metric-view-source-{scope}"] = definition.source
    st.session_state[f"metric-view-filter-{scope}"] = definition.filter or ""
    st.session_state[METRIC_VIEW_TABS_KEY] = "Source"
    st.session_state.pop(METRIC_VIEW_YAML_ERROR_KEY, None)
    _clear_metric_view_error()
    _cancel_field("dimensions")
    _cancel_field("measures")


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

    _METRIC_VIEW_TABLE(
        data={"rows": _metric_view_table_rows(tables)},
        key=METRIC_VIEW_LIST_COMPONENT_KEY,
        on_selected_change=_open_metric_view_from_table,
    )


def _metric_view_table_rows(tables: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "name": str(table["name"]),
            "full_name": str(table["full_name"]),
            "created_at": _format_created_at(table.get("created_at")),
        }
        for table in tables
    ]


def _format_created_at(value: object) -> str:
    if not isinstance(value, int | float):
        return ""
    return datetime.fromtimestamp(value / 1000, UTC).strftime(
        "%d/%m/%Y, %H:%M:%S"
    )


def _empty_view() -> MetricView:
    return MetricView(
        full_name="",
        name="",
        definition=MetricViewDefinition.model_construct(source=""),
    )


def render_metric_views(resources: AppResources, user: User) -> None:
    render_sidebar(resources, user)
    st.space("medium")
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

    source_key = f"metric-view-source-{widget_scope}"
    filter_key = f"metric-view-filter-{widget_scope}"
    name_key = f"metric-view-name-{widget_scope}"
    comment_key = f"metric-view-comment-{widget_scope}"
    st.session_state.setdefault(source_key, view.definition.source)
    st.session_state.setdefault(filter_key, view.definition.filter or "")
    st.session_state.setdefault(name_key, view.name)
    st.session_state.setdefault(comment_key, view.comment)

    st.button(
        "All metric views",
        icon=":material/arrow_back:",
        type="tertiary",
        on_click=_show_metric_view_list,
    )
    with st.container():
        source_tab, dimensions_tab, measures_tab, settings_tab, yaml_tab = st.tabs(
            [
                "Source",
                f"Dimensions ({len(draft['dimensions'])})",
                f"Measures ({len(draft['measures'])})",
                "Settings",
                "Load from YAML",
            ],
            key=METRIC_VIEW_TABS_KEY,
            on_change="rerun",
        )
        with source_tab:
            source = st.text_area(
                "Source table or SQL query",
                key=source_key,
                placeholder="analytics.bronze.source_table",
            )
            filter_expression = st.text_area(
                "Filter",
                key=filter_key,
                placeholder="Optional Spark SQL predicate",
            )
        with dimensions_tab:
            dimension_rows = _render_field_editor("dimensions", "dimension")
        with measures_tab:
            measure_rows = _render_field_editor("measures", "measure")
        with settings_tab:
            name = st.text_input(
                "View name",
                key=name_key,
                disabled=bool(view.full_name),
                placeholder="sales_metrics",
            )
            comment = st.text_area("Description", key=comment_key)
        with yaml_tab:
            with st.form(f"load-metric-view-yaml-{widget_scope}"):
                _yaml_editor(
                    key=f"metric-view-yaml-{widget_scope}",
                    placeholder=(
                        "version: '0.1'\n"
                        "source: analytics.bronze.orders\n"
                        "dimensions:\n"
                        "  - name: region\n"
                        "    expr: region\n"
                        "measures:\n"
                        "  - name: order_count\n"
                        "    expr: count(*)"
                    ),
                )
                st.form_submit_button(
                    "Load YAML",
                    type="primary",
                    icon=":material/upload_file:",
                    on_click=_load_metric_view_yaml,
                    args=(widget_scope,),
                )
            if error := st.session_state.get(METRIC_VIEW_YAML_ERROR_KEY):
                st.error(error)
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
