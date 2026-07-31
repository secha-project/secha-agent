from uuid import uuid4

from pyspark.sql import DataFrame


class UnknownQueryResultReferenceError(LookupError):
    def __init__(self, ref: str) -> None:
        self.ref = ref
        super().__init__(f"Unknown query-result reference {ref!r}.")


class QueryResultRegistry:
    """Map temporary references to live Spark query results."""

    def __init__(self) -> None:
        self._live: dict[str, DataFrame] = {}

    def create(self, result: DataFrame) -> str:
        ref = f"df-{uuid4().hex}"
        self._live[ref] = result
        return ref

    def resolve(self, ref: str) -> DataFrame:
        cached = self._live.get(ref)
        if cached is not None:
            return cached

        raise UnknownQueryResultReferenceError(ref)


def _rows_to_markdown_sections(
    section_name: str,
    field_names: list[str],
    rows: list[list[object]],
) -> str:
    """Render rows as Markdown sections, using each row's first value in its heading."""
    sections = []
    for section_number, *values in rows:
        fields = "\n".join(
            f"{field_name}: {value}" for field_name, value in zip(field_names, values)
        )
        sections.append(f"## {section_name} {section_number}\n\n{fields}")
    return "\n\n".join(sections)


def render_result_preview(
    result: DataFrame,
    start_row: int = 1,
    end_row: int | None = None,
    column_names: list[str] | None = None,
) -> str:
    """Render a 1-based inclusive range of query-result rows as Markdown sections."""
    range_was_requested = end_row is not None
    end_row = end_row or start_row + 9
    requested_row_count = end_row - start_row + 1
    available_columns = result.schema.fieldNames()
    selected_columns = column_names or available_columns[:10]

    unknown_columns = [
        column_name
        for column_name in selected_columns
        if column_name not in available_columns
    ]
    if unknown_columns:
        raise ValueError(
            f"Unknown result columns: {', '.join(unknown_columns)}. "
            f"Available columns: {', '.join(available_columns)}."
        )

    total_rows = result.count()
    collected_rows = result.select(*selected_columns).limit(end_row).collect()
    rows = collected_rows[start_row - 1 : end_row]
    has_more_rows = total_rows > requested_row_count
    has_more_columns = len(selected_columns) < len(available_columns)
    table_rows = [
        [row_number, *row] for row_number, row in enumerate(rows, start=start_row)
    ]
    table = _rows_to_markdown_sections("Row", selected_columns, table_rows)

    overflow_messages = []
    if has_more_rows:
        row_selection = "requested" if range_was_requested else "default"
        row_label = "row" if len(rows) == 1 else "rows"
        overflow_messages.append(
            f"showing only the {row_selection} {len(rows)} {row_label} "
            f"out of the {total_rows} available"
        )
    if has_more_columns:
        column_selection = "requested" if column_names else "default"
        column_label = "column" if len(selected_columns) == 1 else "columns"
        overflow_messages.append(
            f"showing only the {column_selection} {len(selected_columns)} "
            f"{column_label} "
            f"out of the {len(available_columns)} available"
        )

    if not overflow_messages:
        return table

    footer = f"[{', '.join(overflow_messages)}]"
    return f"{table}\n\n{footer}" if table else footer
