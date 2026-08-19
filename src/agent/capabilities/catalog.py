from __future__ import annotations

import re
from typing import Literal
from urllib.error import HTTPError

from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import FunctionToolset

from ..catalog import ColumnMetadata, parse_columns
from ..deps import Deps
from ..prompt import render_prompt


class TableSummary(BaseModel):
    name: str
    description: str | None


class BrowseTablesResult(BaseModel):
    schema_name: str
    tables: list[TableSummary]


class ColumnMatch(ColumnMetadata):
    matched_in: list[Literal["name", "comment"]]


class SearchColumnsResult(BaseModel):
    table_name: str
    query: str
    searched_columns: int
    total_matches: int
    matches: list[ColumnMatch]


class DescribeTableResult(BaseModel):
    table_name: str
    description: str | None
    total_columns: int
    columns: list[ColumnMetadata]
    missing_columns: list[str]

NAME = "catalog"
DESCRIPTION = "Discover canonical tables, columns, and schema metadata."
catalog_toolset = FunctionToolset[Deps]()


def build(selected: frozenset[str]) -> Capability[Deps]:
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(f"capabilities/{NAME}/instructions.md", selected),
        toolsets=[catalog_toolset],
    )


def _truncate_description(
    description: str | None,
    max_length: int = 200,
) -> str | None:
    if not description:
        return None
    description = " ".join(description.split())
    if len(description) <= max_length:
        return description
    return f"{description[: max_length - 1].rstrip()}…"


def _normalize_search_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _full_table_name(
    table_name: str,
    catalog_name: str,
    schema_name: str,
) -> str:
    """Fully qualify a bare or schema-qualified table name."""
    resolved_table_name = (
        f"{schema_name}.{table_name}" if "." not in table_name else table_name
    )
    if resolved_table_name.count(".") == 2:
        return resolved_table_name
    return f"{catalog_name}.{resolved_table_name}"


@catalog_toolset.tool
def browse_tables(ctx: RunContext[Deps]) -> BrowseTablesResult:
    """List tables in the configured schema with canonical names and short descriptions.

    Use this when a table name is unknown, ambiguous, or possibly misspelled.
    This tool does not return column metadata.
    """
    deps = ctx.deps
    schema_name = deps.schema_name.split(".")[-1]
    tables = deps.catalog.list_tables(schema_name)
    return BrowseTablesResult(
        schema_name=f"{deps.catalog.name}.{schema_name}",
        tables=[
            TableSummary(
                name=f"{deps.catalog.name}.{schema_name}.{table['name']}",
                description=_truncate_description(table.get("comment")),
            )
            for table in tables
        ],
    )


@catalog_toolset.tool
def find_columns(
    ctx: RunContext[Deps],
    table_name: str,
    keywords: str,
) -> SearchColumnsResult:
    """Search for keywords in column names and metadata from a specific table.

    This searches schema metadata only. It does not search table data or row
    values; use ``query_data`` to filter or search data.

    Args:
        table_name: A bare, schema-qualified, or fully qualified table name.
            Bare names use the configured schema.
        keywords: One or more column-name or metadata keywords. Columns matching
            any keyword in their name or comment are returned, with more keyword
            matches first. Do not provide table data values.
    """
    deps = ctx.deps
    normalized_keywords = _normalize_search_text(keywords)
    if not normalized_keywords:
        raise ModelRetry("Column search keywords cannot be empty.")
    keyword_terms = tuple(dict.fromkeys(normalized_keywords.split()))

    resolved_table_name = _full_table_name(
        table_name,
        deps.catalog.name,
        deps.schema_name,
    )
    try:
        table = deps.catalog.get_table(resolved_table_name)
    except HTTPError as exc:
        if exc.code != 404:
            raise
        raise ModelRetry(
            f"Table {resolved_table_name!r} was not found. "
            "Call browse_tables to find its canonical name."
        ) from exc

    columns = parse_columns(table)
    ranked_matches = []
    for column in columns:
        normalized_name = _normalize_search_text(column.name)
        normalized_comment = _normalize_search_text(column.comment or "")
        name_matches = {
            keyword for keyword in keyword_terms if keyword in normalized_name
        }
        comment_matches = {
            keyword for keyword in keyword_terms if keyword in normalized_comment
        }
        matched_keywords = name_matches | comment_matches
        matched_in: list[Literal["name", "comment"]] = []
        if name_matches:
            matched_in.append("name")
        if comment_matches:
            matched_in.append("comment")
        if matched_in:
            ranked_matches.append(
                (
                    (
                        -len(matched_keywords),
                        0 if normalized_keywords == normalized_name else 1,
                        0 if normalized_name in keyword_terms else 1,
                        0 if name_matches else 1,
                        column.position,
                    ),
                    ColumnMatch(**column.model_dump(), matched_in=matched_in),
                )
            )

    ranked_matches.sort(key=lambda match: match[0])
    matches = [match for _, match in ranked_matches]
    return SearchColumnsResult(
        table_name=resolved_table_name,
        query=keywords,
        searched_columns=len(columns),
        total_matches=len(matches),
        matches=matches[: deps.table_column_page_size],
    )


@catalog_toolset.tool
def describe_table(
    ctx: RunContext[Deps],
    table_name: str,
    start_column: int = 1,
    end_column: int | None = None,
    column_names: list[str] | None = None,
) -> DescribeTableResult:
    """Show the full description and selected column metadata for one known table.

    By default this returns the configured first page of columns. Use
    ``start_column`` and ``end_column`` for a 1-based inclusive range, or
    ``column_names`` for exact identifiers. ``column_names`` takes precedence
    over the range. Use ``find_columns`` for human-language wording.

    Args:
        table_name: A bare, schema-qualified, or fully qualified table name.
            Bare names use the configured schema.
        start_column: First column position to return, 1-based and inclusive.
        end_column: Last column position to return, 1-based and inclusive.
        column_names: Exact column identifiers to return.
    """
    deps = ctx.deps
    resolved_table_name = _full_table_name(
        table_name,
        deps.catalog.name,
        deps.schema_name,
    )
    try:
        table = deps.catalog.get_table(resolved_table_name)
    except HTTPError as exc:
        if exc.code != 404:
            raise
        raise ModelRetry(
            f"Table {resolved_table_name!r} was not found. "
            "Call browse_tables to find its canonical name."
        ) from exc

    columns = parse_columns(table)
    if start_column < 1:
        raise ModelRetry("start_column must be at least 1.")
    if end_column is not None and end_column < start_column:
        raise ModelRetry("end_column must be greater than or equal to start_column.")
    if column_names:
        columns_by_name = {column.name: column for column in columns}
        selected = [
            columns_by_name[name] for name in column_names if name in columns_by_name
        ]
        missing_columns = [name for name in column_names if name not in columns_by_name]
    else:
        last_column = (
            end_column
            if end_column is not None
            else start_column + deps.table_column_page_size - 1
        )
        selected = columns[start_column - 1 : last_column]
        missing_columns = []

    return DescribeTableResult(
        table_name=resolved_table_name,
        description=table.get("comment"),
        total_columns=len(columns),
        columns=selected,
        missing_columns=missing_columns,
    )
