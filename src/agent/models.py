from typing import Literal

from pydantic import BaseModel


class TableSummary(BaseModel):
    name: str
    description: str | None


class BrowseTablesResult(BaseModel):
    schema_name: str
    tables: list[TableSummary]


class ColumnMetadata(BaseModel):
    position: int
    name: str
    data_type: str
    comment: str | None


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


class QueryDataResult(BaseModel):
    dataframe_ref: str
    preview_markdown: str


class InspectQueryResult(BaseModel):
    dataframe_ref: str
    preview_markdown: str
