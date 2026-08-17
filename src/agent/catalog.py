import json
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel


class ColumnMetadata(BaseModel):
    position: int
    name: str
    data_type: str
    comment: str | None


class UnityCatalogClient:
    def __init__(self, url: str, token: str, catalog_name: str):
        self._url = url
        self._token = token
        self.name = catalog_name

    def _get(self, path: str, params: dict[str, str] | None = None) -> dict:
        url = f"{self._url.rstrip('/')}/api/2.1/unity-catalog{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
        )
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def list_tables(self, schema_name: str) -> list[dict]:
        """Fetch lightweight table metadata for every table in a schema."""
        params = {
            "catalog_name": self.name,
            "schema_name": schema_name,
            "omit_columns": "true",
            "omit_properties": "true",
        }
        tables = []
        while True:
            page = self._get("/tables", params)
            tables.extend(page.get("tables") or [])
            page_token = page.get("next_page_token")
            if not page_token:
                return tables
            params["page_token"] = page_token

    def get_table(self, full_table_name: str) -> dict:
        """Fetch a table and all of its column metadata."""
        return self._get(f"/tables/{quote(full_table_name, safe='.')}")


def parse_columns(table: dict) -> list[ColumnMetadata]:
    return [
        ColumnMetadata(
            position=position,
            name=column["name"],
            data_type=column.get("type_text") or column.get("type_name") or "unknown",
            comment=column.get("comment"),
        )
        for position, column in enumerate(table.get("columns") or [], start=1)
    ]
