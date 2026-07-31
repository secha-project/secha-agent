import pytest

from src.agent.sql_safety import ReadOnlySqlError, require_read_only_sql


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO target SELECT * FROM source",
        "UPDATE target SET value = 1",
        "DELETE FROM target",
        "MERGE INTO target USING source ON target.id = source.id",
        "DROP TABLE target",
        "CREATE TABLE target AS SELECT 1",
        "WITH rows AS (SELECT 1) INSERT INTO target SELECT * FROM rows",
        "SELECT TRANSFORM(value) USING 'side-effecting-script' FROM source",
        "SELECT 1; SELECT 2",
    ],
)
def test_read_only_guard_rejects_mutating_or_multiple_statements(sql: str) -> None:
    with pytest.raises(ReadOnlySqlError):
        require_read_only_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT '; DROP TABLE target' AS harmless_text",
        "SELECT `delete` FROM measurements;",
        "WITH rows AS (SELECT 1 AS value) SELECT value FROM rows",
        "/* DELETE FROM target */ SELECT 1 -- UPDATE target",
    ],
)
def test_read_only_guard_allows_selects_and_masks_non_executable_text(
    sql: str,
) -> None:
    require_read_only_sql(sql)


def test_read_only_guard_rejects_unterminated_comments_and_quotes() -> None:
    with pytest.raises(ReadOnlySqlError, match="unterminated block comment"):
        require_read_only_sql("SELECT 1 /*")

    with pytest.raises(ReadOnlySqlError, match="unterminated quoted value"):
        require_read_only_sql("SELECT 'value")
