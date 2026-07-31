from __future__ import annotations

import re


class ReadOnlySqlError(ValueError):
    pass


_ALLOWED_FIRST_KEYWORDS = frozenset({"SELECT", "WITH"})
_MUTATING_KEYWORDS = frozenset(
    {
        "ADD",
        "ALTER",
        "CACHE",
        "CALL",
        "COPY",
        "CREATE",
        "DELETE",
        "DROP",
        "EXPORT",
        "GRANT",
        "IMPORT",
        "INSERT",
        "LOAD",
        "MERGE",
        "MSCK",
        "OPTIMIZE",
        "REFRESH",
        "REMOVE",
        "REPAIR",
        "REPLACE",
        "REVOKE",
        "SET",
        "TRANSFORM",
        "TRUNCATE",
        "UNCACHE",
        "UPDATE",
        "USE",
        "VACUUM",
    }
)
_WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def require_read_only_sql(sql: str) -> None:
    """Reject SQL that is not a single SELECT or read-only CTE statement."""
    masked = _mask_literals_comments_and_identifiers(sql)
    statement = masked.strip()
    if statement.endswith(";"):
        statement = statement[:-1].rstrip()
    if not statement or ";" in statement:
        raise ReadOnlySqlError("Only one SQL statement is allowed.")

    keywords = [word.upper() for word in _WORD_PATTERN.findall(statement)]
    if not keywords or keywords[0] not in _ALLOWED_FIRST_KEYWORDS:
        raise ReadOnlySqlError(
            "Only SELECT queries and read-only CTE queries are allowed."
        )

    mutation = next(
        (word for word in keywords if word in _MUTATING_KEYWORDS),
        None,
    )
    if mutation is not None:
        raise ReadOnlySqlError(
            f"SQL keyword {mutation!r} is not allowed in a read-only query."
        )


def _mask_literals_comments_and_identifiers(sql: str) -> str:
    """Mask quoted text and comments so only executable SQL tokens remain."""
    masked: list[str] = []
    index = 0
    length = len(sql)

    while index < length:
        char = sql[index]
        following = sql[index + 1] if index + 1 < length else ""

        if char == "-" and following == "-":
            masked.extend((" ", " "))
            index += 2
            while index < length and sql[index] not in "\r\n":
                masked.append(" ")
                index += 1
            continue

        if char == "/" and following == "*":
            masked.extend((" ", " "))
            index += 2
            depth = 1
            while index < length and depth:
                char = sql[index]
                following = sql[index + 1] if index + 1 < length else ""
                if char == "/" and following == "*":
                    depth += 1
                    masked.extend((" ", " "))
                    index += 2
                elif char == "*" and following == "/":
                    depth -= 1
                    masked.extend((" ", " "))
                    index += 2
                else:
                    masked.append("\n" if char == "\n" else " ")
                    index += 1
            if depth:
                raise ReadOnlySqlError("SQL contains an unterminated block comment.")
            continue

        if char in {"'", '"', "`"}:
            quote = char
            masked.append(" ")
            index += 1
            while index < length:
                char = sql[index]
                following = sql[index + 1] if index + 1 < length else ""
                masked.append("\n" if char == "\n" else " ")
                index += 1
                if char == "\\" and index < length:
                    masked.append(" ")
                    index += 1
                elif char == quote and following == quote:
                    masked.append(" ")
                    index += 1
                elif char == quote:
                    break
            else:
                raise ReadOnlySqlError("SQL contains an unterminated quoted value.")
            continue

        masked.append(char)
        index += 1

    return "".join(masked)
