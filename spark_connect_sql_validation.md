# Validating Spark SQL with `df.schema`

For Spark Connect, a practical way to validate SQL without executing the query result is:

```python
df = spark.sql(sql)
schema = df.schema
```

Accessing `df.schema` asks the Spark server to parse and analyze the SQL so it can determine the output columns and data types. If analysis succeeds, Spark returns a `StructType`. If the SQL is invalid, Spark raises a parse or analysis exception.

## Why use `df.schema`?

It provides a useful balance between speed and validation coverage.

It normally validates:

- SQL syntax
- Table and view resolution
- Column resolution
- Function resolution
- Type compatibility
- The resulting output schema

It normally does **not**:

- Scan or return table rows
- Run the full query workload
- Detect failures that depend on actual data values
- Guarantee that the query will complete successfully under production load

Compared with `df.explain()`, requesting the schema usually performs less work because Spark does not need to produce and format a physical plan.

With Spark Connect, this is preferable to JVM-internal approaches such as `_jsparkSession`, because those embedded-JVM APIs are not available to the Connect client.

## Example implementation

```python
from dataclasses import dataclass
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType


@dataclass(frozen=True)
class SqlValidationResult:
    valid: bool
    schema: Optional[StructType] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def validate_sql(
    spark: SparkSession,
    sql: str,
) -> SqlValidationResult:
    if not sql.strip():
        return SqlValidationResult(
            valid=False,
            error_type="EmptySql",
            error_message="SQL text is empty.",
        )

    try:
        schema = spark.sql(sql).schema

        return SqlValidationResult(
            valid=True,
            schema=schema,
        )

    except Exception as exc:
        get_error_class = getattr(exc, "getErrorClass", None)
        error_type = (
            get_error_class()
            if callable(get_error_class)
            else type(exc).__name__
        )

        return SqlValidationResult(
            valid=False,
            error_type=error_type,
            error_message=str(exc),
        )
```

Usage:

```python
result = validate_sql(
    spark,
    "SELECT customer_id, total_amount FROM sales.orders",
)

if result.valid:
    print(result.schema.simpleString())
else:
    print(result.error_type)
    print(result.error_message)
```

A valid query might produce:

```text
struct<customer_id:bigint,total_amount:decimal(18,2)>
```

An invalid query does not return an empty schema. Instead, accessing `schema` raises an exception such as a parse error, unresolved table error, or unresolved column error.

## Safety and limitations

Use a read-only Spark identity when SQL comes from users. SQL can contain commands other than `SELECT`, and client-side validation should not be treated as a security boundary.

For high-volume validation, keep the Spark Connect session alive, limit concurrency, and consider using a local parser such as SQLGlot as a fast first-pass filter. Spark should remain the authoritative validator when catalog, columns, functions, permissions, and actual output types matter.
