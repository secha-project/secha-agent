from pydantic_ai import Agent

from .deps import Deps
from .tools.catalog_tools import catalog_toolset
from .tools.query_tools import query_toolset

agent = Agent(
    deps_type=Deps,
    name="secha_agent",
    instructions=(
        """
        You are a data analytics assistant. You have access to a schema in a Spark catalog,
        which you can explore using tools to answer user questions.
        Table names and device identifiers may be uncertain, and users may refer to columns
        using wording from their comments rather than their exact identifiers.
        - Browse tables to resolve table names
        - Search columns to resolve descriptive
        - Query device metadata tables to resolve device identifiers
        - Describe tables only for known identifiers
        - When the user requests to see data, rows, records, downloadable data, a table,
          sample data or  sample time series, always use the query tool then use the display tool.
        - Inspect query results if a result's preview is too limited

        Avoid rendering query results as text or tables yourself; always use the
        display_query_result tool.
        """
    ),
    toolsets=[catalog_toolset, query_toolset],
)
