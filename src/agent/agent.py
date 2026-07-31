from pydantic_ai import Agent

from .deps import Deps
from .tools.catalog_tools import browse_tables, describe_table, search_columns
from .tools.query_tools import display_query_result, inspect_query_result, query_data

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
)

agent.tool(browse_tables)
agent.tool(search_columns)
agent.tool(describe_table)
agent.tool(retries=3, sequential=True)(query_data)
agent.tool(sequential=True)(inspect_query_result)
agent.tool(sequential=True)(display_query_result)
