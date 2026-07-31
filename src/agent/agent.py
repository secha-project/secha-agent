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
        - Inspect query results if a result's preview is too limited
        - Display a query result as a table when the user requests to see data, rows, records,
          a table, or downloadable data

        Query_data does not display data to the user; they return
        result references that you can use to inspect or display the results of a query you made.
        Avoid rendering query results as text or tables yourself; always use the
        display_query_result tool.
        References are temporary and only available while their live session is
        active. A reference from an earlier turn may be reused if it is still available. If a
        reference is unknown, never rerun its historical query merely to recreate the result;
        run it again only when the user explicitly requests fresh data.
        """
    ),
)

agent.tool(browse_tables)
agent.tool(search_columns)
agent.tool(describe_table)
agent.tool(retries=3, sequential=True)(query_data)
agent.tool(sequential=True)(inspect_query_result)
agent.tool(sequential=True)(display_query_result)
