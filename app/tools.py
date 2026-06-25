from typing import cast

from langchain.tools import tool

from app.tavily import TavilySearchResponse, get_tavily_client


@tool
async def web_search(query: str) -> TavilySearchResponse:
    """Search the internet for the given query and returns the results

    Args:
        query: the search terms to look up on the internet

    Returns:
        the Tavily search response containing the answer and the list of result sources
    """
    client = get_tavily_client()
    response = await client.search(query)
    return cast(TavilySearchResponse, response)
