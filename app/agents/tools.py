from typing import cast

import json
from langchain.tools import tool

from rich.console import Console
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
    response = await client.search(query, contry='brazil')

    Console().print_json(json.dumps(response), indent=4)
    return cast(TavilySearchResponse, response)
