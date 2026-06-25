from functools import lru_cache
from typing import TypedDict

from tavily import AsyncTavilyClient

from app.config import Settings


class TavilyImage(TypedDict):
    url: str
    description: str


class TavilyResult(TypedDict):
    title: str
    url: str
    content: str
    score: float
    raw_content: str | None
    favicon: str
    images: list[TavilyImage]


class TavilyAutoParameters(TypedDict):
    topic: str
    search_depth: str


class TavilyUsage(TypedDict):
    credits: int


class TavilySearchResponse(TypedDict):
    query: str
    answer: str
    images: list[TavilyImage]
    results: list[TavilyResult]
    response_time: str
    auto_parameters: TavilyAutoParameters
    usage: TavilyUsage
    request_id: str


@lru_cache
def get_tavily_client() -> AsyncTavilyClient:
    return AsyncTavilyClient(api_key=Settings().TAVILY_API_KEY)  # type: ignore[call-arg]
