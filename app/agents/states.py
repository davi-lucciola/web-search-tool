from decimal import Decimal
from typing import Annotated

from langchain.agents import AgentState


def take_latest_nonempty(current: str, update: str) -> str:
    return update if update else (current or '')


NextNode = Annotated[str, take_latest_nonempty]


class ChatState(AgentState):
    next: NextNode


class ProductSearch(AgentState):
    budget: Decimal
    products: list[str]
