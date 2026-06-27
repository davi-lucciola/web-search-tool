from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def take_latest_nonempty(current: str, update: str) -> str:
    return update if update else (current or "")


NextNode = Annotated[str, take_latest_nonempty]


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next: NextNode
