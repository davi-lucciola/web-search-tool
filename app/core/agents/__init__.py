from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.core.agents.constants import Agents, Nodes
from app.core.agents.guide import build_guide_node
from app.core.agents.products import build_product_search_node
from app.core.agents.states import ChatState
from app.core.agents.supervisor import build_supervisor_node


def build_agent(checkpointer: BaseCheckpointSaver[Any] | None = None):
    builder = StateGraph(state_schema=ChatState)

    builder.add_node(Nodes.SUPERVISOR, build_supervisor_node())
    builder.add_node(Nodes.GUIDE, build_guide_node())
    builder.add_node(Nodes.PRODUCTS, build_product_search_node())

    builder.add_edge(START, Nodes.SUPERVISOR)

    def get_next(state: ChatState) -> str:
        return state.get('next', '')

    builder.add_conditional_edges(Nodes.SUPERVISOR, get_next, {k: k for k in Agents})
    builder.add_edge([Nodes.GUIDE, Nodes.PRODUCTS], END)

    return builder.compile(checkpointer=checkpointer)


def make_graph():
    return build_agent()
