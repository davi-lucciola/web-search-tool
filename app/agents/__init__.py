from langgraph.graph import END, START, StateGraph
from app.agents.guide import guide_agent
from app.agents.search import search_agent
from app.agents.states import ChatState
from app.agents.constants import Agents, AllowedAgents
from app.agents.supervisor import supervisor_agent


def build_agent():
    builder = StateGraph(state_schema=ChatState)

    builder.add_node(Agents.SUPERVISOR, supervisor_agent)
    builder.add_node(Agents.GUIDE, guide_agent)
    builder.add_node(Agents.SEARCH, search_agent)

    builder.add_edge(START, Agents.SUPERVISOR)

    def get_next(state: ChatState):
        return state['next']

    builder.add_conditional_edges(
        Agents.SUPERVISOR, get_next, {k: k for k in AllowedAgents}
    )
    builder.add_edge(list(Agents), END)

    return builder.compile()
