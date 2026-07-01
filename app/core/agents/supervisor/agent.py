from langchain_core.messages import SystemMessage

from app.core.agents.states import ChatState
from app.core.agents.supervisor.prompt import SUPERVISOR_SYSTEM_PROMPT
from app.core.agents.supervisor.schemas import Router
from app.infra.llm import get_llm


async def build_supervisor_agent(state: ChatState) -> ChatState:
    llm = get_llm().with_structured_output(Router)

    system_message = SystemMessage(SUPERVISOR_SYSTEM_PROMPT)
    messages = [system_message, *state['messages']]
    router = await llm.ainvoke(messages)

    assert isinstance(router, Router)
    return {'messages': [], 'next': router.next.value}
