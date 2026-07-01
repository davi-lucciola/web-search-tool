from typing import Any

from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from app.api.deps import Agent, get_agent
from app.api.schemas.chat import ChatRequest
from app.core.services import chat as chat_service

router = APIRouter(prefix='/threads', tags=['chat'])


@router.post('/{thread_id}/chat')
async def chat(
    thread_id: str,
    req: ChatRequest,
    agent: Agent = Depends(get_agent),
) -> EventSourceResponse:
    config: RunnableConfig = {'configurable': {'thread_id': thread_id}}

    snapshot = await agent.aget_state(config)
    pending = bool(getattr(snapshot, 'interrupts', None))

    graph_input: Any = (
        Command(resume=req.message)
        if pending
        else {'messages': [HumanMessage(req.message)], 'next': ''}
    )

    return EventSourceResponse(chat_service.event_stream(agent, graph_input, config))
