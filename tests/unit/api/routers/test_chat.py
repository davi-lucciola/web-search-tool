from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pytest_mock import MockerFixture
from sse_starlette.sse import EventSourceResponse

from app.api.routers.chat import chat
from app.api.schemas.chat import ChatRequest

CONFIG: Any = {'configurable': {'thread_id': 't1'}}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _aiter(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def make_agent(mocker: MockerFixture, *, interrupts: list[Any] | None = None) -> Any:
    """Agente mockado: `aget_state` awaited retorna snapshot com `.interrupts`."""
    snapshot = SimpleNamespace(values={'messages': []}, interrupts=interrupts)
    agent = mocker.Mock()
    agent.aget_state = mocker.AsyncMock(return_value=snapshot)
    return agent


# --------------------------------------------------------------------------- #
# chat (endpoint, async)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_chat_new_message_when_no_pending_interrupt(
    mocker: MockerFixture,
) -> None:
    agent = make_agent(mocker, interrupts=None)
    stream = mocker.patch(
        'app.core.services.chat.event_stream', return_value=_aiter([])
    )

    response = await chat('t1', ChatRequest(message='quero um celular'), agent)

    assert isinstance(response, EventSourceResponse)
    agent.aget_state.assert_awaited_once_with(CONFIG)
    graph_input = stream.call_args.args[1]
    assert graph_input['next'] == ''
    assert isinstance(graph_input['messages'][0], HumanMessage)
    assert graph_input['messages'][0].content == 'quero um celular'
    assert stream.call_args.args[2] == CONFIG


@pytest.mark.anyio
async def test_chat_resumes_when_pending_interrupt(mocker: MockerFixture) -> None:
    agent = make_agent(mocker, interrupts=[SimpleNamespace(value='?')])
    stream = mocker.patch(
        'app.core.services.chat.event_stream', return_value=_aiter([])
    )

    response = await chat('t1', ChatRequest(message='2000 reais'), agent)

    assert isinstance(response, EventSourceResponse)
    graph_input = stream.call_args.args[1]
    assert isinstance(graph_input, Command)
    assert graph_input.resume == '2000 reais'
