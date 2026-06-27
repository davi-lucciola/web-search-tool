import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pytest_mock import MockerFixture

from app.agents.guide import agent as guide_agent
from app.agents.guide.agent import build_guide_agent
from app.agents.guide.prompt import GUIDE_SYSTEM_PROMPT
from app.agents.states import ChatState


@pytest.mark.anyio
async def test_returns_ai_message_from_llm(mocker: MockerFixture) -> None:
    ai_message = AIMessage("Olá! Como posso ajudar?")
    ainvoke = mocker.AsyncMock(return_value=ai_message)
    llm = mocker.AsyncMock(BaseChatModel)
    mocker.patch.object(llm, "ainvoke", ainvoke)
    mocker.patch.object(guide_agent, "get_llm", return_value=llm)

    state: ChatState = {"messages": [HumanMessage("Oii")], "next": ""}
    result = await build_guide_agent(state)

    assert result == {"messages": [ai_message]}
    ainvoke.assert_awaited_once()


@pytest.mark.anyio
async def test_prepends_system_prompt_before_state_messages(
    mocker: MockerFixture,
) -> None:
    ainvoke = mocker.AsyncMock(return_value=AIMessage("oi"))
    llm = mocker.AsyncMock(BaseChatModel)
    mocker.patch.object(llm, "ainvoke", ainvoke)
    mocker.patch.object(guide_agent, "get_llm", return_value=llm)

    human_message = HumanMessage("Oii")
    state: ChatState = {"messages": [human_message], "next": ""}
    await build_guide_agent(state)

    ainvoke.assert_awaited_once()
    sent_messages = ainvoke.call_args.args[0]
    system_message = sent_messages[0]
    assert isinstance(system_message, SystemMessage)
    assert system_message.content == GUIDE_SYSTEM_PROMPT
    assert sent_messages[1:] == [human_message]
