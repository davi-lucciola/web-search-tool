import asyncio

from langchain_core.messages import HumanMessage

from rich import print

from app.agents import build_agent
from app.agents.states import ChatState


async def main():
    agent = build_agent()

    state: ChatState = {
        'messages': [HumanMessage('Olá, tudo bom?')],
        'next': '',
    }
    response = await agent.ainvoke(state)

    print(response)


if __name__ == '__main__':
    asyncio.run(main())
