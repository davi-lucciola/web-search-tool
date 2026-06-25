import asyncio

from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import HumanMessage

from rich import print

from app.agent import build_agent


async def main():
    agent = build_agent()

    entry = InputAgentState(messages=[HumanMessage('Quais são as últimas notícias sobre inteligência artificial?')])
    response = await agent.ainvoke(entry)

    print(response)


if __name__ == '__main__':
    asyncio.run(main())
