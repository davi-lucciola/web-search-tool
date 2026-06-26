import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from rich import print

from app.agents import build_agent
from app.agents.states import ChatState


async def main():
    agent = build_agent()

    state: ChatState = {
        'messages': [
            HumanMessage('Olá, tudo bom?'),
            AIMessage(
                'Olá! Tudo ótimo, obrigado por perguntar. '
                'Seja bem-vindo(a)! Estou aqui para ajudar você a '
                'encontrar o produto com o melhor custo-benefício para '
                'a sua necessidade. Como posso te ajudar hoje?'
            ),
            HumanMessage(
                'Qual é o melhor celular para tirar fotos que eu posso comprar com 1600 reais?'
            ),
        ],
        'next': '',
    }
    response = await agent.ainvoke(state)

    print(response)


if __name__ == '__main__':
    asyncio.run(main())
