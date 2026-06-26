from langchain_core.messages import AIMessage

from app.agents.states import ChatState


async def search_agent(state: ChatState):
    return {
        'messages': [
            AIMessage(
                'Não estou com acesso a realizar essa pesquisa no momento, tente mais tarde.'
            )
        ]
    }
