from enum import StrEnum


class AllowedAgents(StrEnum):
    GUIDE = 'guide'
    SEARCH = 'search'


class Agents(StrEnum):
    GUIDE = 'guide'
    SEARCH = 'search'
    SUPERVISOR = 'supervisor'


AGENTS_DESCRIPTION: dict[AllowedAgents, str] = {
    AllowedAgents.GUIDE: (
        'Agente que irá recepcionar e guiar o usuário, conduzirá a conversa '
        'explicando sobre o bot até que o deseje realizar alguma operação.'
    ),
    AllowedAgents.SEARCH: (
        'Agente que irá guiar o fluxo de busca de um produto, ele pedirá '
        'para o usuário principalmente informações como orçamento disponivel '
        'e o problema que ele deseja solucionar. Aqui o usuário também pode achar os links de compra do produto.'
    ),
}
