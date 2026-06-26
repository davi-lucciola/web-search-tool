from enum import StrEnum


class AllowedAgents(StrEnum):
    GUIDE = 'guide'
    PRODUCT_SEARCH = 'product-search'


class Agents(StrEnum):
    SUPERVISOR = 'supervisor'
    GUIDE = 'guide'
    PRODUCT_SEARCH = 'product-search'


AGENTS_DESCRIPTION: dict[AllowedAgents, str] = {
    AllowedAgents.GUIDE: (
        'Agente que irá recepcionar e guiar o usuário, conduzirá a conversa '
        'explicando sobre o bot até que o deseje realizar alguma operação.'
    ),
    AllowedAgents.PRODUCT_SEARCH: (
        'Agente que irá guiar o fluxo de busca de um produto, ele pedirá '
        'para o usuário principalmente informações como orçamento disponivel '
        'e o problema que ele deseja solucionar. Aqui o usuário também pode achar os links de compra do produto.'
    ),
}
