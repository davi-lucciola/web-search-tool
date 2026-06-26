from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, field_validator

from app.agents.states import (
    Product,
    ProductSearchState,
    Requirements,
)
from app.agents.tools import deep_search_purchase_links, search_candidates
from app.llm import get_llm


# --------------------------------------------------------------------------- #
# Helpers de (de)serialização: o estado guarda dicts JSON-native (TypedDict);
# os nós reconstroem os modelos pydantic só quando precisam da lógica/validators.
# --------------------------------------------------------------------------- #
def _requirements_from_state(state: ProductSearchState) -> Requirements | None:
    req = state.get('requirements')
    return Requirements.model_validate(req) if req is not None else None


def _products_from_state(state: ProductSearchState) -> list[Product]:
    return [Product.model_validate(p) for p in state.get('products', [])]


# Nomes dos nós do sub-grafo (também aparecem assim no LangGraph Studio).
NODE_COLLECT = 'collect_requirements'
NODE_SEARCH = 'search_products'
NODE_VALIDATE = 'validate_products'
NODE_PRESENT = 'present_recommendations'
NODE_LINKS = 'search_purchase_links'

# Limite de re-buscas no loop de validação para não cair em loop infinito.
MAX_SEARCH_ATTEMPTS = 2
# Quantidade de produtos apresentados ao usuário.
TOP_PRODUCTS = 3


# --------------------------------------------------------------------------- #
# Coleta de requisitos
# --------------------------------------------------------------------------- #
class CollectedInfo(BaseModel):
    """Modelo achatado de extração (modelos menores erram com schema aninhado)."""

    product_type: str | None = Field(
        default=None, description='Tipo/categoria do produto, ex. "celular".'
    )
    use_case: str | None = Field(
        default=None, description='Para que o usuário vai usar o produto.'
    )
    priorities: list[str] = Field(
        default_factory=list, description='Características mais importantes.'
    )
    brand_preferences: list[str] = Field(
        default_factory=list, description='Marcas preferidas ou a evitar.'
    )
    must_haves: list[str] = Field(
        default_factory=list, description='Requisitos obrigatórios.'
    )
    budget: float | None = Field(
        default=None,
        description='Orçamento máximo do usuário em reais (BRL), se informado. Null se não.',
    )

    @field_validator('priorities', 'brand_preferences', 'must_haves', mode='before')
    @classmethod
    def _coerce_none_to_empty_list(cls, value: object) -> object:
        return value if value is not None else []

    def to_requirements(self) -> Requirements:
        return Requirements(
            product_type=self.product_type,
            use_case=self.use_case,
            priorities=self.priorities,
            brand_preferences=self.brand_preferences,
            must_haves=self.must_haves,
        )


EXTRACT_REQUIREMENTS_PROMPT = """
Você é um consultor de compras. A partir do histórico da conversa, extraia os requisitos do
usuário e o orçamento. Preencha apenas o que o usuário realmente disse; deixe campos vazios/null
quando a informação não estiver presente. Não invente dados. Responda em português do Brasil.
"""


async def _extract_info(messages: list) -> CollectedInfo:
    llm = get_llm().with_structured_output(CollectedInfo, method='function_calling')
    info = await llm.ainvoke([SystemMessage(EXTRACT_REQUIREMENTS_PROMPT), *messages])
    assert isinstance(info, CollectedInfo)
    return info


def _next_question(info: CollectedInfo) -> str | None:
    """Retorna a próxima pergunta a fazer, ou None se já há dados suficientes."""
    if not info.to_requirements().is_complete:
        return (
            'Para eu te ajudar a achar o melhor produto: que tipo de produto você procura '
            'e para que vai usar? Tem alguma característica mais importante (ex.: câmera, '
            'bateria, desempenho)?'
        )
    if info.budget is None:
        return 'Qual é o orçamento máximo que você tem em mente, em reais?'
    return None


async def collect_requirements(state: ProductSearchState):
    info = await _extract_info(list(state['messages']))
    question = _next_question(info)

    new_messages: list = []
    if question is not None:
        answer = interrupt({'type': 'collect', 'question': question})
        human = HumanMessage(str(answer))
        new_messages.append(human)
        # Re-extrai já considerando a resposta recém-dada.
        info = await _extract_info([*state['messages'], human])

    return {
        'messages': new_messages,
        'requirements': info.to_requirements().model_dump(mode='json'),
        'budget': info.budget,
    }


def route_after_collect(state: ProductSearchState):
    requirements = _requirements_from_state(state)
    if (
        requirements is not None
        and requirements.is_complete
        and state.get('budget') is not None
    ):
        return NODE_SEARCH
    return NODE_COLLECT


# --------------------------------------------------------------------------- #
# Busca e validação
# --------------------------------------------------------------------------- #
async def search_products(state: ProductSearchState):
    attempts = state.get('search_attempts', 0)
    requirements = _requirements_from_state(state)
    assert requirements is not None

    refine_hint = None
    if attempts > 0:
        # No re-loop, amplia o leque buscando alternativas e melhores avaliações.
        refine_hint = f'{requirements.product_type} alternativas bem avaliadas custo-benefício 2026'

    products = await search_candidates(requirements, state.get('budget'), refine_hint)
    return {
        'products': [p.model_dump(mode='json') for p in products],
        'search_attempts': attempts + 1,
    }


async def validate_products(state: ProductSearchState):
    """Reflexão: descarta itens fora do orçamento; o roteamento decide se re-busca."""
    products = _products_from_state(state)
    budget = state.get('budget')
    if budget is not None:
        products = [
            p
            for p in products
            if p.estimated_price is None or float(p.estimated_price) <= budget
        ]
    return {'products': [p.model_dump(mode='json') for p in products]}


def route_after_validate(state: ProductSearchState):
    products = state.get('products', [])
    attempts = state.get('search_attempts', 0)
    if len(products) >= TOP_PRODUCTS or attempts >= MAX_SEARCH_ATTEMPTS:
        return NODE_PRESENT
    return NODE_SEARCH


# --------------------------------------------------------------------------- #
# Apresentação e escolha
# --------------------------------------------------------------------------- #
def _format_recommendations(products: list[Product]) -> str:
    lines = ['Encontrei estas opções com melhor custo-benefício para você:\n']
    for i, product in enumerate(products[:TOP_PRODUCTS], start=1):
        price = (
            f'R${product.estimated_price}'
            if product.estimated_price is not None
            else 'preço não informado'
        )
        features = (
            f'\n   Destaques: {", ".join(product.key_features)}'
            if product.key_features
            else ''
        )
        lines.append(
            f'{i}. {product.name}'
            f'{f" ({product.brand})" if product.brand else ""} — {price}\n'
            f'   Por quê: {product.reason}{features}'
        )
    return '\n\n'.join(lines)


class ProductChoice(BaseModel):
    index: int = Field(
        description=(
            'Número (1 a N) do produto escolhido pelo usuário. Use 1 se ele pedir o mais '
            'recomendado ou não especificar claramente.'
        )
    )


async def _resolve_choice(answer: str, products: list[Product]) -> Product:
    options = '\n'.join(
        f'{i}. {p.name}' for i, p in enumerate(products[:TOP_PRODUCTS], start=1)
    )
    llm = get_llm().with_structured_output(ProductChoice, method='function_calling')
    choice = await llm.ainvoke(
        [
            SystemMessage(
                'Identifique qual produto o usuário escolheu a partir da resposta dele.'
            ),
            HumanMessage(f'Opções:\n{options}\n\nResposta do usuário: {answer}'),
        ]
    )
    assert isinstance(choice, ProductChoice)
    index = max(1, min(choice.index, len(products[:TOP_PRODUCTS]))) - 1
    return products[index]


async def present_recommendations(state: ProductSearchState):
    products = _products_from_state(state)
    presentation = _format_recommendations(products)

    answer = interrupt(
        {
            'type': 'choice',
            'message': presentation,
            'question': 'Qual modelo te interessou? (responda o número ou o nome)',
        }
    )
    chosen = await _resolve_choice(str(answer), products)

    return {
        'messages': [AIMessage(presentation), HumanMessage(str(answer))],
        'chosen_product': chosen.model_dump(mode='json'),
    }


# --------------------------------------------------------------------------- #
# Pesquisa profunda dos links de compra (após a escolha)
# --------------------------------------------------------------------------- #
def _format_links_message(product: Product, links) -> str:
    if not links:
        return (
            f'Não encontrei links de compra confiáveis para o {product.name} agora. '
            'Quer que eu busque outro modelo?'
        )
    lines = [f'Onde comprar o {product.name}:\n']
    for link in links:
        price = f' — R${link.price}' if link.price is not None else ''
        lines.append(f'- {link.store}{price}: {link.url}')
    return '\n'.join(lines)


async def search_purchase_links(state: ProductSearchState):
    chosen = state.get('chosen_product')
    assert chosen is not None
    product = Product.model_validate(chosen)
    links = await deep_search_purchase_links(product)
    return {'messages': [AIMessage(_format_links_message(product, links))]}


# --------------------------------------------------------------------------- #
# Construção do sub-grafo
# --------------------------------------------------------------------------- #
def build_product_search_subgraph():
    builder = StateGraph(ProductSearchState)

    builder.add_node(NODE_COLLECT, collect_requirements)
    builder.add_node(NODE_SEARCH, search_products)
    builder.add_node(NODE_VALIDATE, validate_products)
    builder.add_node(NODE_PRESENT, present_recommendations)
    builder.add_node(NODE_LINKS, search_purchase_links)

    builder.add_edge(START, NODE_COLLECT)
    builder.add_conditional_edges(
        NODE_COLLECT, route_after_collect, [NODE_COLLECT, NODE_SEARCH]
    )
    builder.add_edge(NODE_SEARCH, NODE_VALIDATE)
    builder.add_conditional_edges(
        NODE_VALIDATE, route_after_validate, [NODE_SEARCH, NODE_PRESENT]
    )
    builder.add_edge(NODE_PRESENT, NODE_LINKS)
    builder.add_edge(NODE_LINKS, END)

    # Compilado sem checkpointer: o checkpointer do grafo pai propaga para o sub-grafo.
    return builder.compile()


# Sub-grafo compilado usado como nó no grafo principal. Como é um grafo compilado
# (e não uma função), o LangGraph o trata como sub-grafo e o checkpointer do pai
# propaga, permitindo que os interrupt() funcionem entre os turnos.
product_search_agent = build_product_search_subgraph()
