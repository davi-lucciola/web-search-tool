from typing import cast

from langchain.tools import tool
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from pydantic import ValidationError

from app.agents.states import (
    Product,
    ProductRecommendations,
    PurchaseLink,
    PurchaseLinks,
)
from app.llm import get_llm
from app.tavily import TavilyResult, get_tavily_client

# Modelos pequenos ocasionalmente emitem JSON malformado na saída estruturada;
# tentar novamente com uma nova amostragem resolve a maioria desses casos.
_STRUCTURED_RETRY_ERRORS = (ValidationError, OutputParserException, ValueError)


def _format_results(results: list[TavilyResult]) -> str:
    """Monta um contexto textual a partir dos resultados do Tavily para extração via LLM."""
    blocks: list[str] = []
    for result in results:
        blocks.append(
            f'- Título: {result["title"]}\n'
            f'  URL: {result["url"]}\n'
            f'  Relevância: {result["score"]}\n'
            f'  Conteúdo: {result["content"]}'
        )
    return '\n'.join(blocks)


FIND_PRODUCTS_PROMPT = """
Você é um especialista em produtos e custo-benefício. A partir dos resultados de busca
fornecidos, identifique os 3 modelos de produto mais adequados para a necessidade e o
orçamento do usuário.

Regras:
- Retorne exatamente 3 produtos, ordenados do melhor para o pior custo-benefício.
- Considere apenas modelos cujo preço aproximado caiba no orçamento informado.
- Baseie-se somente nas informações dos resultados; não invente modelos ou preços.
- Para cada produto, preencha a fonte (source_url) com a URL do resultado correspondente.
- Escreva os textos em português do Brasil.
"""


@tool
async def find_products(need: str, budget: float) -> list[Product]:
    """Encontra os 3 modelos de produto mais adequados para uma necessidade e orçamento.

    Args:
        need: o tipo de produto e a necessidade do usuário (ex. "celular bom para fotos")
        budget: o orçamento máximo disponível, em reais (BRL)

    Returns:
        uma lista com os 3 produtos de melhor custo-benefício, como objetos Product
    """
    client = get_tavily_client()
    response = await client.search(
        f'melhores {need} custo-benefício até R${budget} 2026', country='brazil'
    )

    context = _format_results(cast(list[TavilyResult], response.get('results', [])))
    llm = (
        get_llm()
        .with_structured_output(ProductRecommendations, method='function_calling')
        .with_retry(
            retry_if_exception_type=_STRUCTURED_RETRY_ERRORS, stop_after_attempt=3
        )
    )
    recommendations = await llm.ainvoke(
        [
            SystemMessage(FIND_PRODUCTS_PROMPT),
            HumanMessage(
                f'Necessidade do usuário: {need}\n'
                f'Orçamento máximo: R${budget}\n\n'
                f'Resultados de busca:\n{context}'
            ),
        ]
    )

    assert isinstance(recommendations, ProductRecommendations)
    return recommendations.products


FIND_PURCHASE_LINKS_PROMPT = SystemMessagePromptTemplate.from_template("""
Você é um especialista em encontrar onde comprar produtos online no Brasil. A partir dos
resultados de busca fornecidos, extraia até 2 links de compra para o produto informado.

Regras:
- Priorize as lojas Amazon, Mercado Livre e Shopee, nessa ordem de preferência.
- Retorne no máximo {quantity} links, cada um de uma loja diferente quando possível.
- Use somente URLs presentes nos resultados; não invente links.
- Preencha o preço apenas quando ele estiver claramente disponível nos resultados.
- Escreva os textos em português do Brasil.
""")


@tool
async def find_purchase_links(
    product: Product, quantity: int = 2
) -> list[PurchaseLink]:
    """Pesquisa links de compra para um modelo de produto específico.

    Args:
        product: o produto (modelo Product) para o qual buscar onde comprar
        quantity: quantidade de links de compra que serão consultados

    Returns:
        até N links de compra, sendo N o parametro quantity priorizando Amazon, Mercado Livre e Shopee, como objetos PurchaseLink
    """
    client = get_tavily_client()
    response = await client.search(
        f'comprar {product.name} preço Amazon Mercado Livre Shopee', country='brazil'
    )

    context = _format_results(cast(list[TavilyResult], response.get('results', [])))
    system_message = FIND_PURCHASE_LINKS_PROMPT.format(quantity=quantity)
    human_message = HumanMessage(
        f'Produto: {product.name}'
        f'{f" ({product.brand})" if product.brand else ""}\n\n'
        f'Resultados de busca:\n{context}'
    )
    llm = (
        get_llm()
        .with_structured_output(PurchaseLinks, method='function_calling')
        .with_retry(
            retry_if_exception_type=_STRUCTURED_RETRY_ERRORS, stop_after_attempt=3
        )
    )
    purchase_links = await llm.ainvoke([system_message, human_message])

    assert isinstance(purchase_links, PurchaseLinks)
    return purchase_links.links
