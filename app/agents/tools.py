from typing import cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from pydantic import ValidationError

from app.agents.states import (
    Product,
    ProductRecommendations,
    PurchaseLink,
    PurchaseLinks,
    Requirements,
)
from app.llm import get_llm
from app.tavily import TavilyResult, get_tavily_client

# Modelos pequenos ocasionalmente emitem JSON malformado na saída estruturada;
# tentar novamente com uma nova amostragem resolve a maioria desses casos.
_STRUCTURED_RETRY_ERRORS = (ValidationError, OutputParserException, ValueError)

# Limite de candidatos extraídos por busca e de resultados agregados enviados ao LLM.
_MAX_AGGREGATED_RESULTS = 12


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


def _dedupe_results(results: list[TavilyResult]) -> list[TavilyResult]:
    """Remove resultados duplicados por URL, preservando a ordem de chegada."""
    seen: set[str] = set()
    unique: list[TavilyResult] = []
    for result in results:
        url = result.get('url', '')
        if url and url not in seen:
            seen.add(url)
            unique.append(result)
    return unique


def _build_search_queries(
    requirements: Requirements, budget: float | None, refine_hint: str | None = None
) -> list[str]:
    """Gera múltiplas queries direcionadas a partir dos requisitos do usuário.

    Diferentes ângulos (custo-benefício, reviews/comparativos, prioridades) aumentam a
    chance de cobrir os melhores modelos do que uma única query fixa.
    """
    product = requirements.product_type or 'produto'
    budget_clause = f' até R${budget:.0f}' if budget else ''
    priorities = ' '.join(requirements.priorities)
    use_case = requirements.use_case or ''

    queries = [
        f'melhores {product} custo-benefício{budget_clause} 2026',
        f'{product} {use_case} {priorities} review comparativo 2026'.strip(),
    ]
    if priorities:
        queries.append(f'melhor {product} para {priorities}{budget_clause}')
    if requirements.must_haves:
        queries.append(
            f'{product} com {" ".join(requirements.must_haves)}{budget_clause}'
        )
    if refine_hint:
        queries.append(refine_hint)

    # Remove duplicatas mantendo a ordem.
    return list(dict.fromkeys(q for q in queries if q.strip()))


FIND_PRODUCTS_PROMPT = """
Você é um especialista em produtos e custo-benefício. A partir dos resultados de busca
fornecidos, identifique os 3 modelos de produto mais adequados para a necessidade e o
orçamento do usuário.

Regras:
- Retorne exatamente 3 produtos, ordenados do melhor para o pior custo-benefício.
- Considere apenas modelos cujo preço aproximado caiba no orçamento informado.
- Baseie-se somente nas informações dos resultados; não invente modelos ou preços.
- Para cada produto, preencha o motivo da indicação considerando as prioridades do usuário.
- Escreva os textos em português do Brasil.
"""


async def search_candidates(
    requirements: Requirements, budget: float | None, refine_hint: str | None = None
) -> list[Product]:
    """Encontra os 3 modelos de melhor custo-benefício para os requisitos informados.

    Roda múltiplas buscas direcionadas no Tavily (custo-benefício, reviews/comparativos,
    prioridades), agrega e deduplica os resultados, e extrai os candidatos via LLM.
    `refine_hint` é usado no loop de re-busca quando os resultados anteriores foram fracos.
    """
    client = get_tavily_client()
    queries = _build_search_queries(requirements, budget, refine_hint)

    aggregated: list[TavilyResult] = []
    for query in queries:
        response = await client.search(query, search_depth='advanced', country='brazil')
        aggregated.extend(cast(list[TavilyResult], response.get('results', [])))

    results = _dedupe_results(aggregated)[:_MAX_AGGREGATED_RESULTS]
    context = _format_results(results)

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
                f'Tipo de produto: {requirements.product_type}\n'
                f'Uso pretendido: {requirements.use_case}\n'
                f'Prioridades: {", ".join(requirements.priorities) or "não informadas"}\n'
                f'Marcas preferidas: {", ".join(requirements.brand_preferences) or "indiferente"}\n'
                f'Requisitos obrigatórios: {", ".join(requirements.must_haves) or "nenhum"}\n'
                f'Orçamento máximo: {f"R${budget}" if budget else "não informado"}\n\n'
                f'Resultados de busca:\n{context}'
            ),
        ]
    )

    assert isinstance(recommendations, ProductRecommendations)
    return recommendations.products


FIND_PURCHASE_LINKS_PROMPT = SystemMessagePromptTemplate.from_template("""
Você é um especialista em encontrar onde comprar produtos online no Brasil. A partir dos
resultados de busca e do conteúdo extraído das páginas, extraia até {quantity} links de compra
para o produto informado.

Regras:
- Priorize as lojas Amazon, Mercado Livre e Shopee, nessa ordem de preferência.
- Retorne no máximo {quantity} links, cada um de uma loja diferente quando possível.
- Use somente URLs presentes nos resultados; não invente links.
- Preencha o preço apenas quando ele estiver claramente disponível nos resultados.
- Escreva os textos em português do Brasil.
""")


async def deep_search_purchase_links(
    product: Product, quantity: int = 2
) -> list[PurchaseLink]:
    """Pesquisa profunda de links de compra para um produto já escolhido pelo usuário.

    Primeiro faz uma busca avançada para localizar páginas de lojas e, em seguida, usa o
    `extract` do Tavily para ler o conteúdo dessas páginas e extrair link + preço com mais
    precisão do que apenas os snippets de busca.
    """
    client = get_tavily_client()
    response = await client.search(
        f'comprar {product.name} preço Amazon Mercado Livre Shopee',
        search_depth='advanced',
        country='brazil',
    )

    results = cast(list[TavilyResult], response.get('results', []))
    context = _format_results(results)

    # Lê as páginas das lojas mais relevantes para confirmar link e preço reais.
    top_urls = [r['url'] for r in results[:quantity] if r.get('url')]

    if top_urls:
        try:
            extracted = await client.extract(top_urls)
            for item in extracted.get('results', []):
                raw = item.get('raw_content') or ''
                if raw:
                    context += f'\n\n[Conteúdo extraído de {item.get("url", "")}]:\n{raw[:2000]}'
        except Exception:
            # Se o extract falhar, seguimos apenas com os snippets da busca.
            pass

    # Alternativa mais robusta (e mais cara em créditos): client.research(...).

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
