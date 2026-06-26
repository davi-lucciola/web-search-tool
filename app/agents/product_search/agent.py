from langgraph.graph import END, START, StateGraph

from app.agents.product_search.nodes import (
    NODE_COLLECT,
    NODE_LINKS,
    NODE_PRESENT,
    NODE_SEARCH,
    NODE_VALIDATE,
    collect_requirements_node,
    present_recommendations_node,
    route_after_collect,
    route_after_validate,
    search_products_node,
    search_purchase_links_node,
    validate_products_node,
)
from app.agents.product_search.state import ProductSearchState


def build_product_search_agent():
    """Constrói e compila o sub-grafo de busca de produtos.

    Como retorna um grafo compilado (e não uma função), o LangGraph o trata como
    sub-grafo e o checkpointer do grafo pai propaga, permitindo que os interrupt()
    funcionem entre os turnos. Compilado sem checkpointer de propósito.
    """
    builder = StateGraph(ProductSearchState)

    builder.add_node(NODE_COLLECT, collect_requirements_node)
    builder.add_node(NODE_SEARCH, search_products_node)
    builder.add_node(NODE_VALIDATE, validate_products_node)
    builder.add_node(NODE_PRESENT, present_recommendations_node)
    builder.add_node(NODE_LINKS, search_purchase_links_node)

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

    return builder.compile()
