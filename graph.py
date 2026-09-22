"""LangGraph workflow definition."""
from langgraph.graph import END, START, StateGraph

from nodes import (
    after_sales,
    classify,
    finish,
    human_node,
    finalize_history,
    prepare_history,
    presales,
    response_route,
    route,
    retrieve_knowledge,
)
from schemas import SupportState
from checkpoint import build_checkpointer


def build_graph():
    """Build independent presales/after-sales nodes with bounded retries."""
    graph = StateGraph(SupportState)
    graph.add_node("classify", classify)
    graph.add_node("prepare_history", prepare_history)
    graph.add_node("retrieve", retrieve_knowledge)
    graph.add_node("presales", presales)
    graph.add_node("after_sales", after_sales)
    graph.add_node("other", finish)
    graph.add_node("human", human_node)
    graph.add_node("finalize_history", finalize_history)
    graph.add_edge(START, "prepare_history")
    graph.add_edge("prepare_history", "retrieve")
    graph.add_edge("retrieve", "classify")
    # Classification errors retry classification; three failures circuit-break to human.
    graph.add_conditional_edges(
        "classify", route,
        {
            "retry": "classify",
            "human": "human",
            "presales": "presales",
            "after_sales": "after_sales",
            "other": "other",
        },
    )
    # Response-generation errors retry the same business node, not the whole request.
    for node in ("presales", "after_sales"):
        graph.add_conditional_edges(
            node, response_route,
            {"retry": node, "human": "human", "done": "finalize_history"},
        )
    graph.add_edge("other", "finalize_history")
    graph.add_edge("human", "finalize_history")
    graph.add_edge("finalize_history", END)
    # SQLite keeps thread_id sessions across process restarts.
    return graph.compile(checkpointer=build_checkpointer())


app = build_graph()
