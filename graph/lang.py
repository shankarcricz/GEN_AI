from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from services.embed import call_model, should_continue, tool_call

class State(TypedDict):
    previous_id: str
    function_results: list
    input: str
    output: str
    max_limit: int
    iter_count: int

graph = StateGraph(State)

graph.add_node("call_model", call_model)
graph.add_node("tool_call", tool_call)

graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", should_continue, {"tool_call": "tool_call", END: END})
graph.add_edge("tool_call", "call_model")

app = graph.compile()