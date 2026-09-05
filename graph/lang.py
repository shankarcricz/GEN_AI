from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from services.embed import approve_node, call_model, should_continue, tool_call
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    previous_id: str
    function_results: list
    input: str
    output: str
    approved: bool
    max_limit: int
    iter_count: int


def route_after_tool(state: State):
    if not state.get("approved", False):
        return END
    return "call_model"


graph = StateGraph(State)

graph.add_node("call_model", call_model)
graph.add_node("tool_call", tool_call)
graph.add_node("approve", approve_node)

graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", should_continue, {"approve": "approve", END: END})
graph.add_edge("approve", "tool_call")
graph.add_conditional_edges("tool_call", route_after_tool, {"call_model": "call_model", END: END})
checkpoint = MemorySaver()

app = graph.compile(checkpointer=checkpoint)