from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class State(TypedDict):
    query : str
    approved : bool
    results : str

def approve_node(state: State):
    decision = interrupt({
        "question": "Search the web for this?",
        "query": state['query']
    })
    return {"approved" : decision}

def tool_call(state:State):
    if state['approved']:
        return {"results": "fake tool call happened!!"}
    return {"results": "tool call not approved"}

app = StateGraph(State, initial_state={"query":"", "approved":False, "results":""})

app.add_node("approve", approve_node)
app.add_node("tool_call", tool_call)
app.add_edge(START, "approve")
app.add_edge("approve", "tool_call")
app.add_edge("tool_call", END)

checkpoint = MemorySaver()

workflow = app.compile(checkpointer=checkpoint)

async def main():
    config = {"configurable": {"thread_id": "test-1"}}
    res = await workflow.ainvoke({"query":"What is the weather like today?"}, config=config)
    print(res)
    if "__interrupt__" in res:
        print("Workflow was interrupted. Resuming...")
        user_input = input("Please provide your input: ")
        user_input = user_input.strip().lower() in ["yes", "y"]
        result  = await workflow.ainvoke(Command(resume=user_input), config=config)
        print(result)

if __name__ == "__main__":

    import asyncio
    asyncio.run(main())
   


