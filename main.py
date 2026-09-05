import json
import os
import sys
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from services.chroma import get_from_chroma_db
from services.evals import eval_metrics
from services.embed import generate_content, llm_judge
from services.load import llm_response, load_pdf_and_add_to_chroma
from services.ollama import classification_of_question, could_web_search_help
from tools.retrieve import retrieve_chunks


app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://genai-shankar-rsbuild.s3-website.eu-north-1.amazonaws.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.post("/ingest")
async def upload_resume(input_pdf: UploadFile = File(...), fileType: str = "resume"):
    content = await input_pdf.read()
    await load_pdf_and_add_to_chroma(content, fileType)
    return {"status": 200, "message": "success"}


def graph_input(prompt: str) -> dict:
    return {
        "input": prompt,
        "previous_id": None,
        "function_results": [],
        "output": "",
        "approved": False,
        "max_limit": 5,
        "iter_count": 0,
    }


async def start_web_search(prompt: str):
    from graph.lang import app as search_app

    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = await search_app.ainvoke(graph_input(prompt), config=config)
    return search_app, thread_id, config, result


def sse_event(event_type: str, results, **extra) -> str:
    payload = {"type": event_type, "results": results, **extra}
    return f"data: {json.dumps(payload)}\n\n"


def interrupt_payload(result) -> object:
    interrupts = result.get("__interrupt__", ())
    if not interrupts:
        return None
    interrupt = interrupts[0]
    return getattr(interrupt, "value", interrupt)


async def fetch_answers(query: str):
    results = await retrieve_chunks(query)
    should_search = False

    if not results:
        should_search = await could_web_search_help(query)

    if not results and not should_search:
        yield sse_event("no_context", "No context found")
        return

    yield sse_event("citations", results)

    if should_search:
        prompt = f"""The candidate asked: {query}
        We were not able to find a grounded answer from these chunks and it was not complete.
        Answer the query using the tool call and make sure to give an appropriate answer.
        """
        yield sse_event("webSearch", "Searching across the internet....")
        _, thread_id, _, result = await start_web_search(prompt)
        if "__interrupt__" in result:
            yield sse_event("approval_required", interrupt_payload(result), thread_id=thread_id)
            return
        yield sse_event("answer", {"answer": result.get("output", ""), "citations": []})
        return

    response = await llm_response(
        query,
        results,
        raw_classification=await classification_of_question(query),
    )
    judge_response = await llm_judge(
        query,
        "\n".join(c["chunk"] for c in response["citations"]),
        response["answer"],
    )

    if judge_response["groundedness"] < 3 or judge_response["could_web_search_help"]:
        retrieved_chunks = "\n".join(c["chunk"] for c in response["citations"])
        prompt = f"""The candidate asked: {query}
        Here are the retrieved chunks: {retrieved_chunks}.
        We were not able to find a grounded answer from these chunks and it was not complete.
        Answer the query using the tool call and make sure to give an appropriate answer.
        """
        yield sse_event("webSearch", "Searching across the internet....")
        _, thread_id, _, result = await start_web_search(prompt)
        if "__interrupt__" in result:
            yield sse_event("approval_required", interrupt_payload(result), thread_id=thread_id)
            return
        response["answer"] = result.get("output", "")
        response["citations"] = []

    yield sse_event("answer", response)


@app.get("/retrieve")
async def ask_question(query: str):
    return StreamingResponse(fetch_answers(query), media_type="text/event-stream")


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


@app.post("/retrieve/approval")
async def approve_web_search(request: ApprovalRequest):
    from graph.lang import app as search_app

    config = {"configurable": {"thread_id": request.thread_id}}
    try:
        state = await search_app.aget_state(config)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Unknown search approval request") from exc

    if not state.values or not state.next:
        raise HTTPException(status_code=409, detail="Search approval request is no longer pending")

    result = await search_app.ainvoke(Command(resume=request.approved), config=config)
    if "__interrupt__" in result:
        raise HTTPException(status_code=409, detail="Search approval request is still interrupted")

    return {"type": "answer", "results": {"answer": result.get("output", ""), "citations": []}}


@app.get("/fetch")
async def fetch():
    return {"status": 200, "response": await get_from_chroma_db()}


@app.get("/help")
async def helper():
    await generate_content()


@app.get("/evals")
async def evals():
    return {"status": 200, "response": await eval_metrics()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)