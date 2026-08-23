from services.loadtest import run_load_test
from services.evals import test_case_data
from services.ollama import classification_of_question
from services.load import retrieve
from services.ollama import llm_judge
from services.evals import eval_metrics
from services.chroma import get_from_chroma_db
from services.load import llm_response
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from services.load import load_pdf_and_add_to_chroma
from fastapi.responses import StreamingResponse
import json

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
    return {"status":"OK"}

@app.post("/ingest")
async def upload_resume(input_pdf: UploadFile = File(...), fileType : str = 'resume'):
    content = await input_pdf.read()
    await load_pdf_and_add_to_chroma(content, fileType)
    return {"status":200, "message":"success"}


async def fetch_answers(query :str):
    raw_classification = await classification_of_question(query)
    raw_classification = raw_classification.strip().lower()

    filterBy = ''
    # Parse LLM output — it may return verbose text; extract the known keyword
    if "both" in raw_classification:
        filterBy = "both"
    elif "jd" in raw_classification or "job description" in raw_classification:
        filterBy = "jd"
    else:
        filterBy = "resume"  # default fallback

    print(f"[DEBUG] query='{query}' | raw_classification='{raw_classification}' | filterBy='{filterBy}'")
    results = await retrieve(query, n_results=5, max_distance=0.8, filterBy=filterBy)  
    print(results)
    # return

    if len(results) == 0:
        yield f"data: {json.dumps({'type':'no_context', 'results': 'No context found' })}\n\n"
        return
    
    yield f"data: {json.dumps({'type': 'citations', 'results': results})}\n\n"
    response = await llm_response(query, results, raw_classification)

    judge_response = await llm_judge(query, "\n".join([c["chunk"] for c in response["citations"]]), response["answer"])

    if judge_response["groundedness"] < 3:
        response["answer"] = "Groundedness is less than 3"
        
    
    yield f"data: {json.dumps({'type':'answer', 'results': response})}\n\n"



@app.get("/retrieve")
async def ask_question(query :str):
    return StreamingResponse(fetch_answers(query), media_type="text/event-stream")

@app.get("/fetch")
async def fetch():
    res = await get_from_chroma_db()
    return {"status":200, "response": res}




@app.get('/help')
async def helper():
    return await run_load_test()

    # obj = {}
    # test_cases = test_case_data["test_cases"]
    # for i,test_case in enumerate(test_cases):
    #     filterBy = test_case["category"]
    #     results = retrieve(test_case["question"], n_results=5, max_distance=10, filterBy=filterBy)
    #     obj[test_case['question']] = [r['distance'] for r in results]
    # return obj




     






@app.get("/evals")
async def evals():
    res = await eval_metrics()
    print(res)
    return {"status":200,"response":res}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)





# GET http://localhost:8000/health
# POST http://localhost:8000/ingest




