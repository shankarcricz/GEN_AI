from services.load import llm_response
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from services.load import load_pdf_and_add_to_chroma

app = FastAPI()

origins = [
    "http://localhost:3000"
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

@app.get("/retrieve")
async def fetch_answers(query :str):
    res = await llm_response(query)
    return {"status":200, "response": res}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)





# GET http://localhost:8000/health
# POST http://localhost:8000/ingest




