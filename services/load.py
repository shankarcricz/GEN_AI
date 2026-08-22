
# from services.embed import classification_of_question
from services.ollama import classification_of_question
from services.rate_limit import rate_limited_async
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pypdf

from io import BytesIO
import re
from fastapi import UploadFile, File
# from services.embed import generate_text, llm_chunk
from services.ollama import generate_text, llm_chunk


from services.chunking import semantic_chunk_document
from services.chroma import add_to_chroma_db, get_from_chroma_db,get_count_from_chroma_db, fetch_query_results, filter_results_by_distance

from langfuse.decorators import observe


@observe()
async def load_pdf_and_add_to_chroma(pdf_bytes: bytes, fileType: str ):
    reader = pypdf.PdfReader(BytesIO(pdf_bytes))

   
    source = ''

    page_texts = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            page_texts.append(extracted)

    if not page_texts:
        raise ValueError("Could not extract readable text from the PDF.")

    # 3. Merge and chunk
    text = "\n\n".join(page_texts)

    chunks = await llm_chunk(text)

    print(chunks)

    for i, chunk in enumerate(chunks):
        await add_to_chroma_db(chunk, fileType)

    


    
    # for i, chunk in enumerate(text):
    #     print(chunk)
    #     print("-" * 50)
        # add_to_chroma_db(chunk, source)

# def load_txt_and_add_to_chroma():
#     loader_txt = TextLoader("data/jd_applications_dev_senior_analyst.txt")
#     pages = loader_txt.load()
    
    
    
#     text = "\n\n".join([page.page_content for page in pages])
#     chunks = llm_chunk(text)
#     source_file = pages[0].metadata.get('source', 'Unknown')

    
#     for i, chunk in enumerate(chunks):
#         # print(chunk)
#         # print("-" * 50)
#         add_to_chroma_db(chunk, source_file)


def query():
    query = "Comcast Linx"
    results = fetch_query_results(query, n_results=2)
    return results

def retrieve(query:str, n_results:int = 5, max_distance:float = 0.5, filterBy:str="resume"):
    results = fetch_query_results(query, n_results=n_results, max_distance=max_distance, filterBy=filterBy)
    return results


def questions() -> list[str]:
    interviewQuestions = [
    # "What were the roles & responsibilities at Comcast?",
    "Do you have any coaching experience?",
    # "Where have you used React components throughout your experience?",
    # "Do you know kubernetes and kafka?",
    # "Do you have any common points in both of your work experiences?",
    # "Have you worked on Gen AI at Amazon?",
    # "Are you good at development?",
    "What is your favorite movie?",
    # "Are you good at communicating ideas with colleagues?",
    # "Can you own modules end to end?"
    ]
    return interviewQuestions


@observe()
async def llm_response(query:str) -> object:
    print(f"[llm_response] Processing: '{query[:60]}...'")
    raw_classification = await classification_of_question(query)
    raw_classification = raw_classification.strip().lower()

    print(raw_classification)

    # Parse LLM output — it may return verbose text; extract the known keyword
    if "both" in raw_classification:
        filterBy = "both"
    elif "jd" in raw_classification or "job description" in raw_classification:
        filterBy = "jd"
    else:
        filterBy = "resume"  # default fallback

    print(f"[DEBUG] query='{query}' | raw_classification='{raw_classification}' | filterBy='{filterBy}'")

    results = retrieve(query, n_results=5, max_distance=1, filterBy=filterBy)
    print(f"[DEBUG] results count: {len(results)}")

    generated_text = {"answer":None, "citations":[],"query_classifiction":raw_classification}
    generated_text["answer"] = await generate_text("\n".join([f'[SOURCE: {r["metadata"]["source_file"]}]' + " <-> " + r["document"] for r in results]), query)
    for r in results[:5]:
        generated_text["citations"].append({
            "source_file": r["metadata"]["source_file"],
            "chunk":f'[SOURCE: {r["metadata"]["source_file"]}]' + " <-> " +  r["document"],
            "id" : r['id']
        })

    return generated_text






if __name__ == "__main__":
    # load_pdf_and_add_to_chroma()
    # load_txt_and_add_to_chroma()

    questions = questions()
    results = {}
    for question in questions:
        generated_text = llm_response(question)
        results[question] = generated_text
    print(results)

    count = get_count_from_chroma_db()
    print(f"Total entries in Chroma DB: {count}")
    # results = query()
    
        # print(f"Question: {question}")
        # print("\n")
        # print(f"Generated Answer: {generated_text}")
        # print("-" * 50)
        
        # print(f"Question : {question}")
        # print("\n")
        # print(f"Result : {results}")
        # print("-" * 50)

    
   