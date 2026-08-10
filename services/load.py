
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import re
from embed import generate_text



from chunking import semantic_chunk_document
from chroma import add_to_chroma_db, get_from_chroma_db,get_count_from_chroma_db, fetch_query_results, filter_results_by_distance



def load_pdf_and_add_to_chroma():
    loader_Resume = PyPDFLoader("data/shankar_2026_resume.pdf")
    pages = loader_Resume.load()
    
    
    
    text = "\n\n".join([page.page_content for page in pages])
    chunks = semantic_chunk_document(text, pages[0].metadata)


    
    for i, chunk in enumerate(chunks):
        # print(chunk)
        # print("-" * 50)
        add_to_chroma_db(chunk)




def query():
    query = "Comcast Linx"
    results = fetch_query_results(query, n_results=2)
    return results

def retrieve(query:str, n_results:int = 5, max_distance:float = 0.5):
    results = fetch_query_results(query, n_results=n_results, max_distance=max_distance)
    return results


def questions() -> list[str]:
    interviewQuestions = [
    "What were the roles & responsibilities at Comcast?",
    "Do you have any coaching experience?",
    "Where have you used React components throughout your experience?",
    "Do you know kubernetes and kafka?",
    "Do you have any common points in both of your work experiences?",
    "Have you worked on Gen AI at Amazon?",
    "Are you good at development?",
    "What is your favorite movie?",
    "Are you good at communicating ideas with colleagues?",
    "Can you own modules end to end?"
    ]
    return interviewQuestions





if __name__ == "__main__":
    load_pdf_and_add_to_chroma()

    count = get_count_from_chroma_db()
    print(f"Total entries in Chroma DB: {count}")
    # results = query()
    questions = questions()
    for question in questions:
        results = retrieve(question, n_results=5, max_distance=1)

        # generated_text = generate_text("\n".join([r["document"] for r in results]), question)
        # print(f"Question: {question}")
        # print("\n")
        # print(f"Generated Answer: {generated_text}")
        # print("-" * 50)

        print(f"Result : {results}")
        print("-" * 50)
        break

    print(type(results))
   