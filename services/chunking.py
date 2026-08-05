from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import re
def chunk_document(text:str, chunk_size:int = 400, chunk_overlap:int = 20) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separators=["\n\n", "\n", ". ", " ", ""])
    return text_splitter.split_text(text)


def classify_line(line: str) -> str:
    stripped = line.strip()
    bullet_keywords = ["-", "*", "•"]
    heading_keywords = ["Objective", "Education", "Experience", "Skills", "Projects",
                         "Certifications", "Awards", "Publications", "Interests",
                         "References", "Positions of Responsibility", "Summary", "Profile",
                         "Contact", "Hobbies", "Achievements", "Professional Summary",
                         "Technical Skills", "Work Experience", "Academic Background"]

    if stripped.startswith(tuple(bullet_keywords)):
        return "bullet"
    for keyword in heading_keywords:
        if stripped.lower().startswith(keyword.lower()):
            return "heading"
    if re.search(r"\b(19|20)\d{2}\b", stripped):
        return "subheading"
    return "continuation"


def semantic_chunk_document(text: str) -> list[dict]:
    chunk_arr = []
    obj = {"heading": None, "bullets": [], "subheading": None}

    def flush():
        if obj["heading"] or obj["bullets"] or obj["subheading"]:
            chunk_arr.append(obj.copy())

    for line in text.splitlines():
        classification = classify_line(line)
        print(f"Line: {line} | Classification: {classification}")
        stripped = line.strip()
        if not stripped:
            continue

        if classification == "heading":
            flush()
            obj = {"heading": stripped, "bullets": [], "subheading": None}
        elif classification == "subheading":
            flush()
            obj = {"heading": obj["heading"], "bullets": [], "subheading": stripped}
        elif classification == "bullet":
            obj["bullets"].append(stripped)
        else:  # continuation
            if obj["bullets"]:
                obj["bullets"][-1] += " " + stripped
            else :
                obj["bullets"].append(stripped)

    flush()
    return chunk_arr

    
    





if __name__ == "__main__":
    loader = PyPDFLoader("data/Ashish_Resume_final.pdf")
    pages = loader.load()
    text = "\n\n".join([page.page_content for page in pages])
    chunks = semantic_chunk_document(text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")