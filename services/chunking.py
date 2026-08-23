from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import re




from services.chroma import add_to_chroma_db



def chunk_document(text:str, chunk_size:int = 400, chunk_overlap:int = 20) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separators=["\n\n", "\n", ". ", " ", ""])
    return text_splitter.split_text(text)


MONTH_RE = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
DATE_RANGE_RE = re.compile(
    rf"\b{MONTH_RE}\s+(?:19|20)\d{{2}}\s*[-–—]\s*(?:Present|Current|{MONTH_RE}\s+(?:19|20)\d{{2}})",
    re.IGNORECASE,
)


def classify_line(line: str, source_file: str = None) -> str:
    stripped = line.strip()
    bullet_keywords = ["-", "*", "•"]
    heading_keywords_resume = ["Objective", "Education", "Experience", "Skills", "Projects",
                         "Certifications", "Awards", "Publications", "Interests",
                         "References", "Positions of Responsibility", "Summary", "Profile",
                         "Contact", "Hobbies", "Achievements", "Professional Summary",
                         "Technical Skills", "Work Experience", "Academic Background"]
    heading_keywords_job_description = ["Job Title", "Responsibilities", "Requirements", "Qualifications","Location","Level"
                         "Skills", "Experience", "Education", "Benefits", "Company Overview","About","Opportunity","Key overview","role overview"]
    
    resume_file_types = ["resume","curriculum","cv"]
    jd_file_types = ["jd","job description","job"]
    heading_keywords = heading_keywords_resume
    
    
    if stripped.startswith(tuple(bullet_keywords)):
        return "bullet"
    

    for keyword in heading_keywords:
        if keyword.lower() in stripped.lower():
            return "heading"
    if DATE_RANGE_RE.search(stripped):
        return "subheading"
    return "continuation"


def semantic_chunk_document(text: str, metadata: dict) -> list[dict]:
    chunk_arr = []
    obj = {"heading": None, "bullets": [], "subheading": None, "source_file": metadata.get("source", None),}

    def flush():
        # A heading alone (no subheading, no bullets) carries no embeddable
        # content — it's just the transient state right after seeing a
        # section header, before its content has arrived.
        if obj["bullets"] or obj["subheading"]:
            chunk_arr.append(obj.copy())

    for line in text.splitlines():
        classification = classify_line(line, source_file=metadata.get("source", None))
        stripped = line.strip()
        if not stripped:
            continue

        if classification == "heading":
            flush()
            obj = {"heading": stripped, "bullets": [], "subheading": None, "source_file": metadata.get("source", None)}
        elif classification == "subheading":
            if obj["subheading"] and not obj["bullets"]:
                # Same role/entry described across consecutive lines (e.g. a
                # duplicated title line) — merge instead of flushing an empty chunk.
                obj["subheading"] = f"{obj['subheading']} {stripped}"
            else:
                flush()
                obj = {"heading": obj["heading"], "bullets": [], "subheading": stripped, "source_file": metadata.get("source", None)}
        elif classification == "bullet": obj["bullets"].append(stripped)
        else:  # continuation
            if obj["bullets"]:
                obj["bullets"][-1] += " " + stripped
            else :
                obj["bullets"].append(stripped)

    flush()
    return chunk_arr

    







