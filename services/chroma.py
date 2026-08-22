import chromadb
import hashlib
try:
    from services.embed import embed_text
except ModuleNotFoundError:
    from embed import embed_text
import os
from dotenv import load_dotenv
load_dotenv()


import chromadb

client = chromadb.CloudClient(
  api_key=os.getenv("CHROMADB_API_KEY"),
  tenant='59bb4bd3-b4e7-4be9-ad92-8e90e5226597',
  database='genai'
)


# client = chromadb.PersistentClient(path=db_chroma_path) 
collection = client.get_or_create_collection(name="job_fit_copilot")

async def add_to_chroma_db(resume_text: dict, source_file: str):
    display_document = " " + (resume_text["subheading"] or "") + " " + (resume_text["heading"] or "") + " ".join(resume_text["bullets"]).strip()
    
    text_to_embed = " " + (resume_text["subheading"] or "") + " " + (resume_text["heading"] or "") + " ".join(resume_text["bullets"]).strip()
    text_to_embed = text_to_embed.strip()
    if not text_to_embed:
        text_to_embed = resume_text["subheading"] or resume_text["heading"] or ""
    if not text_to_embed:
        return
    text_embedding = embed_text(text_to_embed)

    bullets_content = " ".join(resume_text["bullets"]).strip()
    hash_source = bullets_content if bullets_content else (resume_text["heading"] or "") + (resume_text["subheading"] or "")
    chunk_hash = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()[:16]
    chunk_id = f"{source_file}:{chunk_hash}"


    collection.upsert(
        ids=[chunk_id],
        documents=[display_document],
        embeddings=[text_embedding],
        metadatas=[{
            "heading": resume_text["heading"] or "",
            "subheading": resume_text["subheading"] or "",
            "source_file": source_file,
        }],
    )

def fetch_query_results(query: str, n_results: int = 5, max_distance: float = 1.0, filterBy : str = "resume") -> list[dict]:
    query_embedding = embed_text(query)
    filterBy = filterBy.strip().lower()
    if filterBy == "both":
        where_clause = {"source_file": {"$in": ["resume", "jd"]}}
    else:
        where_clause = {"source_file": filterBy}
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        # include=["metadatas", "documents", "distances", "embeddings"],
        where=where_clause
    )

    return filter_results_by_distance(results, max_distance=max_distance)


def filter_results_by_distance(results: dict, max_distance: float = 1.0) -> list[dict]:
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    embeddings = results["embeddings"][0] if results.get("embeddings") is not None else [None] * len(ids)

    return [
        {
            "id": id_,
            "document": document,
            "embedding": embedding,
            "metadata": metadata,
            "distance": distance,
        }
        for id_, document, embedding, metadata, distance in zip(ids, documents, embeddings, metadatas, distances)
        if distance < max_distance
    ]


async def get_from_chroma_db():
    #i need to fetch theID column
    results = collection.get(  
    )
    return results
def get_count_from_chroma_db():
    count = collection.count()
    return count