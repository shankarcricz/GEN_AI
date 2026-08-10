import chromadb

try:
    from services.embed import embed_text
except ModuleNotFoundError:
    from embed import embed_text

client = chromadb.PersistentClient(path="./chroma_db6_shankar") 
collection = client.get_or_create_collection(name="job_fit_copilot")

def add_to_chroma_db(resume_text: dict):
    display_document = " ".join(resume_text["bullets"]).strip()
    text_to_embed = " " + (resume_text["subheading"] or "") + " " + (resume_text["heading"] or "") + " ".join(resume_text["bullets"]).strip()
    text_to_embed = text_to_embed.strip()
    if not text_to_embed:
        text_to_embed = resume_text["subheading"] or resume_text["heading"] or ""
    if not text_to_embed:
        return
    text_embedding = embed_text(text_to_embed)
    collection.upsert(
        ids=[f"{resume_text['heading']}:{resume_text['subheading']}:{hash(text_to_embed)}"],
        documents=[display_document],
        embeddings=[text_embedding],
        metadatas=[{
            "heading": resume_text["heading"] or "",
            "subheading": resume_text["subheading"] or "",
        }],
    )

def fetch_query_results(query: str, n_results: int = 5, max_distance: float = 1.0) -> list[dict]:
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        # include=["metadatas", "documents", "distances", "embeddings"],
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


def get_from_chroma_db():
    results = collection.get()
    return results
def get_count_from_chroma_db():
    count = collection.count()
    return count