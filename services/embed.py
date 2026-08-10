
from dotenv import load_dotenv
from google import genai

import os
load_dotenv()




client = genai.Client(api_key=os.getenv("API_KEY"))
def embed_text(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    return result.embeddings[0].values

def generate_text(retrieved_chunks:str, query:str) -> str:
    prompt = f"Based on the following retrieved information: {retrieved_chunks}\n\nAnswer the question: {query}. answer if the context reasonably supports it, even if it takes a small inferential step (e.g., mentorship implies coaching), but refuse if the context has nothing relevant to draw on at all."
    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )
    return response.output_text

