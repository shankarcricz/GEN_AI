
import json
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

def llm_chunk(text:str):
    prompt = f"""You are a document structuring specialist. Convert the raw text below into a structured JSON array of sections.

        DEFINITIONS:
        - "heading": a top-level section title (e.g. "Experience", "Skills", "Responsibilities", "Education"). Headings are short, stand alone on their own line, and introduce a new category of content.
        - "subheading": an entry within a section that groups related bullets together — typically a job title + company + dates (resume), or a sub-category label under a heading (JD), e.g. "Core Technical Proficiency:" under a broader "Qualifications" heading. Not every heading has subheadings — use null if there are none.
        - "bullets": a list of complete, standalone content items belonging to that heading/subheading. Each bullet must be one full, ungrounded thought — never split a sentence across two bullets, never merge two distinct bullets into one.

        RULES:
        1. Preserve all original wording. Do not summarize, paraphrase, or drop any content — every line of the input must appear somewhere in the output.
        2. Never let bullet content leak into a heading or subheading field, and never let a heading get silently absorbed into the previous section's bullets — every real section header must produce its own object in the output array, even if short.
        3. If the input has intro prose with no bullet markers (e.g. a paragraph under "About the job"), treat the whole paragraph as a single bullet unless it clearly contains multiple distinct ideas, in which case split by sentence.
        4. Output ONLY a valid JSON array. No markdown code fences, no explanation, no text before or after the array.

        OUTPUT SCHEMA:
        [{{"heading": "string", "subheading": "string or null", "bullets": ["string", ...]}}, ...]

        EXAMPLE INPUT:
        Responsibilities:
        - Deploy and configure AI models and infrastructure for enterprise clients
        - Troubleshoot and resolve technical issues in real-time

        EXAMPLE OUTPUT:
        [{{"heading": "Responsibilities", "subheading": null, "bullets": ["Deploy and configure AI models and infrastructure for enterprise clients", "Troubleshoot and resolve technical issues in real-time"]}}]

        Now process this text:

        {text}"""
    response = client.interactions.create(
        model="gemini-3.5-flash",
        input=prompt
    )
    raw = response.output_text.strip()
    # Strip markdown fences if the model wraps its output in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]  # drop opening fence line
        raw = raw.rsplit("```", 1)[0]  # drop closing fence
    return json.loads(raw)