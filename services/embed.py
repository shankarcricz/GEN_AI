
from services.rate_limit import rate_limited_async
import json
from dotenv import load_dotenv
from google import genai
import os
load_dotenv()




client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# @rate_limited_async(max_retries=8, base_delay=2.0, request_delay=1.0)
def embed_text(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    return result.embeddings[0].values


@rate_limited_async(max_retries=8, base_delay=2.0, request_delay=1.0)
async def generate_text(retrieved_chunks:str, query:str) -> str:
    prompt = f"""
    <instructions>
    You are an AI interview coach. Your job is to help candidates prepare for interviews by answering their questions based on the context provided.If needed do inferential reasoning to answer the question but make sure the repsone still stays relevant to the context provided and query provided. Dont exaggerate outside of the resume context at all
    </instructions>
    <example>
    <context>
    - worked at comcast as software engineer
    - worked on improving the performance of the application
    - worked on adding new features to the application
    
    </context>
    <question>
    what were the roles and responsibilities at comcast and lina?
    </question>
    <answer>
    At Comcast, I worked as a software engineer where I was responsible for developing and maintaining the Comcast Linx platform. I was also involved in the development of new features and the improvement of existing ones.
    </answer>
    </example>
   

    Here is the input below:

    <context>
    {retrieved_chunks}
    </context>
    
    <question>
    {query}
    </question>
    <answer>
    """
    response = client.interactions.create(
        model="gemini-3.5-flash",
        input=prompt
    )
    return response.output_text


@rate_limited_async(max_retries=8, base_delay=2.0, request_delay=1.0)
async def classification_of_question(query:str):
    prompt = f"""You are a document routing classifier. Given an interview question, decide which document(s) are needed to answer it.

    RULES:
    - Output ONLY one word: resume, jd, or both. No explanation, no punctuation, just the single word.
    - Use "resume" when the question is about the CANDIDATE — their skills, experience, background, projects, achievements, comfort with technologies, or anything about who they are.
    - Use "jd" ONLY when the question is specifically about the JOB itself — salary, company culture, job requirements, location, benefits.
    - Use "both" when BOTH the candidate's background AND the job details are needed to answer.

    EXAMPLES:
    Q: What are the roles and responsibilities at Comcast?  → resume
    Q: Are you comfortable with React?                      → resume
    Q: Do you have experience with Machine Learning?        → resume
    Q: What are your key skills?                           → resume
    Q: What is the salary for this role?                   → jd
    Q: What does the job require in terms of experience?   → jd
    Q: Are your skills a good match for this position?     → both

    Question: {query}
    Answer:"""
    response = client.interactions.create(
        model="gemini-3.5-flash",
        input=prompt
    )
    return response.output_text
    

@rate_limited_async(max_retries=8, base_delay=2.0, request_delay=1.0)
async def llm_chunk(text:str):




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

@rate_limited_async(max_retries=8, base_delay=2.0, request_delay=1.0)
async def llm_judge(question: str, retrieved_chunks: str, answer: str) -> dict:
    prompt = f"""
    <instructions>
    You are evaluating the quality of an AI interview coach's answer. The coach helps a candidate
    assess their resume against a job description, using only the retrieved context provided.
    Score the answer on three dimensions, each from 1-5.
    </instructions>

    <scoring_criteria>
    RELEVANCY (1-5): Does the answer directly address what was asked, without drifting into unrelated information?
    GROUNDEDNESS (1-5): Are all claims in the answer actually supported by the retrieved context?
      A 1 means the answer invents or overstates something not present in the context.
      A 5 means every claim traces directly back to the context.
    COMPLETENESS (1-5): Does the answer engage with the specifics (numbers, facts, reasoning)
      rather than being generic or vague?
    </scoring_criteria>

    <context>
    {retrieved_chunks}
    </context>

    <question>
    {question}
    </question>

    <answer_to_evaluate>
    {answer}
    </answer_to_evaluate>

    Respond ONLY with valid JSON in this exact format, no markdown fences, no explanation outside the JSON:
    {{"relevancy": <int 1-5>, "groundedness": <int 1-5>, "completeness": <int 1-5>, "groundedness_reasoning": "<one sentence justifying the groundedness score specifically>"}}
    """
    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )
    raw = response.output_text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)