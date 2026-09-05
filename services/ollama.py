from services.rate_limit import with_retry
import json
import os
import re
from dotenv import load_dotenv
from google import genai
import ollama
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

load_dotenv()



langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com"
)

print("Langfuse keys loaded:", bool(os.getenv("LANGFUSE_PUBLIC_KEY")), bool(os.getenv("LANGFUSE_SECRET_KEY")))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Local Ollama client & model identifier
ollama_client = ollama.AsyncClient()
MODEL_NAME = "mistral:latest"


def embed_text(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=text
    )
    print("gemini embedding length:", len(result.embeddings[0].values))
    return result.embeddings[0].values


@observe()
@with_retry(max_retries=3, base_delay=5, call_timeout=30)
async def generate_text(retrieved_chunks: str, query: str) -> str:
    system_prompt = (
        "You are an AI interview coach. Your job is to help candidates prepare for interviews by answering "
        "their questions based on the context provided. If needed do inferential reasoning to answer the "
        "question but make sure the response still stays relevant to the context provided and query provided. "
        "Dont exaggerate outside of the resume context at all.\n\n"
        "CRITICAL RULE: Context will have labels like these: [SOURCE: RESUME] or [SOURCE: JD].\n"
        "- Only claim something as the candidate's OWN experience if it appears under [SOURCE: RESUME].\n"
        "- Never attribute JD language (skills, tools, responsibilities) to the candidate unless the SAME thing also appears in [SOURCE: RESUME].\n"
        "- If the JD wants something the resume doesn't show, say so explicitly as a gap — do not blend the two.\n\n"
        "<example>\n"
        "<context>\n"
        "[SOURCE: RESUME]\n"
        "- worked at comcast as software engineer\n"
        "- worked on improving the performance of the application\n"
        "- worked on adding new features to the application\n"
        "[SOURCE: JD]\n"
        "- Experience: 4-7 years\n"
        "- Skills: React, Node.js, MongoDB\n"
        "</context>\n"
        "<question>\n"
        "what were the roles and responsibilities at comcast and lina?\n"
        "</question>\n"
        "<answer>\n"
        "At Comcast, I worked as a software engineer where I was responsible for developing and maintaining "
        "the Comcast Linx platform. I was also involved in the development of new features and the improvement "
        "of existing ones.\n"
        "</answer>\n"
        "</example>"
    )

    user_prompt = (
        f"<context>\n"
        f"{retrieved_chunks}\n"
        f"</context>\n\n"
        f"<question>\n"
        f"{query}\n"
        f"</question>\n"
        f"<answer>"
    )


    response = await ollama_client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.2}
    )
    langfuse_context.update_current_observation(
        usage={
            "input": response["prompt_eval_count"],
            "output": response["eval_count"],
        }
    )
    return response["message"]["content"].strip()

@observe()
@with_retry(max_retries=3, base_delay=5, call_timeout=30)
async def classification_of_question(query: str) -> str:
    system_prompt = (
        "You are a document routing classifier. Given an interview question, decide which document(s) are needed to answer it.\n\n"
        "RULES:\n"
        "- Output ONLY one word: resume, jd, or both. No explanation, no punctuation, just the single word.\n"
        "- Use \"resume\" when the question is about the CANDIDATE — their skills, experience, background, projects, achievements, comfort with technologies, or anything about who they are.\n"
        "- Use \"jd\" ONLY when the question is specifically about the JOB itself — salary, company culture, job requirements, location, benefits.\n"
        "- Use \"both\" when BOTH the candidate's background AND the job details are needed to answer.\n\n"
        "EXAMPLES:\n"
        "Q: What are the roles and responsibilities at Comcast?  → resume\n"
        "Q: Are you comfortable with React?                      → resume\n"
        "Q: Do you have experience with Machine Learning?        → resume\n"
        "Q: What are your key skills?                           → resume\n"
        "Q: What is the salary for this role?                   → jd\n"
        "Q: What does the job require in terms of experience?   → jd\n"
        "Q: Are your skills a good match for this position?     → both"
    )

    response = await ollama_client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\nAnswer:"}
        ],
        options={"temperature": 0.0}
    )
    langfuse_context.update_current_observation(
        usage={
            "input": response["prompt_eval_count"],
            "output": response["eval_count"],
        }
    )
    raw = response["message"]["content"].lower().strip()

    # Robust extraction in case model adds punctuation or conversational prefix
    if "both" in raw:
        return "both"
    elif "jd" in raw:
        return "jd"
    return "resume"


@observe()
@with_retry(max_retries=3, base_delay=5, call_timeout=30)
async def llm_chunk(text: str) -> list[dict]:
    system_prompt = (
        "You are a document structuring specialist. Convert raw text into a structured JSON array of sections.\n\n"
        "DEFINITIONS:\n"
        "- \"heading\": a top-level section title (e.g. \"Experience\", \"Skills\", \"Responsibilities\", \"Education\"). "
        "Headings are short, stand alone on their own line, and introduce a new category of content.\n"
        "- \"subheading\": an entry within a section that groups related bullets together — typically a job title + company + dates (resume), "
        "or a sub-category label under a heading (JD). Not every heading has subheadings — use null if there are none.\n"
        "- \"bullets\": a list of complete, standalone content items belonging to that heading/subheading. Each bullet must be one full, "
        "ungrounded thought — never split a sentence across two bullets, never merge two distinct bullets into one.\n\n"
        "RULES:\n"
        "1. Preserve all original wording. Do not summarize, paraphrase, or drop any content — every line of the input must appear somewhere in the output.\n"
        "2. Never let bullet content leak into a heading or subheading field, and never let a heading get silently absorbed into the previous section's bullets.\n"
        "3. If the input has intro prose with no bullet markers, treat the whole paragraph as a single bullet unless it clearly contains multiple distinct ideas.\n"
        "4. Output ONLY a valid JSON array. No markdown code fences, no explanation, no text before or after the array.\n\n"
        "OUTPUT SCHEMA:\n"
        "[{\"heading\": \"string\", \"subheading\": \"string or null\", \"bullets\": [\"string\", ...]}]\n\n"
        "EXAMPLE INPUT:\n"
        "Responsibilities:\n"
        "- Deploy and configure AI models and infrastructure for enterprise clients\n"
        "- Troubleshoot and resolve technical issues in real-time\n\n"
        "EXAMPLE OUTPUT:\n"
        "[{\"heading\": \"Responsibilities\", \"subheading\": null, \"bullets\": ["
        "\"Deploy and configure AI models and infrastructure for enterprise clients\", "
        "\"Troubleshoot and resolve technical issues in real-time\"]}]"
    )

    user_prompt = f"Now process this text:\n\n{text}"

    response = await ollama_client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        format="json",  # Forces valid JSON schema output mode in Ollama
        options={"temperature": 0.1}
    )

    langfuse_context.update_current_observation(
        usage={
            "input": response["prompt_eval_count"],
            "output": response["eval_count"],
        }
    )

    raw = response["message"]["content"].strip()

    # Strip residual markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
        # Ensure output is a list even if model wrapped it in an object key
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
        return data
    except json.JSONDecodeError:
        return [{"heading": "General", "subheading": None, "bullets": [text]}]


async def could_web_search_help(query:str):
    system_prompt = (
        """You are evaluating whether a live web search could provide genuinely useful 
        information that a resume or job description would never be expected to contain.""
        Answer true if the question is about real-world, external, current, or public information"
        (company news, layoffs, stock price, culture, recent events) — something inherently outside"
        what any resume/JD could contain."
        "Answer false if the question is about the candidate personally (skills, experience) or is"
        inherently private/unanswerable even via web search (salary expectations, marital status).
        The output should be a JSON object with the following format: {"answer": "true" or "false"}
        """
    )

    user_prompt = f"Question: {query}\nAnswer:"

    response = await ollama_client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        format="json",  
        options={"temperature": 0.2}
    )

    langfuse_context.update_current_observation(
        usage={
            "input": response["prompt_eval_count"],
            "output": response["eval_count"],
        }
    )

    raw = response["message"]["content"].lower().strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    
    data = json.loads(raw)
    print(data,"(((((((((((((((())))))))))))))))")
    return str(data["answer"]).strip().lower() == "true"



@observe()
@with_retry(max_retries=3, base_delay=5, call_timeout=30)
async def llm_judge(question: str, retrieved_chunks: str, answer: str):

    print(retrieved_chunks)
    print(answer)
    print(question)
    system_prompt = (
        "You are evaluating the quality of an AI interview coach's answer. "
        "Score the answer on three dimensions and one flag (true or false), each from 1-5, and respond ONLY with valid JSON, "
        "no markdown fences, no explanation outside the JSON."
    )


    user_prompt = f"""<scoring_criteria>
RELEVANCY (1-5): Does the answer directly address what was asked, without drifting into unrelated information?
GROUNDEDNESS (1-5): Specifically check: does the answer attribute any skill, tool, or experience to
the CANDIDATE that only appears in the JD-sourced context, not the resume-sourced context?
If so, that is a fabrication regardless of whether the JD's wording appears in the provided context.
A 1 means the answer claims candidate experience that is only JD language, not resume-sourced.
A 5 means every claim about the candidate is traceable specifically to resume-sourced content.
COMPLETENESS (1-5): Does the answer engage with the specifics (numbers, facts, reasoning) rather than being generic or vague?
COULD_WEB_SEARCH_HELP (true/false): Is this a question about real-world, external, or current
  information (e.g. company news, layoffs, culture, recent events) that the candidate's resume/JD
  would never be expected to contain — where a live web search could provide a genuinely useful
  answer the resume/JD context cannot? Answer true only if the underlying question is legitimately
  answerable via public information, not for personal/private questions (e.g. salary expectations,
  marital status) that no web search could ever resolve.
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

Respond in this exact JSON format: {{"relevancy": <int 1-5>, "groundedness": <int 1-5>, "completeness": <int 1-5>, "could_web_search_help": <bool>, "groundedness_reasoning": "<one sentence>"}}"""

    response = await ollama_client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.2}
    )

    langfuse_context.update_current_observation(
        usage={
            "input": response["prompt_eval_count"],
            "output": response["eval_count"],
        }
)
    try:
        parsed_json = json.loads(response["message"]["content"])
        return parsed_json
    except Exception as e:
        print(e)
        return {
            "status" : "exception",
            "response" : "Exception has occured",
            "groundedness": 0
        }

    