# from services.embed import llm_judge
from services.load import llm_response

from services.ollama import llm_judge

from numpy import array

from services.ollama import langfuse

import functools
import random
import time


import asyncio
import re





test_case_data = {
  "meta": {
    "created": "2026-08-19",
    "updated": "2026-08-20",
    "source_docs": {
      "resume": "Sankar N — Comcast SDE2, Cognizant Associate — real",
      "jd": "synthetic — Full Stack Engineer, Gen AI, Bangalore, 23-28 LPA"
    },
    "notes": "Level 1 unit tests per Hamel Husain 'Your AI Product Needs Evals'. Each assertion should be checkable in code, not judged by feel. Pass rate is a product decision, not required to be 100%. ground_truth_chunk_id added Wed for retrieval metrics (hit rate / MRR) — sourced from live Chroma dump after stable hash-based id fix.",
    "categories": ["resume_only", "jd_only", "both", "generic"],
    "chunk_id_reference": {
      "jd_title": "jd:Full Stack Engineer — Generative AI:Bangalore (Hybrid)  ·  23–28 LPA  ·  4–7 years experience",
      "jd_responsibilities": "jd:Responsibilities:None",
      "jd_required": "jd:Required:None",
      "jd_preferred": "jd:Preferred:None",
      "resume_header": "resume:25a047bb6238ae62",
      "resume_skills": "resume:85141ec3ad36324b",
      "resume_comcast": "resume:80c1bfcbade2e314",
      "resume_cognizant": "resume:cfdb6c606e864987",
      "resume_education": "resume:2605c7b4783498df",
      "resume_achievements": "resume:8fdb6f7e263186c2"
    }
  },
  "test_cases": [
    # {"id": "R01", "category": "resume", "question": "What is my current role and company?", "expected_contains": ["Comcast", "Engineer 2"], "assertion": "all(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["resume:80c1bfcbade2e314"], "notes": "direct fact lookup"},
    # {"id": "R02", "category": "resume_only", "question": "What was my role before Comcast?", "expected_contains": ["Cognizant", "Associate"], "assertion": "all(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["resume:cfdb6c606e864987"]},
    # {"id": "R03", "category": "resume_only", "question": "What backend frameworks or tools have I used?", "expected_contains": ["Node.js", "Fast API"], "assertion": "any(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["resume:85141ec3ad36324b"], "notes": "skills lookup, partial match ok"},
    # {"id": "R04", "category": "resume_only", "question": "What did I build for Jira integration at Comcast?", "expected_contains": ["Jira", "LinX", "Node.js", "React"], "assertion": "sum(kw.lower() in response.lower() for kw in expected_contains) >= 2", "ground_truth_chunk_id": ["resume:80c1bfcbade2e314"]},
    # {"id": "R05", "category": "resume_only", "question": "What performance improvement did I achieve at Cognizant through code splitting?", "expected_contains": ["30%"], "assertion": "'30%' in response or '30 %' in response", "ground_truth_chunk_id": ["resume:cfdb6c606e864987"], "notes": "specific metric recall"},
    # {"id": "R06", "category": "resume_only", "question": "What was the speed improvement from the bulk data operations I built?", "expected_contains": ["5x"], "assertion": "'5x' in response.lower() or '5 x' in response.lower()", "ground_truth_chunk_id": ["resume:80c1bfcbade2e314"]},
    # {"id": "R07", "category": "resume_only", "question": "What unit test coverage did I maintain at Cognizant?", "expected_contains": ["85%"], "assertion": "'85%' in response", "ground_truth_chunk_id": ["resume:cfdb6c606e864987"]},
    # {"id": "R08", "category": "resume_only", "question": "How many user roles did the RBAC system I built support?", "expected_contains": ["7", "8"], "assertion": "'7' in response or '7\u20138' in response or '7-8' in response", "ground_truth_chunk_id": ["resume:80c1bfcbade2e314"]},
    # {"id": "R09", "category": "resume_only", "question": "What GenAI-related work have I done professionally?", "expected_contains": ["RAG", "Text-to-SQL", "Retrieval-Augmented"], "assertion": "any(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["resume:80c1bfcbade2e314"]},
    # {"id": "R10", "category": "resume_only", "question": "What awards or recognitions have I received?", "expected_contains": ["Spotlight", "Sapphire", "Lean Into Action"], "assertion": "any(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["resume:8fdb6f7e263186c2"]},
    # {"id": "R11", "category": "resume_only", "question": "Where did I complete my engineering degree?", "expected_contains": ["R.M.D", "RMD"], "assertion": "'rmd' in response.lower().replace('.', '')", "ground_truth_chunk_id": ["resume:2605c7b4783498df"]},
    {"id": "R12", "category": "resume", "question": "What is my current salary?", "expected_behavior": "refuse_or_not_in_context", "assertion": "not any(c.isdigit() for c in response) or 'not available' in response.lower() or \"don't have\" in response.lower() or 'no information' in response.lower()", "ground_truth_chunk_id": [], "notes": "out-of-scope: not on resume, tests against hallucinated numbers. Empty ground truth is intentional \u2014 no chunk should be a valid 'hit' here."},
    {"id": "R13", "category": "resume", "question": "What is my marital status?", "expected_behavior": "refuse_or_not_in_context", "assertion": "'not' in response.lower() and ('available' in response.lower() or 'mentioned' in response.lower() or 'information' in response.lower())", "ground_truth_chunk_id": [], "notes": "out-of-scope, personal, not on resume. Empty ground truth intentional."},
    # {"id": "R14", "category": "resume_only", "question": "How many knowledge transfer sessions did I conduct at Cognizant?", "expected_contains": ["10"], "assertion": "'10' in response", "ground_truth_chunk_id": ["resume:cfdb6c606e864987"]},
    # {"id": "R15", "category": "resume", "question": "What visualization or mapping work have I done?", "expected_contains": ["OpenStreetMap", "geospatial"], "assertion": "any(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["resume:80c1bfcbade2e314"]},

    # {"id": "J01", "category": "jd", "question": "How many years of experience does this JD require?", "expected_contains": ["4"], "assertion": "'4' in response", "ground_truth_chunk_id": ["jd:Required:None"]},
    # {"id": "J02", "category": "jd_only", "question": "What is the location for this role?", "expected_contains": ["Bangalore"], "assertion": "'bangalore' in response.lower()", "ground_truth_chunk_id": ["jd:Full Stack Engineer \u2014 Generative AI:Bangalore (Hybrid)  \u00b7  23\u201328 LPA  \u00b7  4\u20137 years experience"]},
    # {"id": "J03", "category": "jd_only", "question": "What agentic frameworks does this JD mention as preferred?", "expected_contains": ["LangGraph", "CrewAI"], "assertion": "any(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["jd:Preferred:None"]},
    # {"id": "J04", "category": "jd_only", "question": "Does this JD require MCP experience or is it preferred?", "expected_contains": ["preferred", "familiarity"], "assertion": "'preferred' in response.lower() or 'familiarity' in response.lower()", "ground_truth_chunk_id": ["jd:Preferred:None"], "notes": "tests precision on required-vs-preferred distinction"},
    # {"id": "J05", "category": "jd_only", "question": "What frontend technologies does this JD ask for?", "expected_contains": ["React", "Next.js"], "assertion": "any(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["jd:Responsibilities:None", "jd:Required:None"], "notes": "React/Next.js appears in Responsibilities; React also implied in Required \u2014 either is a valid hit"},
    # {"id": "J06", "category": "jd_only", "question": "What is the salary range for this role?", "expected_contains": ["23", "28", "LPA"], "assertion": "'23' in response and '28' in response", "ground_truth_chunk_id": ["jd:Full Stack Engineer \u2014 Generative AI:Bangalore (Hybrid)  \u00b7  23\u201328 LPA  \u00b7  4\u20137 years experience"]},
    # {"id": "J07", "category": "jd_only", "question": "What cloud provider does this JD prefer?", "expected_contains": ["AWS"], "assertion": "'aws' in response.lower()", "ground_truth_chunk_id": ["jd:Required:None"]},
    {"id": "J08", "category": "jd", "question": "Does this JD mention team size or reporting structure?", "expected_behavior": "refuse_or_not_in_context", "assertion": "'not' in response.lower() and ('mentioned' in response.lower() or 'specif' in response.lower() or 'available' in response.lower())", "ground_truth_chunk_id": [], "notes": "ambiguous/absent info \u2014 tests against over-refusal AND against fabrication; real JD text has no such detail. Empty ground truth intentional."},
    # {"id": "J09", "category": "jd", "question": "What evaluation-related skills does this JD value?", "expected_contains": ["evaluation", "unit tests", "eval"], "assertion": "any(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["jd:Preferred:None"]},

    # {"id": "B01", "category": "both", "question": "Do I meet the minimum experience requirement for this JD?", "expected_behavior": "affirmative_with_reasoning", "assertion": "('yes' in response.lower() or 'meet' in response.lower()) and any(y in response for y in ['2021','2024','4','5'])", "ground_truth_chunk_id": ["jd:Required:None", "resume:80c1bfcbade2e314", "resume:cfdb6c606e864987"], "notes": "fit assessment \u2014 resume shows 2021-present (~5yrs) vs JD's 4+ yrs. Multiple valid ground-truth chunks since answer needs both sides."},
    # {"id": "B02", "category": "both", "question": "What skills does this JD ask for that aren't clearly on my resume?", "expected_contains": ["LangGraph", "MCP", "fine-tuning"], "assertion": "any(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["jd:Preferred:None", "resume:85141ec3ad36324b"], "notes": "gap identification \u2014 resume has RAG/GenAI PoC but not explicit LangGraph/MCP"},
    # {"id": "B03", "category": "both", "question": "How should I frame my AWS/deployment work for this JD, given I don't have explicit AWS experience listed on my resume?", "expected_behavior": "no_fabrication", "assertion": "'aws' not in [c for c in response.lower().split() if 'resume' in response.lower()] or 'not' in response.lower() or 'suggest' in response.lower() or 'consider' in response.lower()", "ground_truth_chunk_id": ["jd:Required:None", "resume:85141ec3ad36324b"], "notes": "important negative test: resume has NO AWS mentioned \u2014 answer must not claim resume shows AWS experience"},
    # {"id": "B04", "category": "both", "question": "Does my RAG/Text-to-SQL PoC work at Comcast align with this JD's RAG pipeline requirement?", "expected_contains": ["RAG", "Retrieval"], "assertion": "any(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["resume:80c1bfcbade2e314", "jd:Responsibilities:None"]},
    # {"id": "B05", "category": "both", "question": "What's my strongest talking point for the 'full-stack' part of this JD?", "expected_contains": ["React", "Node.js", "Redux"], "assertion": "any(kw.lower() in response.lower() for kw in expected_contains)", "ground_truth_chunk_id": ["resume:85141ec3ad36324b", "jd:Required:None"]},
    # {"id": "B06", "category": "both", "question": "Is my current title (Engineer 2) equivalent to what this JD is hiring for?", "expected_behavior": "reasoned_comparison", "assertion": "len(response) > 40", "ground_truth_chunk_id": ["resume:80c1bfcbade2e314", "jd:Full Stack Engineer \u2014 Generative AI:Bangalore (Hybrid)  \u00b7  23\u201328 LPA  \u00b7  4\u20137 years experience"], "notes": "open-ended reasoning question, weak answer-level assertion by design \u2014 flag for Level 2 human/model eval instead of Level 1. Retrieval-level ground truth still meaningful even if answer assertion is weak."},
    # {"id": "B07", "category": "both", "question": "Do I have containerization experience relevant to this JD's deployment requirement?", "expected_contains": ["Docker"], "assertion": "'docker' in response.lower()", "ground_truth_chunk_id": ["resume:85141ec3ad36324b", "jd:Required:None"]},
    # {"id": "B08", "category": "both", "question": "What tool-calling or agent-building experience do I have relevant to this JD?", "expected_behavior": "honest_gap_admission", "assertion": "'not' in response.lower() or 'no' in response.lower() or 'limited' in response.lower() or 'poc' in response.lower()", "ground_truth_chunk_id": ["resume:80c1bfcbade2e314", "jd:Preferred:None"], "notes": "resume shows GenAI PoC/RAG but no explicit agent/tool-calling \u2014 tests against overclaiming fit"}
  ],
  "generic_assertions": [
    {"id": "G01", "check": "no_fabricated_company_names", "assertion": "no company name appears in response that isn't in {Comcast, Cognizant} unless quoting the JD verbatim", "applies_to": "all resume_only and both queries"},
    {"id": "G02", "check": "citation_grounding", "assertion": "every cited chunk_id in response.citations exists in the actual retrieved chunk set for that query", "applies_to": "all queries with citations"},
    {"id": "G03", "check": "correct_routing", "assertion": "response.filterBy == expected_route for the query category (resume/jd/both)", "applies_to": "all queries", "notes": "mechanism confirmed working (Aug 20): LLM classifies query as resume/jd/both, feeds filterBy into Chroma where clause. Concrete per-query confirmation still pending automated run."},
    {"id": "G04", "check": "no_exposed_internal_ids", "assertion": "re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', response) is None", "applies_to": "all queries", "notes": "adapted from Hamel's UUID example \u2014 checks no raw chunk/doc IDs leak into user-facing answer. Chunk ids are now short hashes (e.g. resume:80c1bfcbade2e314), not full UUIDs \u2014 regex may need updating to also catch this shorter hash format leaking into responses."}
  ]
}

def get_MRR(citations: list[dict], test_case : dict) -> float:
    mrr = 0
    if test_case["ground_truth_chunk_id"] == []:
        return 0
    if citations is None:
        return 0
    for rank, citation in enumerate(citations):
        if citation.get("id") in test_case["ground_truth_chunk_id"]:
            mrr += 1/(rank+1)
    print(test_case, mrr/len(test_case["ground_truth_chunk_id"]))
    print("------------------------------------------")
    return mrr/len(test_case["ground_truth_chunk_id"])

def get_hit_rate(citations: list[dict], test_case : dict) -> float:
    for citation in citations:
        if citation.get("id") in test_case["ground_truth_chunk_id"]:
            return 1
    return 0

async def eval_metrics():
    total_mrr = 0
    total_hit_rate = 0
    test_cases = test_case_data["test_cases"]
    judge = []
    for i, tc in enumerate(test_cases):
        print(f"[evals] [{i+1}/{len(test_cases)}] Running: {tc['id']} — {tc['question'][:60]}")

        response = await llm_response(tc["question"])
        langfuse.flush()
        curr_mrr = get_MRR(response.get("citations", []), tc)
        curr_hit_rate = get_hit_rate(response.get("citations", []), tc)
        total_mrr += curr_mrr 
        total_hit_rate += curr_hit_rate
        judge_response = await llm_judge(tc["question"], "\n".join([c["chunk"] for c in response["citations"]]), response["answer"])
        print(judge_response)
        judge.append(judge_response)
        break


    return {
        "mrr" : total_mrr / len(test_cases),
        "hit_rate" : total_hit_rate / len(test_cases),
        "judge_response" : judge
    }


