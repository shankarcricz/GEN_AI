
# import sys
# import os

# # Add project root (one level up from this file's folder) to the search path
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from services.load import retrieve
# from services.ollama import classification_of_question
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ollama import classification_of_question   # just the classifier
from services.chroma import fetch_query_results            # just pgvector retrieval

# async def retrieve_chunks(query:str):
#         raw_classification = await classification_of_question(query)
#         raw_classification = raw_classification.strip().lower()
    
#         filterBy = ''
#         # Parse LLM output — it may return verbose text; extract the known keyword
#         if "both" in raw_classification:
#             filterBy = "both"
#         elif "jd" in raw_classification or "job description" in raw_classification:
#             filterBy = "jd"
#         else:
#             filterBy = "resume"  # default fallback
    
#         print(f"[DEBUG] query='{query}' | raw_classification='{raw_classification}' | filterBy='{filterBy}'")
#         results = await retrieve(query, n_results=5, max_distance=0.42, filterBy=filterBy)  
#         print(results)
#         return results


async def retrieve_chunks(query: str):
    classification = await classification_of_question(query)
    classification = classification.strip().lower()
    results = await fetch_query_results(query, filterBy=classification)
    return results