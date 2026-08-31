


import os
from tavily import TavilyClient
from dotenv import load_dotenv


load_dotenv()


tavily_api_key = os.getenv("travily_api_key")

client = TavilyClient(api_key=tavily_api_key)



async def web_search(query: str):
    results = client.search(
        query=query,
        search_depth="advanced"
    )
    return "\n".join(f"{r['title']}: {r['content']}" for r in results["results"])


# Tool definition compatible with client.interactions.create()
web_search_tool = {
    "type": "function",
    "name": "web_search",
    "description": "Web search tool to get latest information from the internet",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            }
        },
        "required": ["query"],
    }
}

