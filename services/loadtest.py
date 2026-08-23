import asyncio
import httpx
import time

async def hit_endpoint(client, query):
    start = time.time()
    try:
        resp = await client.get("http://localhost:8000/retrieve", params={"query": query})
        return {"status": resp.status_code, "time": time.time() - start}
    except Exception as e:
        return {"error": str(e), "time": time.time() - start}

async def run_load_test():
    queries = ["What is my current role?"] * 20  # or vary them
    async with httpx.AsyncClient(timeout=120) as client:
        results = await asyncio.gather(*[hit_endpoint(client, q) for q in queries])
    return results