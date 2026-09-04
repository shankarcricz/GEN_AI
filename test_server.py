from mcp.server import MCPServer
from tools.retrieve import retrieve_chunks

mcp = MCPServer("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b



@mcp.tool()
async def fetch_chunks(query: str):
    """Retrieve relevant chunks from the candidate's resume and the job description
    to answer questions about their skills, experience, or fit for the role."""
    results = await retrieve_chunks(query)
    return results

if __name__ == "__main__":
    mcp.run()