import os
import sys
import logging
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("perplexity")

PPLX_API_KEY = os.environ.get("PPLX_API_KEY")

if not PPLX_API_KEY:
    logging.error("PPLX_API_KEY is not set in the environment.")
    sys.exit(1)

client = AsyncOpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")

@mcp.tool()
async def perplexity_search(query: str) -> Dict[str, Any]:
    """
    Performs a web search using the Perplexity API.

    Args:
        query: The search query.

    Returns:
        A dictionary containing the search results.
    """
    if not PPLX_API_KEY:
        return {"error": "PPLX_API_KEY is not configured."}

    try:
        response = await client.chat.completions.create(
            model="sonar-small-online",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )
        return {"result": response.choices[0].message.content}
    except Exception as e:
        logging.error(f"Perplexity API error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")
