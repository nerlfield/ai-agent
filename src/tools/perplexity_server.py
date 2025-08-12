import os
import sys
import logging
from typing import Dict, Any, Optional, Literal
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
async def perplexity_search(
    query: str,
    model: Optional[Literal[
        "sonar",
        "sonar-pro",
        "sonar-reasoning",
        "sonar-reasoning-pro",
        "sonar-deep-research",
    ]] = "sonar",
) -> Dict[str, Any]:
    """
    Performs a web search using the Perplexity API.

    Args:
        query: The search query.
        model: Optional model to use. One of:
            - sonar: Search — Lightweight, cost-effective search model with grounding.
            - sonar-pro: Search — Advanced search with grounding; supports complex queries and follow-ups.
            - sonar-reasoning: Reasoning — Fast, real-time reasoning model for problem-solving with search.
            - sonar-reasoning-pro: Reasoning — Precise reasoning powered by DeepSeek-R1 with Chain of Thought (CoT).
            - sonar-deep-research: Research — Expert-level exhaustive research and comprehensive reports.

    Returns:
        A dictionary containing the search results.
    """
    if not PPLX_API_KEY:
        return {"error": "PPLX_API_KEY is not configured."}

    try:
        selected_model = model or "sonar"
        response = await client.chat.completions.create(
            model=selected_model,
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
