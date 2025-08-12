import os
import sys
import logging
import json
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("planner")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY is not set in the environment.")
    sys.exit(1)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

PLANNER_PROMPT = """
You are a planner agent. Your job is to take a complex task and break it down into a series of smaller, manageable steps.
The output should be a JSON object with a "plan" key, which is a list of steps.
Each step in the plan should be an object with a "tool" key and an "args" key.
The "tool" key should be the name of the tool to call, and the "args" key should be a dictionary of arguments for the tool.
For example:
{
  "plan": [
    {
      "tool": "perplexity_search",
      "args": {
        "query": "What is the weather in San Francisco?"
      }
    },
    {
      "tool": "some_other_tool",
      "args": {
        "arg1": "value1",
        "arg2": "value2"
      }
    }
  ]
}
"""

@mcp.tool()
async def create_plan(task: str) -> Dict[str, Any]:
    """
    Creates a plan for a complex task.

    Args:
        task: The complex task to create a plan for.

    Returns:
        A dictionary containing the plan.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": PLANNER_PROMPT,
                },
                {
                    "role": "user",
                    "content": task,
                },
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Planner error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")
