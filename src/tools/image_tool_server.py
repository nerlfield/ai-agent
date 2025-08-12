import os
import sys
import logging
import base64
import httpx
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Union
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("image_processing")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY is not set in the environment.")
    sys.exit(1)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

def to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@mcp.tool()
async def describe_image(source: str) -> Dict[str, Any]:
    """
    Describes an image from a URL or local file path.

    Args:
        source: The URL or local file path of the image.

    Returns:
        A dictionary containing the image description.
    """
    try:
        if source.startswith("http"):
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(source)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(source)

        base64_image = to_base64(image)

        response = await client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return {"description": response.choices[0].message.content}
    except Exception as e:
        logging.error(f"Image processing error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")
