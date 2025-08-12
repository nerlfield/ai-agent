import json
from typing import List, Dict, Any, Optional, NamedTuple, AsyncGenerator

from openai import AsyncOpenAI

from src.config import MODEL, OPENAI_API_KEY
from src.mcp_bridge import MCPManager

class ToolCall(NamedTuple):
    name: str
    arguments: Dict[str, Any]

class AgentResponse(NamedTuple):
    text: Optional[str]
    tool_calls: List[ToolCall]
    reasoning: Optional[str] = None

SYSTEM_DEV_PROMPT = (
    "You are helpful assistant, an agentic writing copilot.\n"
    "- You can call tools (exposed via MCP) when helpful.\n"
    "- Prefer updating persistent 'rules' via MCP tools when a user asks for changes in style.\n"
    "- After calling tools, integrate results into a concise reply.\n"
    "- Keep answers precise; default verbosity=low unless asked.\n"
)

class Agent:
    def __init__(self):
        self.client = None
        self.mcp = None
        self.messages: List[Dict[str, Any]] = [
            {"role": "developer", "content": SYSTEM_DEV_PROMPT}
        ]

    async def initialize(self, mcp_servers: List[str] = None):
        self.mcp = MCPManager()
        if mcp_servers:
            await self.mcp.start(mcp_servers)
        
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def process_message(self, user_input: str) -> AsyncGenerator[AgentResponse, None]:
        self.messages.append({"role": "user", "content": user_input})

        resp = await self.client.responses.create(
            model=MODEL,
            input=self.messages,
            tools=self.mcp.openai_tools_schema(),
            text={"format": {"type": "text"}, "verbosity": "low"},
            reasoning={"effort": "low", "summary": "auto"}
        )

        while True:
            assistant_text_parts: List[str] = []
            tool_calls: List[Any] = []
            reasoning_parts: List[str] = []

            for item in resp.output:
                if item.type == "reasoning":
                    if item.content:
                        reasoning_parts.append(str(item.content))
                    elif item.summary:
                        for summary_item in item.summary:
                            reasoning_parts.append(summary_item.text)
                elif item.type == "message" and item.role == "assistant":
                    for c in item.content or []:
                        if c.type == "output_text" and c.text:
                            assistant_text_parts.append(c.text)
                elif item.type in ("function_tool_call", "function_call", "tool_call"):
                    tool_calls.append(item)

            if tool_calls:
                tool_calls_data = []
                tool_outputs = []

                for call in tool_calls:
                    name = call.name
                    call_id = call.call_id
                    arguments = call.arguments or {}
                    if not isinstance(arguments, dict):
                        try:
                            arguments = json.loads(arguments) if arguments else {}
                        except Exception:
                            arguments = {}

                    tool_calls_data.append(ToolCall(name=name, arguments=arguments))
                    result_str = await self.mcp.call(name, arguments)
                    tool_outputs.append({
                        "tool_call_id": call_id,
                        "output": result_str
                    })

                    # Check if the tool call was to create a plan
                    if name == "create_plan" and result_str:
                        try:
                            plan_data = json.loads(result_str)
                            if "plan" in plan_data and isinstance(plan_data["plan"], list):
                                # Execute the plan
                                plan_results = []
                                for step in plan_data["plan"]:
                                    if "tool" in step and "args" in step:
                                        step_tool = step["tool"]
                                        step_args = step["args"]
                                        step_result = await self.mcp.call(step_tool, step_args)
                                        plan_results.append({
                                            "step": step,
                                            "result": step_result
                                        })

                                # Combine plan results into a single output
                                final_output = "Plan executed. Results:\n" + json.dumps(plan_results, indent=2)
                                tool_outputs[-1]["output"] = final_output
                        except json.JSONDecodeError:
                            # Not a valid plan, proceed as normal
                            pass


                reasoning_text = "\n".join(reasoning_parts).strip() if reasoning_parts else None
                yield AgentResponse(text=None, tool_calls=tool_calls_data, reasoning=reasoning_text)

                resp = await self.client.responses.create(
                    model=MODEL,
                    previous_response_id=resp.id,
                    input=[{
                        "type": "function_call_output",
                        "call_id": o["tool_call_id"],
                        "output": o["output"],
                    } for o in tool_outputs],
                    reasoning={"effort": "low", "summary": "auto"}
                )
                continue

            final_text = "\n".join(assistant_text_parts).strip()
            reasoning_text = "\n".join(reasoning_parts).strip() if reasoning_parts else None
            
            if final_text or reasoning_text:
                if final_text:
                    self.messages.append({"role": "assistant", "content": final_text})
                yield AgentResponse(text=final_text if final_text else None, tool_calls=[], reasoning=reasoning_text)
            break

    async def close(self):
        if self.mcp:
            await self.mcp.aclose()