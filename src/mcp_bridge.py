import asyncio, sys, os
from contextlib import AsyncExitStack
from typing import Dict, Tuple, Any, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPManager:
    """
    Manages one or more MCP stdio servers, discovers tools,
    and exposes them in an OpenAI-compatible tool schema.
    """
    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._sessions: List[ClientSession] = []
        self._registry: Dict[str, Tuple[ClientSession, Any]] = {}

    async def start(self, server_paths: List[str]) -> None:
        for path in server_paths:
            if path.endswith(".py"):
                command = sys.executable
                args = [path]
            elif path.endswith(".js"):
                command = "node"
                args = [path]
            else:
                raise ValueError(f"Unsupported server file: {path}")

            # Ensure subprocesses inherit the current environment (including keys loaded via dotenv)
            params = StdioServerParameters(command=command, args=args, env=os.environ.copy())
            stdio = await self._stack.enter_async_context(stdio_client(params))
            read, write = stdio
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools_resp = await session.list_tools()
            for t in tools_resp.tools:
                self._registry[t.name] = (session, t)
            self._sessions.append(session)

    async def aclose(self) -> None:
        await self._stack.aclose()

    def openai_tools_schema(self) -> List[dict]:
        """
        Convert MCP tool descriptors into OpenAI Responses API tool schema (function tools).
        """
        out = []
        for name, (_, t) in self._registry.items():
            out.append({
                "type": "function",
                "name": name,
                "description": t.description or f"MCP tool {name}",
                "parameters": t.inputSchema or {"type": "object", "properties": {}, "additionalProperties": False}
            })
        return out

    async def call(self, tool_name: str, arguments: dict) -> str:
        """
        Invoke the named MCP tool with given args. Returns a stringified payload for the model.
        """
        if tool_name not in self._registry:
            return f"[MCP ERROR] Unknown tool '{tool_name}'"
        session, _meta = self._registry[tool_name]
        result = await session.call_tool(tool_name, arguments or {})
        chunks = []
        for c in result.content:
            if hasattr(c, "text") and c.text is not None:
                chunks.append(str(c.text))
            elif hasattr(c, "type") and c.type == "json" and hasattr(c, "data"):
                import json
                chunks.append(json.dumps(c.data, ensure_ascii=False))
            else:
                chunks.append(str(c))
        return "\n".join(chunks) if chunks else ""
