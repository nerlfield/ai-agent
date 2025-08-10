import os
import asyncio
import signal
import json

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

import dotenv
dotenv.load_dotenv()

from src.service import Agent

console = Console()

def render_user(text: str) -> None:
    console.print(Panel.fit(Markdown(text), title="You", border_style="magenta"))

def render_assistant(text: str) -> None:
    console.print(Panel.fit(Markdown(text), title="Assistant", border_style="cyan"))

def render_tool_call(name: str, args: dict) -> None:
    console.print(Panel.fit(f"[bold]{name}[/bold]\n{json.dumps(args, ensure_ascii=False, indent=2)}",
                            title="MCP Tool Call", border_style="yellow"))

def render_reasoning(text: str) -> None:
    console.print(Panel.fit(Markdown(text), title="Reasoning", border_style="green"))

async def main() -> None:
    server_names = ["rules_server.py", "perplexity_server.py", "image_tool_server.py", "planner_server.py"]
    server_paths = [os.path.join(os.path.dirname(__file__), "src", "tools", name) for name in server_names]
    
    agent = Agent()
    await agent.initialize(server_paths)

    console.print("[green]Agent (GPT-5 + MCP) started. Type 'exit' to quit.[/green]")

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: None)

    while True:
        user = input("\n> ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        render_user(user)
        
        async for response in agent.process_message(user):
            if response.reasoning:
                render_reasoning(response.reasoning)
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    render_tool_call(tool_call.name, tool_call.arguments)
            if response.text:
                render_assistant(response.text)

    await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
