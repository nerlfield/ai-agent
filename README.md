# AI Agent Framework

A general-purpose AI agent framework extracted from browser-use, designed for building autonomous agents that can execute multi-step tasks using LLMs.

## Features

- 🤖 **Multi-LLM Support**: Works with Anthropic, OpenAI, Google, Groq, and Ollama
- 🔧 **Extensible Action System**: Easy to create and register custom actions
- 📁 **Built-in File System**: Safe file operations within a working directory
- 🔄 **Async First**: Fully asynchronous execution
- 📊 **Structured Output**: Support for typed, structured responses
- 🎯 **Smart Action Discovery**: Context-aware action filtering
- 📝 **Conversation Logging**: Save and replay agent conversations
- 🔍 **Token Usage Tracking**: Monitor LLM token usage and costs
- 🔌 **MCP Ready**: Prepared for Model Context Protocol integration

## Quick Start

```python
import asyncio
import logging
from ai_agent.agent.service import GenericAgent
from ai_agent.agent.context import SimpleExecutionContext
from ai_agent.controller.service import Controller
from ai_agent.filesystem import FileSystem
from ai_agent.llm.factory import create_llm
from ai_agent.llm.views import LLMType
from ai_agent.actions.registry_helper import register_all_actions

async def main():
    # Enable debug logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Set up components
    llm = create_llm(LLMType.ANTHROPIC, model='claude-3-5-sonnet-20241022')
    file_system = FileSystem()
    controller = Controller()
    
    # Register built-in actions
    register_all_actions(controller.registry, file_system)
    
    # Create execution context
    execution_context = SimpleExecutionContext(
        file_system=file_system,
        llm=llm,
        registry=controller.registry,
    )
    
    # Create and run agent
    agent = GenericAgent(
        task="Create a hello.txt file with a greeting",
        llm=llm,
        execution_context=execution_context,
        controller=controller,
        file_system=file_system,
    )
    
    history = await agent.run(max_steps=5)
    if history.last and history.last.model_output:
        print(f"Task completed: {history.last.model_output.is_done}")
        if history.last.model_output.is_done:
            # Check if hello.txt was created
            from pathlib import Path
            hello_file = Path("hello.txt")
            if hello_file.exists():
                print(f"✅ Success! hello.txt was created with content: {hello_file.read_text()}")
            else:
                print("❌ Task marked done but hello.txt was not created")
    else:
        print("Agent failed to complete the task")

asyncio.run(main())
```

## Installation

```bash
pip install -e .
```

## Environment Setup

Set your LLM API keys:

```bash
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here
export GROQ_API_KEY=your_key_here
```

## Creating Custom Actions

```python
from ai_agent.actions.base import BaseAction, ActionParameter, ActionContext
from ai_agent.registry.views import ActionResult
from pydantic import Field

class MyActionParams(ActionParameter):
    message: str = Field(description="Message to process")

class MyCustomAction(BaseAction[MyActionParams, ActionResult, dict]):
    name = 'my_action'
    description = 'My custom action'
    
    async def execute(self, parameters: MyActionParams, context: ActionContext[dict]) -> ActionResult:
        # Your action logic here
        return ActionResult(
            success=True,
            extracted_content=f"Processed: {parameters.message}"
        )

# Register with controller
@controller.registry.action(
    description='My custom action',
    param_model=MyCustomAction().get_parameter_model(),
)
async def my_action(params):
    action = MyCustomAction()
    context = ActionContext(data={})
    return await action.execute(params, context)
```

## Built-in Actions

The framework comes with several built-in actions:

### File Operations
- `read_file` - Read file contents
- `write_file` - Write or append to files
- `delete_file` - Delete files
- `list_files` - List files in directory
- `create_directory` - Create directories

### Data Processing
- `extract_text` - Extract text using regex
- `parse_json` - Parse and query JSON data
- `format_text` - Format text with templates
- `count_words` - Count words, lines, characters
- `replace_text` - Find and replace text

### Web Operations
- `http_request` - Make HTTP requests
- `http_get` - Simplified GET requests

## Examples

See the `examples/` directory for detailed examples:

- **simple_agent.py** - Basic file operations and API calls
- **custom_actions.py** - Creating and registering custom actions
- **advanced_usage.py** - Structured output, workflows, and event tracking

## Architecture

The framework consists of several key components:

- **Agent**: Orchestrates task execution with sophisticated retry and error handling
- **Controller**: Manages action execution and registry
- **Registry**: Handles action registration, discovery, and filtering
- **ExecutionContext**: Provides environment and context for actions
- **FileSystem**: Safe file operations within working directory
- **Actions**: Modular, reusable units of work
- **LLM Integration**: Unified interface for multiple LLM providers

## Development

```bash
pip install -e ".[dev]"
ruff format .
ruff check .
pyright
pytest
```

## License

MIT License