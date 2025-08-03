#!/usr/bin/env python3
"""
Simple agent example demonstrating basic usage of the ai-agent framework.

This example shows how to:
1. Set up a basic agent with file system access
2. Register custom actions
3. Run the agent to complete a task
"""

import asyncio
import logging
from pathlib import Path

from ai_agent.actions.registry_helper import register_all_actions
from ai_agent.agent.context import SimpleExecutionContext
from ai_agent.agent.service import GenericAgent
from ai_agent.agent.views import AgentSettings
from ai_agent.controller.registry.service import Registry
from ai_agent.filesystem import FileSystem
from ai_agent.llm.factory import create_llm
from ai_agent.llm.views import LLMType


async def main():
	"""Run a simple agent example"""
	# Set up logging
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	# Create a working directory for the agent
	working_dir = Path("./agent_workspace")
	working_dir.mkdir(exist_ok=True)
	
	# Initialize components
	file_system = FileSystem(working_directory=working_dir)
	
	# Create LLM (you'll need to set ANTHROPIC_API_KEY environment variable)
	llm = create_llm(LLMType.ANTHROPIC, model='claude-3-5-sonnet-20241022')
	
	# Create registry and register actions
	registry = Registry(context_param_types={'file_system': FileSystem})
	register_all_actions(registry, file_system)
	
	# Create execution context
	execution_context = SimpleExecutionContext(
		file_system=file_system,
		llm=llm,
		registry=registry,
	)
	
	# Create controller
	from ai_agent.controller.service import Controller
	controller = Controller(
		registry=registry,
		context_param_types={'file_system': FileSystem},
	)
	
	# Define the task
	task = """
	Please help me organize my thoughts by:
	1. Creating a file called 'project_ideas.md' with a list of 3 interesting Python project ideas
	2. Creating a file called 'todo.md' with a prioritized list of next steps for the first project idea
	3. Fetching the current Bitcoin price from a public API and saving it to 'bitcoin_price.json'
	4. Creating a summary file 'summary.txt' that contains a brief overview of what was accomplished
	"""
	
	# Configure agent settings
	settings = AgentSettings(
		max_failures=3,
		save_conversation_path=working_dir / "conversation.txt",
		max_actions_per_step=5,
	)
	
	# Create the agent
	agent = GenericAgent(
		task=task,
		llm=llm,
		execution_context=execution_context,
		controller=controller,
		settings=settings,
		file_system=file_system,
	)
	
	logger.info("Starting agent execution...")
	logger.info(f"Task: {task}")
	logger.info(f"Working directory: {working_dir}")
	
	try:
		# Run the agent
		history = await agent.run(max_steps=15)
		
		# Print summary
		logger.info("\n=== Agent Execution Complete ===")
		logger.info(f"Total steps: {len(history)}")
		
		# Check if task was completed
		if history.last and history.last.model_output and history.last.model_output.is_done:
			logger.info("✅ Task completed successfully!")
		else:
			logger.info("❌ Task not fully completed")
		
		# Print errors if any
		errors = history.get_errors()
		if errors:
			logger.warning(f"Errors encountered: {len(errors)}")
			for step, error in errors:
				logger.warning(f"  Step {step}: {error}")
		
		# List created files
		logger.info("\nFiles in workspace:")
		for file in working_dir.glob("*"):
			if file.is_file():
				logger.info(f"  - {file.name} ({file.stat().st_size} bytes)")
		
	except Exception as e:
		logger.error(f"Agent failed with error: {e}")
		raise


async def simple_example():
	"""Even simpler example with minimal setup"""
	from ai_agent.llm.factory import create_llm
	from ai_agent.llm.views import LLMType
	from ai_agent.agent.service import GenericAgent
	from ai_agent.agent.context import SimpleExecutionContext
	from ai_agent.controller.service import Controller
	from ai_agent.filesystem import FileSystem
	from ai_agent.actions.registry_helper import register_all_actions
	
	# Quick setup
	llm = create_llm(LLMType.ANTHROPIC, model='claude-3-5-sonnet-20241022')
	file_system = FileSystem()
	registry = Registry()
	register_all_actions(registry, file_system)
	
	execution_context = SimpleExecutionContext(file_system=file_system, llm=llm, registry=registry)
	controller = Controller(registry=registry)
	
	# Create and run agent
	agent = GenericAgent(
		task="Create a hello.txt file with a friendly greeting",
		llm=llm,
		execution_context=execution_context,
		controller=controller,
		file_system=file_system,
	)
	
	history = await agent.run(max_steps=5)
	print(f"Task completed: {history.last.model_output.is_done if history.last else False}")


if __name__ == "__main__":
	# Run the main example
	asyncio.run(main())
	
	# Or run the simple example
	# asyncio.run(simple_example())