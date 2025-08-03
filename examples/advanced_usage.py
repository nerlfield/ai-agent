#!/usr/bin/env python3
"""
Advanced usage examples showing:
1. Custom execution contexts
2. Action filtering based on context
3. Event bus integration
4. Structured output with agents
5. Multi-step workflows
"""

import asyncio
import logging
from typing import Any

from bubus import EventBus
from pydantic import BaseModel, Field

from ai_agent.actions.registry_helper import register_all_actions
from ai_agent.agent.context import SimpleExecutionContext
from ai_agent.agent.service import GenericAgent
from ai_agent.agent.views import AgentSettings
from ai_agent.controller.service import Controller
from ai_agent.filesystem import FileSystem
from ai_agent.llm.factory import create_llm
from ai_agent.llm.views import LLMType
from ai_agent.registry.views import ActionResult


# Structured output models
class ResearchResult(BaseModel):
	"""Structured output for research tasks"""
	topic: str = Field(description="Research topic")
	summary: str = Field(description="Brief summary of findings")
	key_points: list[str] = Field(description="Key points discovered")
	sources: list[str] = Field(description="Sources used")
	confidence: float = Field(description="Confidence level 0-1", ge=0, le=1)


class ProjectPlan(BaseModel):
	"""Structured output for project planning"""
	project_name: str = Field(description="Name of the project")
	description: str = Field(description="Project description")
	goals: list[str] = Field(description="Project goals")
	milestones: list[dict[str, Any]] = Field(description="Project milestones with dates")
	resources_needed: list[str] = Field(description="Required resources")


# Custom execution context with capabilities
class AdvancedExecutionContext(SimpleExecutionContext):
	"""Advanced execution context with dynamic capabilities"""
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.events_emitted = []
		
		# Set up capability-based action filtering
		self.registry.context_param_types.update({
			'event_bus': EventBus,
			'capabilities': set,
		})
	
	async def execute_action(self, action: ActionModel, **kwargs) -> ActionResult:
		"""Execute with event emission"""
		# Add current capabilities to kwargs
		kwargs['capabilities'] = self.capabilities
		
		# Execute action
		result = await super().execute_action(action, **kwargs)
		
		# Emit event
		if 'event_bus' in kwargs and kwargs['event_bus']:
			event_data = {
				'action': action.model_dump(),
				'result': result.model_dump() if result else None,
				'timestamp': asyncio.get_event_loop().time(),
			}
			self.events_emitted.append(event_data)
			await kwargs['event_bus'].emit('action_executed', event_data)
		
		return result


async def research_agent_example():
	"""Example of agent producing structured research output"""
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	# Set up components
	llm = create_llm(LLMType.ANTHROPIC, model='claude-3-5-sonnet-20241022')
	file_system = FileSystem()
	controller = Controller(output_model=ResearchResult)
	
	# Register actions
	register_all_actions(controller.registry, file_system)
	
	# Create execution context
	execution_context = AdvancedExecutionContext(
		file_system=file_system,
		llm=llm,
		registry=controller.registry,
		capabilities={'file_system', 'web_access', 'research'},
	)
	
	# Research task
	task = """
	Research the topic of 'quantum computing applications in cryptography' and provide a structured summary.
	Save your findings to research_output.json and create a markdown report in quantum_crypto_report.md.
	"""
	
	# Create agent with structured output
	agent = GenericAgent(
		task=task,
		llm=llm,
		execution_context=execution_context,
		controller=controller,
		settings=AgentSettings(
			max_actions_per_step=3,
			use_vision=False,
		),
		structured_output_type=ResearchResult,
		file_system=file_system,
	)
	
	logger.info("🔬 Starting research agent...")
	history = await agent.run(max_steps=10)
	
	# Extract structured output
	if history.last and history.last.model_output:
		output = history.last.model_output
		if hasattr(output, 'structured_output'):
			research = output.structured_output
			logger.info(f"\n📊 Research Results:")
			logger.info(f"Topic: {research.topic}")
			logger.info(f"Confidence: {research.confidence:.2%}")
			logger.info(f"Key Points: {len(research.key_points)}")
			for point in research.key_points:
				logger.info(f"  • {point}")


async def workflow_agent_example():
	"""Example of multi-step workflow with event tracking"""
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	# Set up event bus
	event_bus = EventBus()
	events_captured = []
	
	# Event handler
	async def capture_events(event_data):
		events_captured.append(event_data)
		action = event_data['action']
		logger.info(f"📡 Event: {list(action.keys())[0] if action else 'unknown'}")
	
	event_bus.subscribe('action_executed', capture_events)
	
	# Set up components
	llm = create_llm(LLMType.ANTHROPIC, model='claude-3-5-sonnet-20241022')
	file_system = FileSystem()
	controller = Controller()
	
	register_all_actions(controller.registry, file_system)
	
	execution_context = AdvancedExecutionContext(
		file_system=file_system,
		llm=llm,
		registry=controller.registry,
	)
	
	# Multi-step workflow task
	task = """
	Execute this workflow:
	1. Create a project structure:
	   - Create directories: src/, tests/, docs/
	   - Create README.md with project description
	   - Create src/__init__.py
	2. Generate a simple Python module in src/calculator.py with add, subtract, multiply functions
	3. Create a test file in tests/test_calculator.py with basic test cases
	4. Create a summary in workflow_complete.json with:
	   - List of all created files
	   - Total lines of code written
	   - Timestamp of completion
	"""
	
	agent = GenericAgent(
		task=task,
		llm=llm,
		execution_context=execution_context,
		controller=controller,
		settings=AgentSettings(
			max_actions_per_step=5,
			save_conversation_path='workflow_conversation.txt',
		),
		event_bus=event_bus,
		file_system=file_system,
	)
	
	logger.info("🔧 Starting workflow agent...")
	history = await agent.run(max_steps=20)
	
	# Summary
	logger.info(f"\n📊 Workflow Summary:")
	logger.info(f"Total steps: {len(history)}")
	logger.info(f"Events captured: {len(events_captured)}")
	logger.info(f"Actions executed: {len([e for e in events_captured if 'action' in e])}")
	
	# List created files
	import os
	for root, dirs, files in os.walk('.'):
		for file in files:
			if any(d in root for d in ['src', 'tests', 'docs']):
				logger.info(f"  📄 {os.path.join(root, file)}")


async def capability_based_actions():
	"""Example showing capability-based action filtering"""
	llm = create_llm(LLMType.ANTHROPIC, model='claude-3-5-sonnet-20241022')
	controller = Controller()
	
	# Register actions with capability requirements
	@controller.registry.action(
		description='Special admin action',
		context_filters={'admin_mode': True},
	)
	async def admin_action(params=None):
		return ActionResult(
			success=True,
			extracted_content="Admin action executed",
		)
	
	@controller.registry.action(
		description='Read-only action',
		filter_function=lambda ctx: 'read_only' in ctx.get('capabilities', set()),
	)
	async def read_only_action(params=None):
		return ActionResult(
			success=True,
			extracted_content="Read-only action executed",
		)
	
	# Test with different contexts
	# Context 1: Admin mode
	admin_context = SimpleExecutionContext(
		llm=llm,
		registry=controller.registry,
		capabilities={'admin_mode'},
	)
	admin_context.state.metadata['admin_mode'] = True
	
	# Context 2: Read-only mode  
	readonly_context = SimpleExecutionContext(
		llm=llm,
		registry=controller.registry,
		capabilities={'read_only'},
	)
	
	print("Testing capability-based action filtering...")
	
	# This would only see admin actions
	admin_actions = controller.registry.get_available_actions({'admin_mode': True})
	print(f"Admin context sees {len(admin_actions)} actions")
	
	# This would only see read-only actions
	readonly_actions = controller.registry.get_available_actions({'capabilities': {'read_only'}})
	print(f"Read-only context sees {len(readonly_actions)} actions")


if __name__ == "__main__":
	# Choose which example to run
	
	# Research agent with structured output
	asyncio.run(research_agent_example())
	
	# Multi-step workflow with event tracking
	# asyncio.run(workflow_agent_example())
	
	# Capability-based action filtering
	# asyncio.run(capability_based_actions())