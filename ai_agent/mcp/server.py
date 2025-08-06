"""MCP Server for ai-agent - exposes generic agent capabilities via Model Context Protocol.

This server provides tools for:
- Running autonomous agent tasks with customizable execution contexts
- Direct agent control (pause, resume, stop, status monitoring)
- Action registry introspection and management
- Execution context configuration

Usage:
    python -m ai_agent.mcp.server

Or as an MCP server in Claude Desktop or other MCP clients:
    {
        "mcpServers": {
            "ai-agent": {
                "command": "python",
                "args": ["-m", "ai_agent.mcp.server"],
                "env": {
                    "OPENAI_API_KEY": "sk-proj-1234567890",
                }
            }
        }
    }
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import and configure logging to use stderr before other imports
def _configure_mcp_server_logging():
	"""Configure logging for MCP server mode - redirect all logs to stderr to prevent JSON RPC interference."""
	# Set environment to suppress ai-agent logging during server mode
	os.environ['AI_AGENT_LOGGING_LEVEL'] = 'error'
	os.environ['AI_AGENT_SETUP_LOGGING'] = 'false'  # Prevent automatic logging setup

	# Configure logging to stderr for MCP mode
	logging.root.handlers = []
	stderr_handler = logging.StreamHandler(sys.stderr)
	stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	logging.root.addHandler(stderr_handler)
	logging.root.setLevel(logging.ERROR)

	# Configure all existing loggers to use stderr
	for name in list(logging.root.manager.loggerDict.keys()):
		logger_obj = logging.getLogger(name)
		logger_obj.handlers = []
		logger_obj.addHandler(stderr_handler)
		logger_obj.setLevel(logging.ERROR)
		logger_obj.propagate = False


# Configure MCP server logging before any ai_agent imports
_configure_mcp_server_logging()

# Import ai_agent modules
from ai_agent.agent.context import SimpleExecutionContext
from ai_agent.agent.service import GenericAgent, ExecutionContext, Controller
from ai_agent.registry import ActionRegistry, Registry
from ai_agent.llm.factory import create_llm
from ai_agent.llm.types import LLMType
from ai_agent.filesystem.file_system import FileSystem
from ai_agent.agent.views import AgentSettings, AgentState

logger = logging.getLogger(__name__)


def _ensure_all_loggers_use_stderr():
	"""Ensure ALL loggers only output to stderr, not stdout."""
	# Get the stderr handler
	stderr_handler = None
	for handler in logging.root.handlers:
		if hasattr(handler, 'stream') and handler.stream == sys.stderr:  # type: ignore
			stderr_handler = handler
			break

	if not stderr_handler:
		stderr_handler = logging.StreamHandler(sys.stderr)
		stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

	# Configure root logger
	logging.root.handlers = [stderr_handler]
	logging.root.setLevel(logging.ERROR)

	# Configure all existing loggers
	for name in list(logging.root.manager.loggerDict.keys()):
		logger_obj = logging.getLogger(name)
		logger_obj.handlers = [stderr_handler]
		logger_obj.setLevel(logging.ERROR)
		logger_obj.propagate = False


# Ensure stderr logging after all imports
_ensure_all_loggers_use_stderr()


# Try to import MCP SDK
try:
	import mcp.server.stdio
	import mcp.types as types
	from mcp.server import NotificationOptions, Server
	from mcp.server.models import InitializationOptions

	MCP_AVAILABLE = True

	# Configure MCP SDK logging to stderr as well
	mcp_logger = logging.getLogger('mcp')
	mcp_logger.handlers = []
	mcp_logger.addHandler(logging.root.handlers[0] if logging.root.handlers else logging.StreamHandler(sys.stderr))
	mcp_logger.setLevel(logging.ERROR)
	mcp_logger.propagate = False
except ImportError:
	MCP_AVAILABLE = False
	logger.error('MCP SDK not installed. Install with: pip install mcp')
	sys.exit(1)


class AIAgentServer:
	"""MCP Server for ai-agent capabilities."""

	def __init__(self):
		# Ensure all logging goes to stderr (in case new loggers were created)
		_ensure_all_loggers_use_stderr()

		self.server = Server('ai-agent')
		self.agents: Dict[str, GenericAgent] = {}
		self.execution_contexts: Dict[str, ExecutionContext] = {}
		self.action_registry = ActionRegistry()
		self.registry = Registry()
		self.file_system: Optional[FileSystem] = None
		self._start_time = time.time()

		# Setup handlers
		self._setup_handlers()

	def _setup_handlers(self):
		"""Setup MCP server handlers."""

		@self.server.list_tools()
		async def handle_list_tools() -> List[types.Tool]:
			"""List all available ai-agent tools."""
			return [
				# Agent execution tools
				types.Tool(
					name='run_agent_task',
					description='Run an autonomous agent task with configurable settings',
					inputSchema={
						'type': 'object',
						'properties': {
							'task': {
								'type': 'string',
								'description': 'The high-level goal and detailed description of the task the AI agent needs to attempt'
							},
							'max_steps': {
								'type': 'integer',
								'description': 'Maximum number of steps the agent can take',
								'default': 50
							},
							'llm_provider': {
								'type': 'string',
								'description': 'LLM provider to use (openai, anthropic, google, groq, etc.)',
								'default': 'openai'
							},
							'llm_model': {
								'type': 'string',
								'description': 'LLM model to use (e.g., gpt-4, claude-3-opus-20240229)',
								'default': 'gpt-4'
							},
							'execution_context_type': {
								'type': 'string',
								'description': 'Type of execution context to use (generic, file_system, web, custom)',
								'default': 'generic'
							},
							'settings': {
								'type': 'object',
								'description': 'Agent settings override',
								'properties': {
									'use_thinking': {'type': 'boolean', 'default': True},
									'use_vision': {'type': 'boolean', 'default': False},
									'max_actions_per_step': {'type': 'integer', 'default': 5},
									'calculate_cost': {'type': 'boolean', 'default': True}
								}
							}
						},
						'required': ['task']
					}
				),
				
				types.Tool(
					name='step_agent',
					description='Execute a single step for an existing agent',
					inputSchema={
						'type': 'object',
						'properties': {
							'agent_id': {
								'type': 'string',
								'description': 'ID of the agent to step'
							}
						},
						'required': ['agent_id']
					}
				),

				# Agent control tools
				types.Tool(
					name='get_agent_status',
					description='Get the current status of an agent',
					inputSchema={
						'type': 'object',
						'properties': {
							'agent_id': {
								'type': 'string',
								'description': 'ID of the agent to check status for'
							}
						},
						'required': ['agent_id']
					}
				),

				types.Tool(
					name='pause_agent',
					description='Pause an active agent',
					inputSchema={
						'type': 'object',
						'properties': {
							'agent_id': {
								'type': 'string',
								'description': 'ID of the agent to pause'
							}
						},
						'required': ['agent_id']
					}
				),

				types.Tool(
					name='resume_agent',
					description='Resume a paused agent',
					inputSchema={
						'type': 'object',
						'properties': {
							'agent_id': {
								'type': 'string',
								'description': 'ID of the agent to resume'
							}
						},
						'required': ['agent_id']
					}
				),

				types.Tool(
					name='stop_agent',
					description='Stop an active agent',
					inputSchema={
						'type': 'object',
						'properties': {
							'agent_id': {
								'type': 'string',
								'description': 'ID of the agent to stop'
							}
						},
						'required': ['agent_id']
					}
				),

				types.Tool(
					name='list_agents',
					description='List all active agents',
					inputSchema={
						'type': 'object',
						'properties': {}
					}
				),

				# Action registry introspection tools
				types.Tool(
					name='list_actions',
					description='List all available actions in the registry',
					inputSchema={
						'type': 'object',
						'properties': {
							'category': {
								'type': 'string',
								'description': 'Filter actions by category (optional)'
							},
							'tags': {
								'type': 'array',
								'items': {'type': 'string'},
								'description': 'Filter actions by tags (optional)'
							}
						}
					}
				),

				types.Tool(
					name='get_action_info',
					description='Get detailed information about a specific action',
					inputSchema={
						'type': 'object',
						'properties': {
							'action_name': {
								'type': 'string',
								'description': 'Name of the action to get information about'
							}
						},
						'required': ['action_name']
					}
				),

				types.Tool(
					name='get_registry_stats',
					description='Get comprehensive statistics about the action registry',
					inputSchema={
						'type': 'object',
						'properties': {}
					}
				),

				# Execution context tools
				types.Tool(
					name='create_execution_context',
					description='Create a new execution context for agent operations',
					inputSchema={
						'type': 'object',
						'properties': {
							'context_type': {
								'type': 'string',	
								'description': 'Type of execution context (generic, file_system, web, custom)',
								'default': 'generic'
							},
							'config': {
								'type': 'object',
								'description': 'Configuration for the execution context'
							}
						},
						'required': ['context_type']
					}
				),

				types.Tool(
					name='list_execution_contexts',
					description='List all available execution contexts',
					inputSchema={
						'type': 'object',
						'properties': {}
					}
				)
			]

		@self.server.call_tool()
		async def handle_call_tool(name: str, arguments: Dict[str, Any] | None) -> List[types.TextContent]:
			"""Handle tool execution."""
			start_time = time.time()
			error_msg = None
			try:
				result = await self._execute_tool(name, arguments or {})
				return [types.TextContent(type='text', text=result)]
			except Exception as e:
				error_msg = str(e)
				logger.error(f'Tool execution failed: {e}', exc_info=True)
				return [types.TextContent(type='text', text=f'Error: {str(e)}')]

	async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
		"""Execute an ai-agent tool."""

		# Agent execution tools
		if tool_name == 'run_agent_task':
			return await self._run_agent_task(
				task=arguments['task'],
				max_steps=arguments.get('max_steps', 50),
				llm_provider=arguments.get('llm_provider', 'openai'),
				llm_model=arguments.get('llm_model', 'gpt-4'),
				execution_context_type=arguments.get('execution_context_type', 'generic'),
				settings=arguments.get('settings', {})
			)

		elif tool_name == 'step_agent':
			return await self._step_agent(arguments['agent_id'])

		# Agent control tools
		elif tool_name == 'get_agent_status':
			return await self._get_agent_status(arguments['agent_id'])

		elif tool_name == 'pause_agent':
			return await self._pause_agent(arguments['agent_id'])

		elif tool_name == 'resume_agent':
			return await self._resume_agent(arguments['agent_id'])

		elif tool_name == 'stop_agent':
			return await self._stop_agent(arguments['agent_id'])

		elif tool_name == 'list_agents':
			return await self._list_agents()

		# Action registry tools
		elif tool_name == 'list_actions':
			return await self._list_actions(
				category=arguments.get('category'),
				tags=arguments.get('tags')
			)

		elif tool_name == 'get_action_info':
			return await self._get_action_info(arguments['action_name'])

		elif tool_name == 'get_registry_stats':
			return await self._get_registry_stats()

		# Execution context tools
		elif tool_name == 'create_execution_context':
			return await self._create_execution_context(
				context_type=arguments['context_type'],
				config=arguments.get('config', {})
			)

		elif tool_name == 'list_execution_contexts':
			return await self._list_execution_contexts()

		return f'Unknown tool: {tool_name}'

	# Agent execution methods
	async def _run_agent_task(
		self,
		task: str,
		max_steps: int = 50,
		llm_provider: str = 'openai',
		llm_model: str = 'gpt-4',
		execution_context_type: str = 'generic',
		settings: Dict[str, Any] = None
	) -> str:
		"""Run an autonomous agent task."""
		logger.debug(f'Running agent task: {task}')

		try:
			# Create LLM
			llm_type = getattr(LLMType, llm_provider.upper(), None)
			if not llm_type:
				return f'Unsupported LLM provider: {llm_provider}'
			
			llm = create_llm(
				llm_type=llm_type,
				model=llm_model,
				api_key=os.getenv(f'{llm_provider.upper()}_API_KEY')
			)

			# Create execution context
			execution_context = await self._get_or_create_execution_context(execution_context_type)

			# Create agent settings
			agent_settings = AgentSettings(**(settings or {}))

			# Create file system
			if not self.file_system:
				self.file_system = FileSystem(base_dir=Path.home() / '.ai-agent-mcp')

			# Create agent
			agent = GenericAgent(
				task=task,
				llm=llm,
				execution_context=execution_context,
				settings=agent_settings,
				file_system=self.file_system
			)

			# Store agent for later reference
			self.agents[agent.id] = agent

			# Run agent
			history = await agent.run(max_steps=max_steps)

			# Format results
			results = []
			results.append(f'Agent {agent.id} completed task in {len(history.history)} steps')
			results.append(f'Success: {history.is_successful()}')

			# Get final result if available
			final_result = history.final_result()
			if final_result:
				results.append(f'\nFinal result:\n{final_result}')

			# Include any errors
			errors = history.errors()
			if errors:
				results.append(f'\nErrors encountered:\n{json.dumps(errors, indent=2)}')

			return '\n'.join(results)

		except Exception as e:
			logger.error(f'Agent task failed: {e}', exc_info=True)
			return f'Agent task failed: {str(e)}'

	async def _step_agent(self, agent_id: str) -> str:
		"""Execute a single step for an existing agent."""
		if agent_id not in self.agents:
			return f'Agent {agent_id} not found'

		agent = self.agents[agent_id]

		try:
			await agent.step()
			return f'Agent {agent_id} step completed. Current step: {agent.state.n_steps}'
		except Exception as e:
			logger.error(f'Agent step failed: {e}', exc_info=True)
			return f'Agent step failed: {str(e)}'

	# Agent control methods
	async def _get_agent_status(self, agent_id: str) -> str:
		"""Get the current status of an agent."""
		if agent_id not in self.agents:
			return f'Agent {agent_id} not found'

		agent = self.agents[agent_id]
		
		status = {
			'agent_id': agent_id,
			'task': agent.task,
			'started': agent.started,
			'current_step': agent.state.n_steps,
			'paused': agent.state.paused,
			'stopped': agent.state.stopped,
			'consecutive_failures': agent.state.consecutive_failures,
			'total_history_items': len(agent.history.history)
		}

		if agent.token_cost_service:
			status['total_cost'] = agent.token_cost_service.get_current_cost()

		return json.dumps(status, indent=2)

	async def _pause_agent(self, agent_id: str) -> str:
		"""Pause an active agent."""
		if agent_id not in self.agents:
			return f'Agent {agent_id} not found'

		agent = self.agents[agent_id]
		agent.pause()
		return f'Agent {agent_id} paused'

	async def _resume_agent(self, agent_id: str) -> str:
		"""Resume a paused agent."""
		if agent_id not in self.agents:
			return f'Agent {agent_id} not found'

		agent = self.agents[agent_id]
		agent.resume()
		return f'Agent {agent_id} resumed'

	async def _stop_agent(self, agent_id: str) -> str:
		"""Stop an active agent."""
		if agent_id not in self.agents:
			return f'Agent {agent_id} not found'

		agent = self.agents[agent_id]
		agent.stop()
		return f'Agent {agent_id} stopped'

	async def _list_agents(self) -> str:
		"""List all active agents."""
		if not self.agents:
			return 'No active agents'

		agents_info = []
		for agent_id, agent in self.agents.items():
			agents_info.append({
				'agent_id': agent_id,
				'task': agent.task[:100] + '...' if len(agent.task) > 100 else agent.task,
				'started': agent.started,
				'current_step': agent.state.n_steps,
				'paused': agent.state.paused,
				'stopped': agent.state.stopped
			})

		return json.dumps(agents_info, indent=2)

	# Action registry methods
	async def _list_actions(self, category: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
		"""List all available actions in the registry."""
		actions = list(self.action_registry.actions.values())

		# Apply filters if provided
		if category:
			actions = [
				action for action in actions
				if action.classification and action.classification.category == category
			]

		if tags:
			actions = [
				action for action in actions
				if action.classification and set(tags).issubset(
					{tag.value for tag in action.classification.tags}
				)
			]

		if not actions:
			return 'No actions found matching criteria'

		actions_info = []
		for action in actions:
			action_info = {
				'name': action.name,
				'description': action.description,
				'category': action.classification.category if action.classification else None,
				'execution_count': action.execution_count,
				'success_rate': f'{action.get_success_rate():.1f}%'
			}
			actions_info.append(action_info)

		return json.dumps(actions_info, indent=2)

	async def _get_action_info(self, action_name: str) -> str:
		"""Get detailed information about a specific action."""
		action = self.action_registry.get_action_by_name(action_name)
		
		if not action:
			return f'Action {action_name} not found'

		action_info = {
			'name': action.name,
			'description': action.description,
			'execution_count': action.execution_count,
			'success_count': action.success_count,
			'failure_count': action.failure_count,
			'success_rate': f'{action.get_success_rate():.1f}%',
			'average_execution_time': f'{action.get_average_execution_time():.3f}s'
		}

		if action.classification:
			action_info['classification'] = {
				'category': action.classification.category,
				'tags': [tag.value for tag in action.classification.tags],
				'capabilities': [cap.name for cap in action.classification.capabilities]
			}

		if action.documentation:
			action_info['documentation'] = {
				'stability': action.documentation.stability,
				'llm_description': action.documentation.llm_description
			}

		return json.dumps(action_info, indent=2)

	async def _get_registry_stats(self) -> str:
		"""Get comprehensive statistics about the action registry."""
		stats = self.action_registry.get_registry_statistics()
		return json.dumps(stats, indent=2)

	# Execution context methods
	async def _create_execution_context(self, context_type: str, config: Dict[str, Any]) -> str:
		"""Create a new execution context."""
		context_id = str(uuid.uuid4())
		
		if context_type == 'generic':
			execution_context = SimpleExecutionContext(
				file_system=self.file_system,
				registry=self.registry,
				**config
			)
		else:
			return f'Unsupported execution context type: {context_type}'

		self.execution_contexts[context_id] = execution_context
		return f'Created execution context {context_id} of type {context_type}'

	async def _list_execution_contexts(self) -> str:
		"""List all available execution contexts."""
		if not self.execution_contexts:
			return 'No execution contexts available'

		contexts_info = []
		for context_id, context in self.execution_contexts.items():
			contexts_info.append({
				'context_id': context_id,
				'type': type(context).__name__
			})

		return json.dumps(contexts_info, indent=2)

	async def _get_or_create_execution_context(self, context_type: str) -> ExecutionContext:
		"""Get or create an execution context of the specified type."""
		# For now, always create a new simple context
		# In future, we could cache and reuse contexts
		if context_type == 'generic':
			return SimpleExecutionContext(
				file_system=self.file_system,
				registry=self.registry
			)
		else:
			raise ValueError(f'Unsupported execution context type: {context_type}')

	async def run(self):
		"""Run the MCP server."""
		async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
			await self.server.run(
				read_stream,
				write_stream,
				InitializationOptions(
					server_name='ai-agent',
					server_version='1.0.0',
					capabilities=self.server.get_capabilities(
						notification_options=NotificationOptions(),
						experimental_capabilities={},
					),
				),
			)


async def main():
	"""Main entry point."""
	if not MCP_AVAILABLE:
		print('MCP SDK is required. Install with: pip install mcp', file=sys.stderr)
		sys.exit(1)

	server = AIAgentServer()
	try:
		await server.run()
	finally:
		pass  # Cleanup if needed


if __name__ == '__main__':
	asyncio.run(main())