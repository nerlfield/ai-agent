from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from ai_agent.agent.service import ExecutionContext
from ai_agent.controller.registry.service import Registry
from ai_agent.filesystem import FileSystem
from ai_agent.llm.base import BaseChatModel
from ai_agent.prompts.context import ContextProvider
from ai_agent.registry.views import ActionModel, ActionResult


T = TypeVar('T', bound=BaseModel)


class SimpleContextState(BaseModel):
	"""Simple context state for basic agent operations"""
	
	name: str = 'simple_context'
	current_task: str | None = None
	environment: dict[str, Any] = Field(default_factory=dict)
	metadata: dict[str, Any] = Field(default_factory=dict)


class SimpleExecutionContext(ExecutionContext[dict[str, Any], SimpleContextState]):
	"""Simple execution context for basic agent operations"""
	
	def __init__(
		self,
		file_system: FileSystem | None = None,
		llm: BaseChatModel | None = None,
		registry: Registry | None = None,
		capabilities: set[str] | None = None,
	):
		self.file_system = file_system or FileSystem()
		self.llm = llm
		self.registry = registry or Registry()
		self.capabilities = capabilities or {'file_system', 'basic_operations'}
		self.state = SimpleContextState()
	
	async def get_current_state(self, cache_elements: bool = True, include_media: bool = True) -> SimpleContextState:
		"""Get current context state"""
		# Update state with current environment
		self.state.environment = {
			'working_directory': str(self.file_system.working_directory),
			'capabilities': list(self.capabilities),
			'has_llm': self.llm is not None,
		}
		return self.state
	
	async def execute_action(self, action: ActionModel, **kwargs) -> ActionResult:
		"""Execute an action through the registry"""
		# Add context parameters
		kwargs['file_system'] = self.file_system
		kwargs['llm'] = self.llm
		kwargs['context'] = self
		
		return await self.registry.execute_action(action, **kwargs)
	
	async def recover_from_error(self, error: Exception) -> None:
		"""Attempt to recover from execution errors"""
		# Simple recovery - just log and continue
		import logging
		logger = logging.getLogger(__name__)
		logger.error(f'Error in execution context: {error}')
	
	def get_action_registry(self) -> Registry:
		"""Get the action registry"""
		return self.registry
	
	def get_context_providers(self, **kwargs) -> list[ContextProvider]:
		"""Get context providers for prompt building"""
		from ai_agent.prompts.context import EnvironmentContextProvider
		
		providers = []
		
		# Add environment context
		if 'context_state' in kwargs:
			providers.append(SimpleEnvironmentProvider(kwargs['context_state']))
		
		return providers


class SimpleEnvironmentProvider(ContextProvider):
	"""Environment context provider for simple execution context"""
	
	def __init__(self, context_state: SimpleContextState):
		self.context_state = context_state
	
	def get_context_name(self) -> str:
		return "environment_state"
	
	def get_context_description(self) -> str:
		lines = [
			f"Environment: Simple Execution Context",
			f"Working Directory: {self.context_state.environment.get('working_directory', 'Unknown')}",
			f"Available Capabilities: {', '.join(self.context_state.environment.get('capabilities', []))}",
		]
		
		if self.context_state.current_task:
			lines.append(f"Current Task: {self.context_state.current_task}")
		
		if self.context_state.metadata:
			lines.append("Metadata:")
			for key, value in self.context_state.metadata.items():
				lines.append(f"  {key}: {value}")
		
		return '\n'.join(lines)
	
	def get_input_description(self) -> str:
		return "<environment_state>: Current environment state and available capabilities"