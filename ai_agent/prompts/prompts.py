from __future__ import annotations

from typing import Any, Literal, Type

from ai_agent.agent.message_manager.interfaces import AgentStepInfo, ContextState
from ai_agent.llm.messages import UserMessage
from ai_agent.prompts.base import GenericSystemPrompt
from ai_agent.prompts.builder import GenericMessagePrompt
from ai_agent.prompts.context import (
	ContextProvider,
	FileSystem,
	HistoryContextProvider,
	ReadStateContextProvider,
	TaskContextProvider,
)


class AgentMessagePrompt:
	"""Main class for building agent messages with context"""
	
	vision_detail_level: Literal['auto', 'low', 'high']
	
	def __init__(
		self,
		context_state: ContextState,
		file_system: FileSystem,
		agent_history_description: str | None = None,
		read_state_description: str | None = None,
		task: str | None = None,
		include_attributes: list[str] | None = None,
		step_info: AgentStepInfo | None = None,
		page_filtered_actions: str | None = None,
		sensitive_data: str | None = None,
		available_file_paths: list[str] | None = None,
		media_items: list[str | bytes] | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		custom_context_providers: list[ContextProvider] | None = None,
		**kwargs: Any,
	):
		self.context_state = context_state
		self.file_system = file_system
		self.agent_history_description = agent_history_description
		self.read_state_description = read_state_description
		self.task = task
		self.include_attributes = include_attributes
		self.step_info = step_info
		self.page_filtered_actions = page_filtered_actions
		self.sensitive_data = sensitive_data
		self.available_file_paths = available_file_paths
		self.media_items = media_items or []
		self.vision_detail_level = vision_detail_level
		self.custom_context_providers = custom_context_providers or []
		self.kwargs = kwargs
		
		# Extract media from context state if available
		if hasattr(context_state, 'screenshot') and context_state.screenshot:
			self.media_items.append(context_state.screenshot)
		elif hasattr(context_state, 'media_data') and context_state.media_data:
			self.media_items.extend(context_state.media_data.values())
	
	def _build_context_providers(self) -> list[ContextProvider]:
		"""Build the list of context providers"""
		providers = []
		
		# Add history context
		if self.agent_history_description:
			providers.append(HistoryContextProvider(self.agent_history_description))
		
		# Add task/agent state context
		providers.append(
			TaskContextProvider(
				task=self.task,
				file_system=self.file_system,
				step_info=self.step_info,
				sensitive_data=self.sensitive_data,
				available_file_paths=self.available_file_paths,
			)
		)
		
		# Add custom context providers (e.g., browser state, API state, etc.)
		providers.extend(self.custom_context_providers)
		
		# Add read state context if available
		if self.read_state_description:
			providers.append(ReadStateContextProvider(self.read_state_description))
		
		return providers
	
	def get_user_message(self, use_vision: bool = True) -> UserMessage:
		"""Build and return the user message"""
		context_providers = self._build_context_providers()
		
		prompt_builder = GenericMessagePrompt(
			context_providers=context_providers,
			media_items=self.media_items,
			page_filtered_actions=self.page_filtered_actions,
			vision_detail_level=self.vision_detail_level,
			**self.kwargs,
		)
		
		return prompt_builder.get_user_message(use_vision)


# Re-export commonly used classes
__all__ = [
	'GenericSystemPrompt',
	'AgentMessagePrompt',
	'ContextProvider',
	'TaskContextProvider',
	'HistoryContextProvider',
	'ReadStateContextProvider',
]