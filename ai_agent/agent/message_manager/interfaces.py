from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class ContextState(Protocol):
	"""Protocol for any context state that can be used with the message manager"""
	
	@property
	def url(self) -> str | None:
		"""Optional URL or identifier for the current context"""
		...
	
	@property
	def screenshot(self) -> bytes | None:
		"""Optional visual representation of the context"""
		...
	
	@property
	def media_data(self) -> dict[str, Any] | None:
		"""Optional media attachments for the context"""
		...


@runtime_checkable
class ContextPromptBuilder(Protocol):
	"""Protocol for building prompts from context state"""
	
	def __init__(
		self,
		context_state: ContextState,
		file_system: Any,
		agent_history_description: str | None,
		read_state_description: str | None,
		task: str,
		**kwargs: Any,
	):
		...
	
	def get_user_message(self, use_vision: bool = True) -> Any:
		"""Build user message from the context"""
		...


@runtime_checkable
class ExecutionResult(Protocol):
	"""Protocol for execution results"""
	
	@property
	def include_extracted_content_only_once(self) -> bool:
		...
	
	@property
	def extracted_content(self) -> str | None:
		...
	
	@property
	def long_term_memory(self) -> str | None:
		...
	
	@property
	def error(self) -> str | None:
		...


class AgentStepInfo(BaseModel):
	"""Information about the current agent step"""
	step_number: int
	max_steps: int | None = None
	
	def is_last_step(self) -> bool:
		return self.max_steps is not None and self.step_number >= self.max_steps - 1