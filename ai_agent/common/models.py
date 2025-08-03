"""Common models used across the ai_agent package"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActionResult(BaseModel):
	"""Result of an action execution"""
	model_config = ConfigDict(extra='ignore')
	
	success: bool = True
	is_done: bool = False
	errors: list[str] = Field(default_factory=list)
	
	# Generic attachments instead of specific file types
	attachments: dict[str, Any] = Field(default_factory=dict)
	output_data: Any | None = None
	
	# Memory and state management
	include_in_memory: bool = True
	extracted_content: str | None = None
	context_updates: dict[str, Any] = Field(default_factory=dict)
	
	# Action metadata
	action_type: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	
	# Additional fields for compatibility
	value: Any | None = None
	long_term_memory: str | None = None
	include_extracted_content_only_once: bool = False
	
	@property
	def error(self) -> str | None:
		"""Get the first error message if any"""
		return self.errors[0] if self.errors else None


class ExecutionResult(BaseModel):
	"""Extended result from execution context"""
	model_config = ConfigDict(extra='forbid')
	
	error: str | None = None
	extracted_content: str | None = None
	include_in_memory: bool = True
	value: Any | None = None
	
	# Additional fields
	long_term_memory: str | None = None
	include_extracted_content_only_once: bool = False