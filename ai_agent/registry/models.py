"""Unified registry models consolidating all registry-related data structures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Re-export ActionResult for backward compatibility
from ai_agent.common.models import ActionResult
from ai_agent.actions.base import ActionMetadata

# Simplified classification classes
class ActionClassification(BaseModel):
	"""Simple action classification model"""
	model_config = ConfigDict(extra='forbid')
	
	category: str = "general"
	tags: list[str] = Field(default_factory=list)
	usage_count: int = 0
	success_rate: float = 0.0
	average_duration: float = 0.0
	required_context_types: list[str] = Field(default_factory=list)
	capabilities: list[dict] = Field(default_factory=list)

class ActionDocumentation(BaseModel):
	"""Simple action documentation model"""
	model_config = ConfigDict(extra='forbid')
	
	llm_description: str | None = None
	stability: str = "stable"
	parameters: list[dict] = Field(default_factory=list)


class RegisteredAction(BaseModel):
	"""Unified model for a registered action with optional enhanced features"""
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
	
	# Core properties
	name: str
	description: str
	function: Callable
	param_model: type[BaseModel]
	
	# Context filtering
	context_filters: dict[str, Any] | None = None
	filter_function: Callable[[dict[str, Any]], bool] | None = None
	
	# Enhanced metadata (optional)
	classification: ActionClassification | None = None
	documentation: ActionDocumentation | None = None
	
	# Lifecycle hooks (optional)
	pre_execute_hooks: list[Callable] = Field(default_factory=list)
	post_execute_hooks: list[Callable] = Field(default_factory=list)
	
	# Performance tracking (optional)
	execution_count: int = 0
	total_execution_time: float = 0.0
	success_count: int = 0
	failure_count: int = 0
	
	# Dynamic loading (optional)
	module_path: str | None = None
	lazy_loaded: bool = False
	
	def prompt_description(self) -> str:
		"""Get a description of the action for the prompt"""
		skip_keys = ['title']
		s = f'{self.description}: \n'
		s += '{' + str(self.name) + ': '
		s += str(
			{
				k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k not in skip_keys}
				for k, v in self.param_model.model_json_schema()['properties'].items()
			}
		)
		s += '}'
		return s
	
	def get_average_execution_time(self) -> float:
		"""Get average execution time"""
		if self.execution_count == 0:
			return 0.0
		return self.total_execution_time / self.execution_count
	
	def get_success_rate(self) -> float:
		"""Get success rate as a percentage"""
		if self.execution_count == 0:
			return 0.0
		return (self.success_count / self.execution_count) * 100
	
	def update_execution_stats(self, success: bool, execution_time: float) -> None:
		"""Update execution statistics"""
		self.execution_count += 1
		self.total_execution_time += execution_time
		
		if success:
			self.success_count += 1
		else:
			self.failure_count += 1
		
		# Update classification statistics if available
		if self.classification:
			self.classification.usage_count = self.execution_count
			self.classification.success_rate = self.get_success_rate() / 100
			self.classification.average_duration = self.get_average_execution_time()


class ActionModel(BaseModel):
	"""Base model for dynamically created action models"""
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='ignore')
	
	def get_index(self) -> int | None:
		"""Get the index of the action if it has one"""
		params = self.model_dump(exclude_unset=True).values()
		if not params:
			return None
		for param in params:
			if param is not None and isinstance(param, dict) and 'index' in param:
				return param['index']
		return None
	
	def set_index(self, index: int):
		"""Set the index of the action if it supports indexing"""
		action_data = self.model_dump(exclude_unset=True)
		action_name = next(iter(action_data.keys()))
		action_params = getattr(self, action_name)
		
		if hasattr(action_params, 'index'):
			action_params.index = index


class ActionDiscoveryFilter(BaseModel):
	"""Filter criteria for discovering actions"""
	model_config = ConfigDict(extra='forbid')
	
	# Basic filters
	name_pattern: str | None = None
	category: str | None = None
	tags: set[str] = Field(default_factory=set)
	
	# Capability filters
	required_capabilities: set[str] = Field(default_factory=set)
	context_type: str | None = None
	
	# Quality filters
	min_success_rate: float | None = None
	max_execution_time: float | None = None
	stability_level: str | None = None  # experimental, beta, stable
	
	# Usage filters
	min_usage_count: int | None = None
	exclude_deprecated: bool = True


class ActionRegistry(BaseModel):
	"""Model representing the action registry"""
	model_config = ConfigDict(extra='forbid')
	
	actions: dict[str, RegisteredAction] = Field(default_factory=dict)
	action_models: dict[str, type] = Field(default_factory=dict)
	
	# Registry metadata
	registry_version: str = "1.0.0"
	last_updated: float | None = None


class SpecialActionParameters:
	"""Special parameters that can be injected into actions"""
	
	context: Any | None = None
	llm: Any | None = None
	file_system: Any | None = None
	sensitive_data: str | None = None
	task_context: dict[str, Any] | None = None


__all__ = [
	'ActionResult',
	'RegisteredAction',
	'ActionModel', 
	'ActionDiscoveryFilter',
	'ActionRegistry',
	'SpecialActionParameters',
]