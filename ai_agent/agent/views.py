from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, model_validator
from uuid_extensions import uuid7str

from ai_agent.common.models import ActionResult
from ai_agent.llm.base import BaseChatModel
from ai_agent.registry.views import ActionModel
from ai_agent.tokens.views import UsageSummary

if TYPE_CHECKING:
	from ai_agent.agent.message_manager.views import MessageManagerState

T = TypeVar('T')


# Import the real MessageManagerState from message_manager module
# This will be imported later to avoid circular imports


class FileSystemState(BaseModel):
	"""Placeholder for file system state - can be implemented as needed"""
	model_config = ConfigDict(extra='forbid')
	
	current_dir: Path = Field(default_factory=Path.cwd)
	files: dict[str, Any] = Field(default_factory=dict)


class ContextState(BaseModel, Generic[T]):
	"""Generic context state that can hold any type of execution context"""
	model_config = ConfigDict(extra='allow')
	
	id: str = Field(default_factory=uuid7str)
	name: str
	type: str
	data: T
	metadata: dict[str, Any] = Field(default_factory=dict)
	attachments: dict[str, Any] = Field(default_factory=dict)  # screenshots, files, logs, etc.
	interaction_map: dict[int, Any] = Field(default_factory=dict)  # replaces SelectorMap


class ExecutionContext(BaseModel):
	"""Holds the current execution context for the agent"""
	model_config = ConfigDict(extra='forbid')
	
	states: dict[str, ContextState] = Field(default_factory=dict)
	active_context: str | None = None
	
	def get_active_state(self) -> ContextState | None:
		if self.active_context and self.active_context in self.states:
			return self.states[self.active_context]
		return None


class AgentSettings(BaseModel):
	"""Configuration options for the Agent"""
	model_config = ConfigDict(extra='forbid')
	
	# Core agent settings
	max_failures: int = 3
	retry_delay: int = 10
	validate_output: bool = False
	message_context: str | None = None
	override_system_message: str | None = None
	extend_system_message: str | None = None
	max_actions_per_step: int = 10
	use_thinking: bool = True
	use_vision: bool = True
	flash_mode: bool = False
	max_history_items: int = 40
	
	# Execution settings
	save_conversation_path: str | Path | None = None
	save_conversation_path_encoding: str | None = 'utf-8'
	calculate_cost: bool = False
	include_tool_call_examples: bool = False
	llm_timeout: int = 60
	step_timeout: int = 180
	
	# LLM settings
	extraction_llm: BaseChatModel | None = None
	planner_llm: BaseChatModel | None = None
	planner_interval: int = 1
	is_planner_reasoning: bool = False
	extend_planner_system_message: str | None = None
	
	# Generic capabilities (replaces browser-specific settings)
	enabled_capabilities: set[str] = Field(default_factory=lambda: {'vision', 'planning'})
	capability_configs: dict[str, Any] = Field(default_factory=dict)
	
	# Tool/action specific configs
	tool_configs: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
	"""Holds all state information for an Agent"""
	model_config = ConfigDict(extra='forbid')
	
	agent_id: str = Field(default_factory=uuid7str)
	n_steps: int = 1
	consecutive_failures: int = 0
	last_result: list[ActionResult] | None = None
	last_plan: str | None = None
	last_model_output: AgentOutput | None = None
	paused: bool = False
	stopped: bool = False
	
	message_manager_state: Any = Field(default=None)
	file_system_state: FileSystemState | None = None
	
	# Generic context management
	execution_context: ExecutionContext = Field(default_factory=ExecutionContext)
	context_states: dict[str, Any] = Field(default_factory=dict)


@dataclass
class AgentStepInfo:
	step_number: int
	max_steps: int
	
	def is_last_step(self) -> bool:
		return self.step_number >= self.max_steps - 1


# ActionResult is now imported from common.models




class AgentBrain(BaseModel):
	"""The brain of the agent - thinking and planning components"""
	model_config = ConfigDict(extra='forbid')
	
	thinking: str | None = None
	evaluation_previous_goal: str | None = None
	memory: str
	next_goal: str


class AgentOutput(BaseModel):
	"""The output of the agent model"""
	model_config = ConfigDict(
		extra='ignore',
		arbitrary_types_allowed=True,
	)
	
	is_done: bool
	action: list[ActionModel] = Field(default_factory=list)
	extracted_content: str | None = None
	
	current_state: AgentBrain
	
	@model_validator(mode='before')
	@classmethod
	def validate_model(cls, data: Any) -> Any:
		if isinstance(data, dict):
			if 'actions' in data and 'action' not in data:
				data['action'] = data.pop('actions')
			
			if 'action' in data and data['action'] is None:
				data['action'] = []
			elif 'action' in data and not isinstance(data['action'], list):
				data['action'] = [data['action']]
			
			# Convert dict-based actions to ActionModel instances
			if 'action' in data and isinstance(data['action'], list):
				converted_actions = []
				for action_item in data['action']:
					if isinstance(action_item, dict):
						# Convert dict format like {"write_file": {"path": "...", "content": "..."}}
						# to proper ActionModel instance
						converted_action = cls._dict_to_action_model(action_item)
						converted_actions.append(converted_action)
					else:
						converted_actions.append(action_item)
				data['action'] = converted_actions
		
		return data
	
	@classmethod
	def _dict_to_action_model(cls, action_dict: dict) -> ActionModel:
		"""Convert a dictionary action to an ActionModel instance"""
		if not action_dict:
			return ActionModel()
		
		# Get the action name (first key) and parameters
		action_name = next(iter(action_dict.keys()))
		action_params = action_dict[action_name]
		
		# Create a dynamic ActionModel with the action as an attribute
		from pydantic import create_model
		
		# Create the parameter model first
		param_model_name = f"{action_name}_params"
		param_fields = {}
		if isinstance(action_params, dict):
			for key, value in action_params.items():
				param_fields[key] = (type(value), value)
		
		ParamModel = create_model(param_model_name, **param_fields)
		param_instance = ParamModel(**action_params) if action_params else ParamModel()
		
		# Create the action model
		ActionModelClass = create_model(
			f"{action_name}_ActionModel",
			__base__=ActionModel,
			**{action_name: (type(param_instance), param_instance)}
		)
		
		return ActionModelClass(**{action_name: param_instance})


class AgentStructuredOutput(BaseModel, Generic[T]):
	"""Structured output from agent"""
	model_config = ConfigDict(extra='forbid')
	
	is_done: bool
	structured_output: T
	current_state: AgentBrain


class AgentError:
	"""Static methods for error formatting"""
	
	@staticmethod
	def format_error(error: Exception, action: ActionModel | None = None, include_trace: bool = True) -> str:
		parts = []
		
		if action:
			# Get the action name from the ActionModel
			action_data = action.model_dump(exclude_unset=True)
			if action_data:
				action_name = next(iter(action_data.keys()), 'unknown')
				parts.append(f"Action: {action_name}")
				if action_name in action_data and action_data[action_name]:
					parts.append(f"Parameters: {json.dumps(action_data[action_name], indent=2)}")
			else:
				parts.append(f"Action: {type(action).__name__}")
		
		parts.append(f"Error: {type(error).__name__}: {str(error)}")
		
		if include_trace:
			parts.append(f"Traceback:\n{traceback.format_exc()}")
		
		return "\n".join(parts)
	
	@staticmethod
	def format_rate_limit_error(error: Exception) -> str:
		return f"Rate limit reached. Please wait before retrying. Error: {str(error)}"
	
	@staticmethod
	def format_validation_error(error: ValidationError) -> str:
		errors = []
		for err in error.errors():
			loc = " -> ".join(str(l) for l in err['loc'])
			errors.append(f"- {loc}: {err['msg']}")
		return "Validation errors:\n" + "\n".join(errors)


class StepMetadata(BaseModel):
	"""Metadata for a single agent step"""
	model_config = ConfigDict(extra='forbid')
	
	step_number: int
	timestamp: float
	duration: float | None = None
	token_usage: UsageSummary | None = None
	error: str | None = None
	retry_count: int = 0


class ContextStateHistory(BaseModel, Generic[T]):
	"""Generic state history that can work with any context type"""
	model_config = ConfigDict(extra='allow')
	
	context_type: str
	context_data: T
	attachments: dict[str, Any] = Field(default_factory=dict)
	interaction_history: list[dict[str, Any]] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class AgentHistory(BaseModel):
	"""Single history item for agent execution"""
	model_config = ConfigDict(extra='forbid')
	
	id: str = Field(default_factory=uuid7str)
	model_output: AgentOutput | None
	result: list[ActionResult]
	state: ContextStateHistory
	metadata: StepMetadata
	
	def get_interacted_elements(self) -> list[dict[str, Any]]:
		"""Get all interacted elements from the context state"""
		return self.state.interaction_history


class AgentHistoryList(BaseModel):
	"""List of agent history items with analysis methods"""
	model_config = ConfigDict(extra='forbid')
	
	history: list[AgentHistory] = Field(default_factory=list)
	
	def __len__(self) -> int:
		return len(self.history)
	
	def __getitem__(self, index: int) -> AgentHistory:
		return self.history[index]
	
	def __iter__(self):
		return iter(self.history)
	
	def append(self, item: AgentHistory) -> None:
		self.history.append(item)
	
	@property
	def last(self) -> AgentHistory | None:
		return self.history[-1] if self.history else None
	
	def get_errors(self) -> list[tuple[int, str]]:
		"""Get all errors with their step numbers"""
		errors = []
		for i, item in enumerate(self.history):
			if item.metadata.error:
				errors.append((i, item.metadata.error))
			for result in item.result:
				for error in result.errors:
					errors.append((i, error))
		return errors
	
	def get_total_duration(self) -> float:
		"""Get total duration of all steps"""
		return sum(
			item.metadata.duration or 0
			for item in self.history
			if item.metadata.duration
		)
	
	def get_attachments(self, attachment_type: str | None = None) -> list[dict[str, Any]]:
		"""Get all attachments of a specific type or all attachments"""
		attachments = []
		for item in self.history:
			if attachment_type:
				if attachment_type in item.state.attachments:
					attachments.append({
						'step': item.metadata.step_number,
						'data': item.state.attachments[attachment_type]
					})
			else:
				for att_type, att_data in item.state.attachments.items():
					attachments.append({
						'step': item.metadata.step_number,
						'type': att_type,
						'data': att_data
					})
		return attachments
	
	def model_dump(self, **kwargs) -> dict[str, Any]:
		data = super().model_dump(**kwargs)
		data['len'] = len(self.history)
		return data
	
	@classmethod
	def model_validate(cls, obj: Any) -> AgentHistoryList:
		if isinstance(obj, dict):
			if 'len' in obj:
				obj.pop('len')
		return super().model_validate(obj)


# Rebuild models to resolve forward references
AgentState.model_rebuild()
AgentOutput.model_rebuild()
AgentHistory.model_rebuild()
AgentHistoryList.model_rebuild()