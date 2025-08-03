from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

# Re-export ActionResult for backward compatibility
from ai_agent.common.models import ActionResult


class RegisteredAction(BaseModel):
	"""Model for a registered action"""
	model_config = ConfigDict(arbitrary_types_allowed=True)
	
	name: str
	description: str
	function: Callable
	param_model: type[BaseModel]
	
	# Generic filters for action availability
	context_filters: dict[str, Any] | None = None
	filter_function: Callable[[dict[str, Any]], bool] | None = None
	
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


class ActionRegistry(BaseModel):
	"""Model representing the action registry"""
	model_config = ConfigDict(extra='forbid')
	
	actions: dict[str, RegisteredAction] = {}
	action_models: dict[str, type[ActionModel]] = {}


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
	'ActionRegistry',
	'SpecialActionParameters',
]