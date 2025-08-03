from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from inspect import Parameter, iscoroutinefunction, signature
from types import UnionType
from typing import Any, Generic, Optional, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model

from ai_agent.registry.views import (
	ActionModel,
	ActionRegistry,
	RegisteredAction,
	SpecialActionParameters,
)

Context = TypeVar('Context')

logger = logging.getLogger(__name__)


class Registry(Generic[Context]):
	"""Service for registering and managing actions"""

	def __init__(
		self,
		exclude_actions: list[str] | None = None,
		context_param_types: dict[str, type] | None = None,
	):
		self.registry = ActionRegistry()
		self.exclude_actions = exclude_actions if exclude_actions is not None else []
		self.context_param_types = context_param_types or {}

	def _get_special_param_types(self) -> dict[str, type | UnionType | None]:
		"""Get the expected types for special parameters"""
		# Base special parameters that are always available
		base_params = {
			'context': None,  # Context is a TypeVar, so we can't validate type
			'llm': None,  # Generic LLM interface
			'file_system': None,  # Generic file system interface
			'task_context': dict,  # Task-specific context
			'sensitive_data': str,  # Sensitive data string
		}
		
		# Merge with user-provided context parameters
		return {**base_params, **self.context_param_types}

	def _normalize_action_function_signature(
		self,
		func: Callable,
		description: str,
		param_model: type[BaseModel] | None = None,
	) -> tuple[Callable, type[BaseModel]]:
		"""
		Normalize action function to accept only kwargs.

		Returns:
			- Normalized function that accepts (*_, params: ParamModel, **special_params)
			- The param model to use for registration
		"""
		sig = signature(func)
		parameters = list(sig.parameters.values())
		special_param_types = self._get_special_param_types()
		special_param_names = set(special_param_types.keys())

		# Step 1: Validate no **kwargs in original function signature
		for param in parameters:
			if param.kind == Parameter.VAR_KEYWORD:
				raise ValueError(
					f"Action '{func.__name__}' has **{param.name} which is not allowed. "
					f'Actions must have explicit positional parameters only.'
				)

		# Step 2: Separate special and action parameters
		action_params = []
		special_params = []
		param_model_provided = param_model is not None

		for i, param in enumerate(parameters):
			# Check if this is a Type 1 pattern (first param is BaseModel)
			if i == 0 and param_model_provided and param.name not in special_param_names:
				# This is Type 1 pattern - skip the params argument
				continue

			if param.name in special_param_names:
				# Validate special parameter type
				expected_type = special_param_types.get(param.name)
				if param.annotation != Parameter.empty and expected_type is not None:
					# Handle Optional types - normalize both sides
					param_type = param.annotation
					origin = get_origin(param_type)
					if origin is Union:
						args = get_args(param_type)
						# Find non-None type
						param_type = next((arg for arg in args if arg is not type(None)), param_type)

					# Check if types are compatible
					types_compatible = (
						param_type == expected_type
						or (
							inspect.isclass(param_type)
							and inspect.isclass(expected_type)
							and issubclass(param_type, expected_type)
						)
						or
						# Handle list[T] vs list comparison
						(expected_type is list and (param_type is list or get_origin(param_type) is list))
					)

					if not types_compatible:
						expected_type_name = getattr(expected_type, '__name__', str(expected_type))
						param_type_name = getattr(param_type, '__name__', str(param_type))
						raise ValueError(
							f"Action '{func.__name__}' parameter '{param.name}: {param_type_name}' "
							f"conflicts with special argument injected by controller: '{param.name}: {expected_type_name}'"
						)
				special_params.append(param)
			else:
				action_params.append(param)

		# Step 3: Create or validate param model
		if not param_model_provided:
			# Type 2: Generate param model from action params
			if action_params:
				params_dict = {}
				for param in action_params:
					annotation = param.annotation if param.annotation != Parameter.empty else str
					default = ... if param.default == Parameter.empty else param.default
					params_dict[param.name] = (annotation, default)

				param_model = create_model(f'{func.__name__}_Params', __base__=ActionModel, **params_dict)
			else:
				# No action params, create empty model
				param_model = create_model(
					f'{func.__name__}_Params',
					__base__=ActionModel,
				)
		assert param_model is not None, f'param_model is None for {func.__name__}'

		# Step 4: Create normalized wrapper function
		@functools.wraps(func)
		async def normalized_wrapper(*args, params: BaseModel | None = None, **kwargs):
			"""Normalized action that only accepts kwargs"""
			# Validate no positional args
			if args:
				raise TypeError(f'{func.__name__}() does not accept positional arguments, only keyword arguments are allowed')

			# Prepare arguments for original function
			call_args = []
			call_kwargs = {}

			# Handle Type 1 pattern (first arg is the param model)
			if param_model_provided and parameters and parameters[0].name not in special_param_names:
				if params is None:
					raise ValueError(f"{func.__name__}() missing required 'params' argument")
				# For Type 1, we'll use the params object as first argument
				pass
			else:
				# Type 2 pattern - need to unpack params
				# If params is None, try to create it from kwargs
				if params is None and action_params:
					# Extract action params from kwargs
					action_kwargs = {}
					for param in action_params:
						if param.name in kwargs:
							action_kwargs[param.name] = kwargs[param.name]
					if action_kwargs:
						# Use the param_model which has the correct types defined
						params = param_model(**action_kwargs)

			# Build call_args by iterating through original function parameters in order
			params_dict = params.model_dump() if params is not None else {}

			for i, param in enumerate(parameters):
				# Skip first param for Type 1 pattern (it's the model itself)
				if param_model_provided and i == 0 and param.name not in special_param_names:
					call_args.append(params)
				elif param.name in special_param_names:
					# This is a special parameter
					if param.name in kwargs:
						value = kwargs[param.name]
						# Check if required special param is None
						if value is None and param.default == Parameter.empty:
							raise ValueError(f"{func.__name__}() missing required special parameter '{param.name}'")
						call_args.append(value)
					elif param.default != Parameter.empty:
						call_args.append(param.default)
					else:
						# Special param is required but not provided
						raise ValueError(f"{func.__name__}() missing required special parameter '{param.name}'")
				else:
					# This is an action parameter
					if param.name in params_dict:
						call_args.append(params_dict[param.name])
					elif param.default != Parameter.empty:
						call_args.append(param.default)
					else:
						raise ValueError(f"{func.__name__}() missing required parameter '{param.name}'")

			# Call original function with positional args
			if iscoroutinefunction(func):
				return await func(*call_args)
			else:
				return await asyncio.to_thread(func, *call_args)

		# Update wrapper signature to be kwargs-only
		new_params = [Parameter('params', Parameter.KEYWORD_ONLY, default=None, annotation=Optional[param_model])]

		# Add special params as keyword-only
		for sp in special_params:
			new_params.append(Parameter(sp.name, Parameter.KEYWORD_ONLY, default=sp.default, annotation=sp.annotation))

		# Add **kwargs to accept and ignore extra params
		new_params.append(Parameter('kwargs', Parameter.VAR_KEYWORD))

		normalized_wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]

		return normalized_wrapper, param_model

	def action(
		self,
		description: str,
		param_model: type[BaseModel] | None = None,
		context_filters: dict[str, Any] | None = None,
		filter_function: Callable[[dict[str, Any]], bool] | None = None,
	):
		"""
		Decorator for registering actions.
		
		Args:
			description: Human-readable description of the action
			param_model: Optional Pydantic model for action parameters
			context_filters: Dict of context attributes that must match for action availability
			filter_function: Custom function to determine if action is available
		"""
		def decorator(func: Callable):
			# Skip registration if action is in exclude_actions
			if func.__name__ in self.exclude_actions:
				return func

			# Normalize the function signature
			normalized_func, actual_param_model = self._normalize_action_function_signature(func, description, param_model)

			action = RegisteredAction(
				name=func.__name__,
				description=description,
				function=normalized_func,
				param_model=actual_param_model,
				context_filters=context_filters,
				filter_function=filter_function,
			)
			self.registry.actions[func.__name__] = action
			return func

		return decorator

	async def execute_action(self, action: ActionModel, **kwargs) -> Any:
		"""Execute a registered action with parameter injection"""
		# Get action name from the model
		action_dict = action.model_dump(exclude_unset=True)
		if not action_dict:
			raise ValueError('Action model is empty')

		action_name = next(iter(action_dict.keys()))
		action_params = action_dict[action_name]

		# Find registered action
		registered_action = self.registry.actions.get(action_name)
		if not registered_action:
			raise ValueError(f"Action '{action_name}' not found in registry")

		# Create params model
		try:
			params = registered_action.param_model(**action_params) if action_params else None
		except Exception as e:
			raise ValueError(f'Failed to validate parameters for action {action_name}: {e}')

		# Handle sensitive data replacement if provided
		if 'sensitive_data' in kwargs and kwargs['sensitive_data'] and params:
			params = self._replace_sensitive_data(params, kwargs['sensitive_data'])

		# Prepare special parameters
		special_kwargs = {}
		for key, value in kwargs.items():
			if key in self._get_special_param_types():
				special_kwargs[key] = value

		# Execute action
		try:
			return await registered_action.function(params=params, **special_kwargs)
		except Exception as e:
			logger.error(f'Error executing action {action_name}: {e}')
			raise

	def _replace_sensitive_data(self, params: BaseModel, sensitive_data: str) -> BaseModel:
		"""Replace sensitive data placeholders in parameters"""
		# This is a simplified version - can be extended based on needs
		import re
		
		def replace_in_value(value: Any) -> Any:
			if isinstance(value, str):
				# Find all <secret>...</secret> patterns
				pattern = r'<secret>(.*?)</secret>'
				matches = re.findall(pattern, value)
				for match in matches:
					# In a real implementation, look up the actual secret value
					# For now, just replace with a placeholder
					value = value.replace(f'<secret>{match}</secret>', f'[REDACTED:{match}]')
			return value
		
		# Replace in all string fields
		data = params.model_dump()
		for key, value in data.items():
			data[key] = replace_in_value(value)
		
		return params.__class__(**data)

	def get_available_actions(self, context: dict[str, Any] | None = None) -> list[RegisteredAction]:
		"""Get all actions available in the current context"""
		if context is None:
			return list(self.registry.actions.values())
		
		available = []
		for action in self.registry.actions.values():
			# Check context filters
			if action.context_filters:
				if not all(context.get(k) == v for k, v in action.context_filters.items()):
					continue
			
			# Check filter function
			if action.filter_function:
				if not action.filter_function(context):
					continue
			
			available.append(action)
		
		return available

	def create_action_model(
		self,
		action_names: list[str] | None = None,
		context: dict[str, Any] | None = None,
	) -> type[BaseModel]:
		"""Create a union model of all available actions for LLM tool calling"""
		# Get available actions
		available_actions = self.get_available_actions(context)
		
		# Filter by action names if provided
		if action_names:
			available_actions = [a for a in available_actions if a.name in action_names]
		
		if not available_actions:
			raise ValueError('No actions available in current context')
		
		# Create individual action models
		action_models = {}
		for action in available_actions:
			# Create a model that wraps the param model
			class ActionWrapper(BaseModel):
				model_config = ConfigDict(extra='ignore')
			
			# Add the action as a field
			ActionWrapper.__annotations__ = {action.name: action.param_model}
			ActionWrapper.__fields__ = {
				action.name: Field(description=action.description)
			}
			
			action_models[action.name] = ActionWrapper
		
		# Create union of all action models
		if len(action_models) == 1:
			return next(iter(action_models.values()))
		else:
			return Union[tuple(action_models.values())]  # type: ignore

	@property
	def action_models(self) -> dict[str, type]:
		"""Get the action models from the internal registry"""
		return self.registry.action_models
	
	def get_prompt_description(self) -> str:
		"""Get a description of all actions for the prompt"""
		descriptions = []
		for action in self.registry.actions.values():
			descriptions.append(action.prompt_description())
		return '\n'.join(descriptions)