"""Unified registry service consolidating controller and enhanced registry functionality."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import re
import time
from collections.abc import Callable
from inspect import Parameter, iscoroutinefunction, signature
from types import UnionType
from typing import Any, Generic, Optional, Type, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model

from ai_agent.actions.base import ActionProtocol, BaseAction, ActionMetadata
from ai_agent.registry.models import (
	ActionClassification,
	ActionDiscoveryFilter,
	ActionDocumentation,
	ActionModel,
	ActionRegistry,
	RegisteredAction,
	SpecialActionParameters,
)

Context = TypeVar('Context')

logger = logging.getLogger(__name__)


class Registry(Generic[Context]):
	"""Unified service for registering, managing, and executing actions with enhanced features"""

	def __init__(
		self,
		exclude_actions: list[str] | None = None,
		context_param_types: dict[str, type] | None = None,
	):
		self.registry = ActionRegistry()
		self.exclude_actions = exclude_actions if exclude_actions is not None else []
		self.context_param_types = context_param_types or {}
		
		# Enhanced management components - simplified for now
		# self.category_manager = CategoryManager()
		# self.doc_generator = ActionDocGenerator()

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

	def register_action(
		self,
		action: Type[BaseAction] | ActionProtocol,
		name: str | None = None,
		description: str | None = None,
		category: str | None = None,
		tags: set[str] | None = None,
		classification: ActionClassification | None = None,
		force_update: bool = False
	) -> RegisteredAction:
		"""Register an action with enhanced metadata"""
		# Determine action name
		action_name = name or getattr(action, 'name', None) or action.__class__.__name__
		
		# Check if already registered
		if action_name in self.registry.actions and not force_update:
			raise ValueError(f"Action '{action_name}' already registered. Use force_update=True to override.")
		
		# Extract or create parameter model
		if hasattr(action, 'parameter_model'):
			param_model = action.parameter_model
		else:
			# Create basic parameter model for non-BaseAction types
			param_model = type('ActionParameters', (BaseModel,), {})
		
		# Extract metadata
		if hasattr(action, 'metadata'):
			metadata = action.metadata
		else:
			metadata = ActionMetadata(
				name=action_name,
				description=description or "No description provided"
			)
		
		# Generate or use provided classification (simplified)
		if classification is None:
			classification = ActionClassification(category="general")
		
		# Generate documentation (simplified)
		documentation = ActionDocumentation(llm_description=description or metadata.description)
		
		# Create registered action
		registered_action = RegisteredAction(
			name=action_name,
			description=description or metadata.description,
			function=action.execute if hasattr(action, 'execute') else action,
			param_model=param_model,
			classification=classification,
			documentation=documentation
		)
		
		# Store in registry
		self.registry.actions[action_name] = registered_action
		
		# Create dynamic action model
		self._create_action_model(action_name, param_model)
		
		# Update registry metadata
		self.registry.last_updated = time.time()
		
		return registered_action

	def _create_action_model(self, action_name: str, param_model: type) -> None:
		"""Create a dynamic action model for the registered action"""
		# Create dynamic model class
		model_fields = {
			action_name: (param_model, ...)
		}
		
		action_model_class = type(
			f"{action_name}Model",
			(ActionModel,),
			{
				'model_fields': model_fields,
				'__annotations__': {action_name: param_model}
			}
		)
		
		self.registry.action_models[action_name] = action_model_class

	def unregister_action(self, action_name: str) -> bool:
		"""Unregister an action from the registry"""
		if action_name not in self.registry.actions:
			return False
		
		del self.registry.actions[action_name]
		
		if action_name in self.registry.action_models:
			del self.registry.action_models[action_name]
		
		self.registry.last_updated = time.time()
		
		return True

	async def execute_action(self, action: ActionModel, **kwargs) -> Any:
		"""Execute a registered action with parameter injection and performance tracking"""
		start_time = time.time()
		
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

		# Execute pre-hooks
		for hook in registered_action.pre_execute_hooks:
			if iscoroutinefunction(hook):
				await hook(params, **special_kwargs)
			else:
				hook(params, **special_kwargs)

		# Execute action
		success = True
		try:
			result = await registered_action.function(params=params, **special_kwargs)
		except Exception as e:
			success = False
			logger.error(f'Error executing action {action_name}: {e}')
			raise
		finally:
			# Update performance statistics
			execution_time = time.time() - start_time
			registered_action.update_execution_stats(success, execution_time)
			
			# Execute post-hooks
			for hook in registered_action.post_execute_hooks:
				if iscoroutinefunction(hook):
					await hook(params, result if success else None, **special_kwargs)
				else:
					hook(params, result if success else None, **special_kwargs)

		return result

	def _replace_sensitive_data(self, params: BaseModel, sensitive_data: str) -> BaseModel:
		"""Replace sensitive data placeholders in parameters"""
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

	def discover_actions(self, filter_criteria: ActionDiscoveryFilter) -> list[RegisteredAction]:
		"""Discover actions based on filter criteria"""
		candidates = list(self.registry.actions.values())
		
		# Apply filters
		if filter_criteria.name_pattern:
			pattern = re.compile(filter_criteria.name_pattern, re.IGNORECASE)
			candidates = [action for action in candidates if pattern.search(action.name)]
		
		if filter_criteria.category:
			candidates = [
				action for action in candidates
				if action.classification and action.classification.category == filter_criteria.category
			]
		
		if filter_criteria.tags:
			candidates = [
				action for action in candidates
				if action.classification and filter_criteria.tags.issubset(
					set(action.classification.tags)
				)
			]
		
		if filter_criteria.required_capabilities:
			candidates = [
				action for action in candidates
				if action.classification and any(
					cap_name in filter_criteria.required_capabilities
					for cap in action.classification.capabilities
					for cap_name in [cap.get('name', '')] if cap_name
				)
			]
		
		if filter_criteria.context_type:
			candidates = [
				action for action in candidates
				if action.classification and filter_criteria.context_type in action.classification.required_context_types
			]
		
		if filter_criteria.min_success_rate is not None:
			candidates = [
				action for action in candidates
				if action.get_success_rate() >= filter_criteria.min_success_rate
			]
		
		if filter_criteria.max_execution_time is not None:
			candidates = [
				action for action in candidates
				if action.get_average_execution_time() <= filter_criteria.max_execution_time
			]
		
		if filter_criteria.stability_level:
			candidates = [
				action for action in candidates
				if action.documentation and action.documentation.stability == filter_criteria.stability_level
			]
		
		if filter_criteria.min_usage_count is not None:
			candidates = [
				action for action in candidates
				if action.execution_count >= filter_criteria.min_usage_count
			]
		
		if filter_criteria.exclude_deprecated:
			candidates = [
				action for action in candidates
				if not (action.documentation and action.documentation.stability == "deprecated")
			]
		
		# Sort by relevance (usage count, success rate, then alphabetically)
		candidates.sort(key=lambda x: (-x.execution_count, -x.get_success_rate(), x.name))
		
		return candidates

	def get_action_by_name(self, action_name: str) -> RegisteredAction | None:
		"""Get a registered action by name"""
		return self.registry.actions.get(action_name)

	def get_actions_by_category(self, category: str) -> list[RegisteredAction]:
		"""Get all actions in a specific category"""
		return [
			action for action in self.registry.actions.values()
			if action.classification and action.classification.category == category
		]

	def get_action_suggestions(self, partial_name: str, limit: int = 10) -> list[RegisteredAction]:
		"""Get action suggestions based on partial name"""
		suggestions = []
		partial_lower = partial_name.lower()
		
		for action in self.registry.actions.values():
			if partial_lower in action.name.lower():
				suggestions.append(action)
		
		# Sort by relevance
		suggestions.sort(key=lambda x: (
			x.name.lower() != partial_lower,  # Exact matches first
			-x.execution_count,  # Then by usage
			x.name  # Finally alphabetically
		))
		
		return suggestions[:limit]

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

	def get_prompt_description(self) -> str:
		"""Get LLM-friendly description of all actions"""
		if not self.registry.actions:
			return "No actions available"
		
		descriptions = []
		for action in self.registry.actions.values():
			if action.documentation and action.documentation.llm_description:
				descriptions.append(f"{action.name}: {action.documentation.llm_description}")
			else:
				descriptions.append(action.prompt_description())
		
		return "\n".join(descriptions)

	def generate_documentation(self, format: str = "markdown") -> str:
		"""Generate comprehensive documentation for all actions"""
		if format == "markdown":
			return self._generate_markdown_docs()
		elif format == "json":
			return self._generate_json_docs()
		else:
			raise ValueError(f"Unsupported documentation format: {format}")

	def _generate_markdown_docs(self) -> str:
		"""Generate Markdown documentation"""
		lines = [
			"# Action Registry Documentation",
			"",
			f"Total Actions: {len(self.registry.actions)}",
			""
		]
		
		# Group by category
		categories = {}
		for action in self.registry.actions.values():
			category = action.classification.category if action.classification else "uncategorized"
			if category not in categories:
				categories[category] = []
			categories[category].append(action)
		
		# Generate documentation for each category
		for category, actions in sorted(categories.items()):
			lines.append(f"## {category.title()} Actions")
			lines.append("")
			
			for action in sorted(actions, key=lambda x: x.name):
				lines.append(f"### {action.name}")
				lines.append("")
				lines.append(action.description)
				lines.append("")
				
				if action.documentation:
					# Add parameters
					if action.documentation.parameters:
						lines.append("**Parameters:**")
						for param in action.documentation.parameters:
							req_text = "required" if param.required else "optional"
							lines.append(f"- `{param.name}` ({param.type}, {req_text}): {param.description}")
						lines.append("")
					
					# Add usage statistics
					if action.execution_count > 0:
						lines.append("**Statistics:**")
						lines.append(f"- Usage count: {action.execution_count}")
						lines.append(f"- Success rate: {action.get_success_rate():.1f}%")
						lines.append(f"- Average execution time: {action.get_average_execution_time():.3f}s")
						lines.append("")
				
				lines.append("---")
				lines.append("")
		
		return "\n".join(lines)

	def _generate_json_docs(self) -> str:
		"""Generate JSON documentation"""
		doc_data = {
			"registry_info": {
				"version": self.registry.registry_version,
				"total_actions": len(self.registry.actions),
				"last_updated": self.registry.last_updated
			},
			"actions": {}
		}
		
		for action_name, action in self.registry.actions.items():
			action_data = {
				"name": action.name,
				"description": action.description,
				"category": action.classification.category if action.classification else None,
				"tags": [tag.value for tag in action.classification.tags] if action.classification else [],
				"statistics": {
					"execution_count": action.execution_count,
					"success_rate": action.get_success_rate(),
					"average_execution_time": action.get_average_execution_time()
				}
			}
			
			if action.documentation:
				action_data["documentation"] = action.documentation.model_dump()
			
			doc_data["actions"][action_name] = action_data
		
		return json.dumps(doc_data, indent=2)

	def get_registry_statistics(self) -> dict[str, Any]:
		"""Get comprehensive registry statistics"""
		if not self.registry.actions:
			return {"message": "No actions registered"}
		
		# Basic stats
		total_actions = len(self.registry.actions)
		total_executions = sum(action.execution_count for action in self.registry.actions.values())
		total_execution_time = sum(action.total_execution_time for action in self.registry.actions.values())
		
		# Category breakdown
		categories = {}
		for action in self.registry.actions.values():
			category = action.classification.category if action.classification else "uncategorized"
			if category not in categories:
				categories[category] = {"count": 0, "executions": 0}
			categories[category]["count"] += 1
			categories[category]["executions"] += action.execution_count
		
		# Performance stats
		execution_times = [action.get_average_execution_time() for action in self.registry.actions.values() if action.execution_count > 0]
		success_rates = [action.get_success_rate() for action in self.registry.actions.values() if action.execution_count > 0]
		
		stats = {
			"registry_info": {
				"total_actions": total_actions,
				"total_executions": total_executions,
				"total_execution_time": total_execution_time,
				"registry_version": self.registry.registry_version,
				"last_updated": self.registry.last_updated
			},
			"categories": categories,
			"performance": {
				"average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
				"average_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
				"fastest_action": min(execution_times) if execution_times else 0,
				"slowest_action": max(execution_times) if execution_times else 0
			},
			"most_used_actions": sorted(
				[(action.name, action.execution_count) for action in self.registry.actions.values()],
				key=lambda x: x[1],
				reverse=True
			)[:10]
		}
		
		return stats

	def export_registry(self) -> dict[str, Any]:
		"""Export the entire registry for backup or migration"""
		return {
			"registry_metadata": {
				"version": self.registry.registry_version,
				"last_updated": self.registry.last_updated,
				"total_actions": len(self.registry.actions)
			},
			"actions": {
				name: {
					"name": action.name,
					"description": action.description,
					"classification": action.classification.model_dump() if action.classification else None,
					"documentation": action.documentation.model_dump() if action.documentation else None,
					"statistics": {
						"execution_count": action.execution_count,
						"total_execution_time": action.total_execution_time,
						"success_count": action.success_count,
						"failure_count": action.failure_count
					}
				}
				for name, action in self.registry.actions.items()
			}
		}

	def clear_registry(self) -> None:
		"""Clear all registered actions (for testing or reset)"""
		self.registry.actions.clear()
		self.registry.action_models.clear()
		self.registry.last_updated = time.time()

	@property
	def action_models(self) -> dict[str, type]:
		"""Get the action models from the internal registry"""
		return self.registry.action_models

	@property
	def actions(self) -> dict[str, RegisteredAction]:
		"""Get all registered actions"""
		return self.registry.actions