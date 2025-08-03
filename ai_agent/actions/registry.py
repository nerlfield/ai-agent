"""
Enhanced action registry system that extends the base registry with comprehensive features.

This module provides advanced registry functionality including:
- Action discovery and filtering
- Dynamic action loading
- Action lifecycle management
- Integration with categorization and documentation systems
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Type

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.actions.base import ActionProtocol, BaseAction, ActionMetadata
from ai_agent.actions.categories import ActionClassification, CategoryManager
from ai_agent.actions.documentation import ActionDocGenerator, ActionDocumentation
from ai_agent.registry.views import RegisteredAction as BaseRegisteredAction
from ai_agent.registry.views import ActionRegistry as BaseActionRegistry


class RegisteredAction(BaseRegisteredAction):
	"""Enhanced registered action with additional metadata and capabilities"""
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
	
	# Enhanced metadata
	classification: ActionClassification | None = None
	documentation: ActionDocumentation | None = None
	
	# Lifecycle hooks
	pre_execute_hooks: list[Callable] = Field(default_factory=list)
	post_execute_hooks: list[Callable] = Field(default_factory=list)
	
	# Performance tracking
	execution_count: int = 0
	total_execution_time: float = 0.0
	success_count: int = 0
	failure_count: int = 0
	
	# Dynamic loading
	module_path: str | None = None
	lazy_loaded: bool = False
	
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


class ActionRegistry(BaseActionRegistry):
	"""Enhanced action registry with comprehensive management features"""
	model_config = ConfigDict(extra='forbid')
	
	# Enhanced collections
	actions: dict[str, RegisteredAction] = Field(default_factory=dict)
	action_models: dict[str, type] = Field(default_factory=dict)
	
	# Management components
	category_manager: CategoryManager = Field(default_factory=CategoryManager)
	doc_generator: ActionDocGenerator = Field(default_factory=ActionDocGenerator)
	
	# Registry metadata
	registry_version: str = "1.0.0"
	last_updated: float | None = None
	
	# Dynamic loading configuration
	module_search_paths: list[str] = Field(default_factory=list)
	auto_discovery_enabled: bool = False
	
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
		import time
		
		# Determine action name
		action_name = name or getattr(action, 'name', None) or action.__class__.__name__
		
		# Check if already registered
		if action_name in self.actions and not force_update:
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
		
		# Generate or use provided classification
		if classification is None:
			self.category_manager.register_action(action)
			classification = self.category_manager.action_classifications.get(action_name)
		
		# Generate documentation
		documentation = self.doc_generator.generate_documentation(action, classification)
		
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
		self.actions[action_name] = registered_action
		
		# Create dynamic action model
		self._create_action_model(action_name, param_model)
		
		# Update registry metadata
		self.last_updated = time.time()
		
		return registered_action
	
	def _create_action_model(self, action_name: str, param_model: type) -> None:
		"""Create a dynamic action model for the registered action"""
		from ai_agent.registry.views import ActionModel
		
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
		
		self.action_models[action_name] = action_model_class
	
	def unregister_action(self, action_name: str) -> bool:
		"""Unregister an action from the registry"""
		if action_name not in self.actions:
			return False
		
		del self.actions[action_name]
		
		if action_name in self.action_models:
			del self.action_models[action_name]
		
		import time
		self.last_updated = time.time()
		
		return True
	
	def discover_actions(self, filter_criteria: ActionDiscoveryFilter) -> list[RegisteredAction]:
		"""Discover actions based on filter criteria"""
		candidates = list(self.actions.values())
		
		# Apply filters
		if filter_criteria.name_pattern:
			import re
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
					{tag.value for tag in action.classification.tags}
				)
			]
		
		if filter_criteria.required_capabilities:
			candidates = [
				action for action in candidates
				if action.classification and any(
					cap.name in filter_criteria.required_capabilities
					for cap in action.classification.capabilities
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
		return self.actions.get(action_name)
	
	def get_actions_by_category(self, category: str) -> list[RegisteredAction]:
		"""Get all actions in a specific category"""
		return [
			action for action in self.actions.values()
			if action.classification and action.classification.category == category
		]
	
	def get_action_suggestions(self, partial_name: str, limit: int = 10) -> list[RegisteredAction]:
		"""Get action suggestions based on partial name"""
		suggestions = []
		partial_lower = partial_name.lower()
		
		for action in self.actions.values():
			if partial_lower in action.name.lower():
				suggestions.append(action)
		
		# Sort by relevance
		suggestions.sort(key=lambda x: (
			x.name.lower() != partial_lower,  # Exact matches first
			-x.execution_count,  # Then by usage
			x.name  # Finally alphabetically
		))
		
		return suggestions[:limit]
	
	def get_prompt_description(self) -> str:
		"""Get LLM-friendly description of all actions"""
		if not self.actions:
			return "No actions available"
		
		descriptions = []
		for action in self.actions.values():
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
			f"Total Actions: {len(self.actions)}",
			""
		]
		
		# Group by category
		categories = {}
		for action in self.actions.values():
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
		import json
		
		doc_data = {
			"registry_info": {
				"version": self.registry_version,
				"total_actions": len(self.actions),
				"last_updated": self.last_updated
			},
			"actions": {}
		}
		
		for action_name, action in self.actions.items():
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
		if not self.actions:
			return {"message": "No actions registered"}
		
		# Basic stats
		total_actions = len(self.actions)
		total_executions = sum(action.execution_count for action in self.actions.values())
		total_execution_time = sum(action.total_execution_time for action in self.actions.values())
		
		# Category breakdown
		categories = {}
		for action in self.actions.values():
			category = action.classification.category if action.classification else "uncategorized"
			if category not in categories:
				categories[category] = {"count": 0, "executions": 0}
			categories[category]["count"] += 1
			categories[category]["executions"] += action.execution_count
		
		# Performance stats
		execution_times = [action.get_average_execution_time() for action in self.actions.values() if action.execution_count > 0]
		success_rates = [action.get_success_rate() for action in self.actions.values() if action.execution_count > 0]
		
		stats = {
			"registry_info": {
				"total_actions": total_actions,
				"total_executions": total_executions,
				"total_execution_time": total_execution_time,
				"registry_version": self.registry_version,
				"last_updated": self.last_updated
			},
			"categories": categories,
			"performance": {
				"average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
				"average_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
				"fastest_action": min(execution_times) if execution_times else 0,
				"slowest_action": max(execution_times) if execution_times else 0
			},
			"most_used_actions": sorted(
				[(action.name, action.execution_count) for action in self.actions.values()],
				key=lambda x: x[1],
				reverse=True
			)[:10]
		}
		
		return stats
	
	def export_registry(self) -> dict[str, Any]:
		"""Export the entire registry for backup or migration"""
		return {
			"registry_metadata": {
				"version": self.registry_version,
				"last_updated": self.last_updated,
				"total_actions": len(self.actions)
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
				for name, action in self.actions.items()
			}
		}
	
	def clear_registry(self) -> None:
		"""Clear all registered actions (for testing or reset)"""
		self.actions.clear()
		self.action_models.clear()
		import time
		self.last_updated = time.time()