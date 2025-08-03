from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel

from ai_agent.actions.base import BaseAction
from ai_agent.registry.views import RegisteredAction


class ActionDocGenerator:
	"""Generate documentation for actions"""
	
	@staticmethod
	def generate_markdown_doc(action: BaseAction | RegisteredAction) -> str:
		"""Generate markdown documentation for an action"""
		if isinstance(action, BaseAction):
			metadata = action.get_metadata()
			param_model = action.get_parameter_model()
		else:
			metadata = {
				'name': action.name,
				'description': action.description,
				'category': getattr(action, 'category', None),
				'tags': getattr(action, 'tags', set()),
				'priority': getattr(action, 'priority', 0),
				'requires_capabilities': getattr(action, 'requires_capabilities', set()),
			}
			param_model = action.param_model
		
		lines = [
			f"## {metadata['name']}",
			"",
			f"**Description**: {metadata['description']}",
			"",
		]
		
		if metadata.get('category'):
			lines.append(f"**Category**: {metadata['category']}")
		
		if metadata.get('tags'):
			tags_str = ', '.join(f"`{tag}`" for tag in metadata['tags'])
			lines.append(f"**Tags**: {tags_str}")
		
		if metadata.get('priority'):
			lines.append(f"**Priority**: {metadata['priority']}")
		
		if metadata.get('requires_capabilities'):
			caps_str = ', '.join(f"`{cap}`" for cap in metadata['requires_capabilities'])
			lines.append(f"**Required Capabilities**: {caps_str}")
		
		lines.extend(["", "### Parameters", ""])
		
		# Document parameters
		if param_model:
			schema = param_model.model_json_schema()
			properties = schema.get('properties', {})
			required = set(schema.get('required', []))
			
			if not properties:
				lines.append("No parameters required.")
			else:
				for name, prop in properties.items():
					param_type = prop.get('type', 'any')
					description = prop.get('description', 'No description')
					is_required = name in required
					
					lines.append(f"- **{name}** ({param_type})")
					lines.append(f"  - Description: {description}")
					lines.append(f"  - Required: {'Yes' if is_required else 'No'}")
					
					if 'default' in prop:
						lines.append(f"  - Default: `{prop['default']}`")
					
					if 'minimum' in prop or 'maximum' in prop:
						constraints = []
						if 'minimum' in prop:
							constraints.append(f"min: {prop['minimum']}")
						if 'maximum' in prop:
							constraints.append(f"max: {prop['maximum']}")
						lines.append(f"  - Constraints: {', '.join(constraints)}")
					
					if 'enum' in prop:
						lines.append(f"  - Allowed values: {', '.join(f'`{v}`' for v in prop['enum'])}")
					
					if 'examples' in prop:
						lines.append(f"  - Examples: {', '.join(f'`{ex}`' for ex in prop['examples'])}")
					
					lines.append("")
		
		if hasattr(action, 'example_usage') and action.example_usage:
			lines.extend([
				"### Example Usage",
				"",
				"```json",
				action.example_usage,
				"```",
				""
			])
		
		return '\n'.join(lines)
	
	@staticmethod
	def generate_llm_description(action: RegisteredAction) -> str:
		"""Generate LLM-friendly action description"""
		# This is already handled by RegisteredAction.prompt_description()
		return action.prompt_description()
	
	@staticmethod
	def generate_json_schema(action: BaseAction | RegisteredAction) -> dict[str, Any]:
		"""Generate JSON schema for the action"""
		if isinstance(action, BaseAction):
			metadata = action.get_metadata()
			param_model = action.get_parameter_model()
		else:
			metadata = {
				'name': action.name,
				'description': action.description,
			}
			param_model = action.param_model
		
		schema = {
			'name': metadata['name'],
			'description': metadata['description'],
			'parameters': param_model.model_json_schema() if param_model else {'type': 'object', 'properties': {}}
		}
		
		return schema
	
	@staticmethod
	def generate_openapi_operation(action: BaseAction | RegisteredAction) -> dict[str, Any]:
		"""Generate OpenAPI operation schema for the action"""
		if isinstance(action, BaseAction):
			metadata = action.get_metadata()
			param_model = action.get_parameter_model()
		else:
			metadata = {
				'name': action.name,
				'description': action.description,
				'tags': getattr(action, 'tags', set()),
			}
			param_model = action.param_model
		
		operation = {
			'operationId': metadata['name'],
			'summary': metadata['description'],
			'tags': list(metadata.get('tags', [])),
			'requestBody': {
				'required': True,
				'content': {
					'application/json': {
						'schema': param_model.model_json_schema() if param_model else {'type': 'object'}
					}
				}
			},
			'responses': {
				'200': {
					'description': 'Action executed successfully',
					'content': {
						'application/json': {
							'schema': {
								'$ref': '#/components/schemas/ActionResult'
							}
						}
					}
				},
				'400': {
					'description': 'Invalid parameters'
				},
				'500': {
					'description': 'Action execution failed'
				}
			}
		}
		
		return operation
	
	@classmethod
	def generate_collection_doc(cls, actions: list[BaseAction | RegisteredAction], format: str = 'markdown') -> str:
		"""Generate documentation for a collection of actions"""
		if format == 'markdown':
			sections = [
				"# Available Actions",
				"",
				f"Total actions: {len(actions)}",
				"",
				"## Table of Contents",
				"",
			]
			
			# Group by category
			by_category: dict[str, list[Any]] = {}
			for action in actions:
				if isinstance(action, BaseAction):
					category = action.category or 'Uncategorized'
				else:
					category = getattr(action, 'category', 'Uncategorized')
				
				if category not in by_category:
					by_category[category] = []
				by_category[category].append(action)
			
			# Generate TOC
			for category, cat_actions in sorted(by_category.items()):
				sections.append(f"- [{category}](#{category.lower().replace(' ', '-').replace('.', '-')})")
				for action in cat_actions:
					name = action.name if hasattr(action, 'name') else action.__class__.__name__
					sections.append(f"  - [{name}](#{name.lower().replace(' ', '-')})")
			
			sections.append("")
			
			# Generate documentation for each category
			for category, cat_actions in sorted(by_category.items()):
				sections.append(f"# {category}")
				sections.append("")
				
				for action in cat_actions:
					sections.append(cls.generate_markdown_doc(action))
					sections.append("---")
					sections.append("")
			
			return '\n'.join(sections)
		
		elif format == 'json':
			docs = []
			for action in actions:
				docs.append(cls.generate_json_schema(action))
			return json.dumps(docs, indent=2)
		
		else:
			raise ValueError(f'Unsupported format: {format}')