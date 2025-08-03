from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field

from ai_agent.actions.base import ActionContext, ActionParameter, BaseAction
from ai_agent.actions.categories import ActionCategory, ActionTag
from ai_agent.actions.validation import NonEmptyStringField
from ai_agent.registry.views import ActionResult


class ExtractTextParams(ActionParameter):
	"""Parameters for extracting text patterns"""
	text: str = NonEmptyStringField(description='Text to extract from')
	pattern: str = NonEmptyStringField(description='Regex pattern to extract')
	group_index: int = Field(default=0, description='Regex group index to extract')


class ParseJsonParams(ActionParameter):
	"""Parameters for parsing JSON"""
	text: str = NonEmptyStringField(description='JSON text to parse')
	query: str | None = Field(None, description='Optional JSONPath-like query (e.g., "data.items[0].name")')


class FormatTextParams(ActionParameter):
	"""Parameters for formatting text"""
	template: str = NonEmptyStringField(description='Template string with {placeholders}')
	values: dict[str, Any] = Field(description='Values to substitute into template')


class CountWordsParams(ActionParameter):
	"""Parameters for counting words"""
	text: str = NonEmptyStringField(description='Text to count words in')


class ReplaceTextParams(ActionParameter):
	"""Parameters for replacing text"""
	text: str = NonEmptyStringField(description='Text to perform replacement in')
	find: str = NonEmptyStringField(description='Text to find')
	replace: str = Field(default='', description='Text to replace with')
	count: int = Field(default=-1, description='Maximum number of replacements (-1 for all)')


class ExtractTextAction(BaseAction[ExtractTextParams, ActionResult, Any]):
	"""Action to extract text using regex"""
	
	name = 'extract_text'
	description = 'Extract text patterns using regular expressions'
	category = ActionCategory.TEXT_PROCESSING
	tags = {ActionTag.SAFE, ActionTag.READ_ONLY, ActionTag.OFFLINE_CAPABLE, ActionTag.FAST}
	
	async def execute(self, parameters: ExtractTextParams, context: ActionContext[Any]) -> ActionResult:
		try:
			pattern = re.compile(parameters.pattern)
			matches = pattern.findall(parameters.text)
			
			if not matches:
				return ActionResult(
					success=True,
					extracted_content='No matches found',
					metadata={'pattern': parameters.pattern, 'match_count': 0},
				)
			
			# Handle group extraction
			if parameters.group_index > 0:
				# Extract specific group from each match
				extracted = []
				for match in matches:
					if isinstance(match, tuple) and len(match) > parameters.group_index - 1:
						extracted.append(match[parameters.group_index - 1])
					elif parameters.group_index == 1 and isinstance(match, str):
						extracted.append(match)
			else:
				extracted = matches
			
			# Format results
			if len(extracted) == 1:
				content = str(extracted[0])
			else:
				content = '\n'.join(str(item) for item in extracted)
			
			return ActionResult(
				success=True,
				extracted_content=content,
				output_data=extracted,
				metadata={'pattern': parameters.pattern, 'match_count': len(extracted)},
			)
		except re.error as e:
			return ActionResult(
				success=False,
				error=f'Invalid regex pattern: {str(e)}',
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Failed to extract text: {str(e)}',
			)


class ParseJsonAction(BaseAction[ParseJsonParams, ActionResult, Any]):
	"""Action to parse JSON data"""
	
	name = 'parse_json'
	description = 'Parse JSON text and optionally extract specific values'
	category = ActionCategory.DATA_TRANSFORMATION
	tags = {ActionTag.SAFE, ActionTag.READ_ONLY, ActionTag.OFFLINE_CAPABLE, ActionTag.FAST}
	
	async def execute(self, parameters: ParseJsonParams, context: ActionContext[Any]) -> ActionResult:
		try:
			# Parse JSON
			data = json.loads(parameters.text)
			
			# Apply query if provided
			if parameters.query:
				result = self._query_json(data, parameters.query)
			else:
				result = data
			
			return ActionResult(
				success=True,
				extracted_content=json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result),
				output_data=result,
				metadata={'query': parameters.query, 'type': type(result).__name__},
			)
		except json.JSONDecodeError as e:
			return ActionResult(
				success=False,
				error=f'Invalid JSON: {str(e)}',
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Failed to parse JSON: {str(e)}',
			)
	
	def _query_json(self, data: Any, query: str) -> Any:
		"""Simple JSONPath-like query implementation"""
		parts = query.replace('[', '.').replace(']', '').split('.')
		result = data
		
		for part in parts:
			if not part:
				continue
			
			if isinstance(result, dict):
				result = result.get(part)
			elif isinstance(result, list):
				try:
					index = int(part)
					result = result[index]
				except (ValueError, IndexError):
					return None
			else:
				return None
			
			if result is None:
				return None
		
		return result


class FormatTextAction(BaseAction[FormatTextParams, ActionResult, Any]):
	"""Action to format text using templates"""
	
	name = 'format_text'
	description = 'Format text using template strings'
	category = ActionCategory.TEXT_PROCESSING
	tags = {ActionTag.SAFE, ActionTag.OFFLINE_CAPABLE, ActionTag.FAST}
	
	async def execute(self, parameters: FormatTextParams, context: ActionContext[Any]) -> ActionResult:
		try:
			formatted = parameters.template.format(**parameters.values)
			
			return ActionResult(
				success=True,
				extracted_content=formatted,
				metadata={'template_length': len(parameters.template), 'values_count': len(parameters.values)},
			)
		except KeyError as e:
			return ActionResult(
				success=False,
				error=f'Missing template value: {str(e)}',
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Failed to format text: {str(e)}',
			)


class CountWordsAction(BaseAction[CountWordsParams, ActionResult, Any]):
	"""Action to count words in text"""
	
	name = 'count_words'
	description = 'Count words, lines, and characters in text'
	category = ActionCategory.TEXT_PROCESSING
	tags = {ActionTag.SAFE, ActionTag.READ_ONLY, ActionTag.OFFLINE_CAPABLE, ActionTag.FAST}
	
	async def execute(self, parameters: CountWordsParams, context: ActionContext[Any]) -> ActionResult:
		try:
			text = parameters.text
			
			# Count metrics
			words = len(text.split())
			lines = len(text.splitlines())
			characters = len(text)
			characters_no_spaces = len(text.replace(' ', '').replace('\t', '').replace('\n', ''))
			
			stats = {
				'words': words,
				'lines': lines,
				'characters': characters,
				'characters_no_spaces': characters_no_spaces,
			}
			
			content = '\n'.join(f'{key}: {value}' for key, value in stats.items())
			
			return ActionResult(
				success=True,
				extracted_content=content,
				output_data=stats,
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Failed to count words: {str(e)}',
			)


class ReplaceTextAction(BaseAction[ReplaceTextParams, ActionResult, Any]):
	"""Action to replace text"""
	
	name = 'replace_text'
	description = 'Find and replace text'
	category = ActionCategory.TEXT_PROCESSING
	tags = {ActionTag.SAFE, ActionTag.OFFLINE_CAPABLE, ActionTag.FAST}
	
	async def execute(self, parameters: ReplaceTextParams, context: ActionContext[Any]) -> ActionResult:
		try:
			if parameters.count == -1:
				result = parameters.text.replace(parameters.find, parameters.replace)
				replacement_count = parameters.text.count(parameters.find)
			else:
				result = parameters.text.replace(parameters.find, parameters.replace, parameters.count)
				replacement_count = min(parameters.count, parameters.text.count(parameters.find))
			
			return ActionResult(
				success=True,
				extracted_content=result,
				metadata={
					'find': parameters.find,
					'replace': parameters.replace,
					'replacement_count': replacement_count,
				},
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Failed to replace text: {str(e)}',
			)