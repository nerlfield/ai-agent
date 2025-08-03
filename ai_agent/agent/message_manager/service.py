from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, Type

from ai_agent.agent.message_manager.interfaces import (
	AgentStepInfo,
	ContextPromptBuilder,
	ContextState,
	ExecutionResult,
)
from ai_agent.agent.message_manager.views import (
	HistoryItem,
	MessageManagerState,
)
from ai_agent.agent.views import AgentOutput
from ai_agent.llm.messages import (
	BaseMessage,
	SystemMessage,
)
from ai_agent.utils.core import time_execution_sync

logger = logging.getLogger(__name__)


# ========== Logging Helper Functions ==========
# These functions are used ONLY for formatting debug log output.
# They do NOT affect the actual message content sent to the LLM.
# All logging functions start with _log_ for easy identification.


def _log_get_message_emoji(message: BaseMessage) -> str:
	"""Get emoji for a message type - used only for logging display"""
	emoji_map = {
		'UserMessage': '💬',
		'SystemMessage': '🧠',
		'AssistantMessage': '🔨',
	}
	return emoji_map.get(message.__class__.__name__, '🎮')


def _log_format_message_line(message: BaseMessage, content: str, is_last_message: bool, terminal_width: int) -> list[str]:
	"""Format a single message for logging display"""
	try:
		lines = []

		# Get emoji and token info
		emoji = _log_get_message_emoji(message)
		# token_str = str(message.metadata.tokens).rjust(4)
		# TODO: fix the token count
		token_str = '??? (TODO)'
		prefix = f'{emoji}[{token_str}]: '

		# Calculate available width (emoji=2 visual cols + [token]: =8 chars)
		content_width = terminal_width - 10

		# Handle last message wrapping
		if is_last_message and len(content) > content_width:
			# Find a good break point
			break_point = content.rfind(' ', 0, content_width)
			if break_point > content_width * 0.7:  # Keep at least 70% of line
				first_line = content[:break_point]
				rest = content[break_point + 1 :]
			else:
				# No good break point, just truncate
				first_line = content[:content_width]
				rest = content[content_width:]

			lines.append(prefix + first_line)

			# Second line with 10-space indent
			if rest:
				if len(rest) > terminal_width - 10:
					rest = rest[: terminal_width - 10]
				lines.append(' ' * 10 + rest)
		else:
			# Single line - truncate if needed
			if len(content) > content_width:
				content = content[:content_width]
			lines.append(prefix + content)

		return lines
	except Exception as e:
		logger.warning(f'Failed to format message line for logging: {e}')
		# Return a simple fallback line
		return ['❓[   ?]: [Error formatting message]']


# ========== End of Logging Helper Functions ==========


class FileSystem(Protocol):
	"""Protocol for file system operations"""
	pass


class MessageManager:
	vision_detail_level: Literal['auto', 'low', 'high']

	def __init__(
		self,
		task: str,
		system_message: SystemMessage,
		file_system: FileSystem,
		state: MessageManagerState = MessageManagerState(),
		use_thinking: bool = True,
		include_attributes: list[str] | None = None,
		message_context: str | None = None,
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
		max_history_items: int | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		include_tool_call_examples: bool = False,
		prompt_builder_class: Type[ContextPromptBuilder] | None = None,
	):
		self.task = task
		self.state = state
		self.system_prompt = system_message
		self.file_system = file_system
		self.sensitive_data_description = ''
		self.use_thinking = use_thinking
		self.max_history_items = max_history_items
		self.vision_detail_level = vision_detail_level
		self.include_tool_call_examples = include_tool_call_examples
		self.prompt_builder_class = prompt_builder_class

		assert max_history_items is None or max_history_items > 5, 'max_history_items must be None or greater than 5'

		# Store settings as direct attributes instead of in a settings object
		self.include_attributes = include_attributes or []
		self.message_context = message_context
		self.sensitive_data = sensitive_data
		self.last_input_messages = []
		# Only initialize messages if state is empty
		if len(self.state.history.get_messages()) == 0:
			self._add_message_with_type(self.system_prompt, 'system')

	@property
	def agent_history_description(self) -> str:
		"""Build agent history description from list of items, respecting max_history_items limit"""
		if self.max_history_items is None:
			# Include all items
			return '\n'.join(item.to_string() for item in self.state.agent_history_items)

		total_items = len(self.state.agent_history_items)

		# If we have fewer items than the limit, just return all items
		if total_items <= self.max_history_items:
			return '\n'.join(item.to_string() for item in self.state.agent_history_items)

		# We have more items than the limit, so we need to omit some
		omitted_count = total_items - self.max_history_items

		# Show first item + omitted message + most recent (max_history_items - 1) items
		# The omitted message doesn't count against the limit, only real history items do
		recent_items_count = self.max_history_items - 1  # -1 for first item

		items_to_include = [
			self.state.agent_history_items[0].to_string(),  # Keep first item (initialization)
			f'<sys>[... {omitted_count} previous steps omitted...]</sys>',
		]
		# Add most recent items
		items_to_include.extend([item.to_string() for item in self.state.agent_history_items[-recent_items_count:]])

		return '\n'.join(items_to_include)

	def add_new_task(self, new_task: str) -> None:
		self.task = new_task
		task_update_item = HistoryItem(system_message=f'User updated <user_request> to: {new_task}')
		self.state.agent_history_items.append(task_update_item)

	def _update_agent_history_description(
		self,
		model_output: AgentOutput | None = None,
		result: list[ExecutionResult] | None = None,
		step_info: AgentStepInfo | None = None,
	) -> None:
		"""Update the agent history description"""

		if result is None:
			result = []
		step_number = step_info.step_number if step_info else None

		self.state.read_state_description = ''

		action_results = ''
		result_len = len(result)
		for idx, action_result in enumerate(result):
			if action_result.include_extracted_content_only_once and action_result.extracted_content:
				self.state.read_state_description += action_result.extracted_content + '\n'
				logger.debug(f'Added extracted_content to read_state_description: {action_result.extracted_content}')

			if action_result.long_term_memory:
				action_results += f'Action {idx + 1}/{result_len}: {action_result.long_term_memory}\n'
				logger.debug(f'Added long_term_memory to action_results: {action_result.long_term_memory}')
			elif action_result.extracted_content and not action_result.include_extracted_content_only_once:
				action_results += f'Action {idx + 1}/{result_len}: {action_result.extracted_content}\n'
				logger.debug(f'Added extracted_content to action_results: {action_result.extracted_content}')

			if action_result.error:
				if len(action_result.error) > 200:
					error_text = action_result.error[:100] + '......' + action_result.error[-100:]
				else:
					error_text = action_result.error
				action_results += f'Action {idx + 1}/{result_len}: {error_text}\n'
				logger.debug(f'Added error to action_results: {error_text}')

		if action_results:
			action_results = f'Action Results:\n{action_results}'
		action_results = action_results.strip('\n') if action_results else None

		# Build the history item
		if model_output is None:
			# Only add error history item if we have a valid step number
			if step_number is not None and step_number > 0:
				history_item = HistoryItem(step_number=step_number, error='Agent failed to output in the right format.')
				self.state.agent_history_items.append(history_item)
		else:
			history_item = HistoryItem(
				step_number=step_number,
				evaluation_previous_goal=model_output.current_state.evaluation_previous_goal,
				memory=model_output.current_state.memory,
				next_goal=model_output.current_state.next_goal,
				action_results=action_results,
			)
			self.state.agent_history_items.append(history_item)

	def _get_sensitive_data_description(self, context_url: str | None) -> str:
		sensitive_data = self.sensitive_data
		if not sensitive_data:
			return ''

		# Collect placeholders for sensitive data
		placeholders: set[str] = set()

		for key, value in sensitive_data.items():
			if isinstance(value, dict):
				# New format: {domain: {key: value}}
				if context_url and self._match_url_with_domain_pattern(context_url, key):
					placeholders.update(value.keys())
			else:
				# Old format: {key: value}
				placeholders.add(key)

		if placeholders:
			placeholder_list = sorted(list(placeholders))
			info = f'Here are placeholders for sensitive data:\n{placeholder_list}\n'
			info += 'To use them, write <secret>the placeholder name</secret>'
			return info

		return ''

	def _match_url_with_domain_pattern(self, url: str, pattern: str) -> bool:
		"""Simple URL pattern matching - can be overridden for specific implementations"""
		return pattern in url

	def add_state_message(
		self,
		context_state: ContextState,
		model_output: AgentOutput | None = None,
		result: list[ExecutionResult] | None = None,
		step_info: AgentStepInfo | None = None,
		use_vision=True,
		page_filtered_actions: str | None = None,
		sensitive_data=None,
		available_file_paths: list[str] | None = None,
		**kwargs: Any,
	) -> None:
		"""Add context state as human message"""

		self._update_agent_history_description(model_output, result, step_info)
		if sensitive_data:
			self.sensitive_data_description = self._get_sensitive_data_description(getattr(context_state, 'url', None))

		# Use only the current screenshot/media if available
		media_items = []
		if hasattr(context_state, 'screenshot') and context_state.screenshot:
			media_items.append(context_state.screenshot)
		elif hasattr(context_state, 'media_data') and context_state.media_data:
			media_items.extend(context_state.media_data.values())

		# Create prompt using the provided prompt builder class
		if self.prompt_builder_class:
			state_message = self.prompt_builder_class(
				context_state=context_state,
				file_system=self.file_system,
				agent_history_description=self.agent_history_description,
				read_state_description=self.state.read_state_description,
				task=self.task,
				include_attributes=self.include_attributes,
				step_info=step_info,
				page_filtered_actions=page_filtered_actions,
				sensitive_data=self.sensitive_data_description,
				available_file_paths=available_file_paths,
				media_items=media_items,
				vision_detail_level=self.vision_detail_level,
				**kwargs,
			).get_user_message(use_vision)
		else:
			# Fallback: create a simple user message if no prompt builder provided
			from ai_agent.llm.messages import UserMessage
			state_message = UserMessage(content=f"Current task: {self.task}")

		self._add_message_with_type(state_message, 'state')

	def _log_history_lines(self) -> str:
		"""Generate a formatted log string of message history for debugging / printing to terminal"""
		# TODO: fix logging
		return ''

	def get_messages(self) -> list[BaseMessage]:
		"""Get current message list, potentially trimmed to max tokens"""

		# Log message history for debugging
		logger.debug(self._log_history_lines())
		self.last_input_messages = self.state.history.get_messages()
		return self.last_input_messages

	def _add_message_with_type(self, message: BaseMessage, message_type: Literal['system', 'state', 'consistent']) -> None:
		"""Add message to history"""

		# filter out sensitive data from the message
		if self.sensitive_data:
			message = self._filter_sensitive_data(message)

		if message_type == 'system':
			self.state.history.system_message = message
		elif message_type == 'state':
			self.state.history.state_message = message
		elif message_type == 'consistent':
			self.state.history.consistent_messages.append(message)

	def _filter_sensitive_data(self, message: BaseMessage) -> BaseMessage:
		"""Filter sensitive data from messages - implementation specific"""
		# This is a placeholder - specific implementations can override
		return message

	def add_assistant_message(self, output: AgentOutput) -> None:
		"""Add assistant message to history"""
		from ai_agent.llm.messages import AssistantMessage

		# Create assistant message with output
		assistant_msg = AssistantMessage(
			content=output.model_dump_json(exclude_unset=True)
		)
		self._add_message_with_type(assistant_msg, 'consistent')

	def add_user_message(self, content: str) -> None:
		"""Add user message to history"""
		from ai_agent.llm.messages import UserMessage

		user_msg = UserMessage(content=content)
		self._add_message_with_type(user_msg, 'consistent')