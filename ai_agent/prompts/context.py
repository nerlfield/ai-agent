from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Protocol


class FileSystem(Protocol):
	"""Protocol for file system operations"""
	
	def describe(self) -> str:
		"""Describe the file system"""
		...
	
	def get_todo_contents(self) -> str:
		"""Get TODO contents"""
		...


class ContextProvider(ABC):
	"""Base class for providing context to agent prompts"""
	
	@abstractmethod
	def get_context_name(self) -> str:
		"""Return the name for this context section"""
		pass
	
	@abstractmethod
	def get_context_description(self) -> str:
		"""Return the context content for the prompt"""
		pass
	
	@abstractmethod
	def get_input_description(self) -> str:
		"""Return description for the <input> section"""
		pass
	
	def get_template_variables(self) -> dict[str, str]:
		"""Return template variables for system prompt"""
		return {}


class TaskContextProvider(ContextProvider):
	"""Provides task and agent state context"""
	
	def __init__(
		self,
		task: str,
		file_system: FileSystem | None = None,
		step_info: Any | None = None,
		sensitive_data: str | None = None,
		available_file_paths: list[str] | None = None,
	):
		self.task = task
		self.file_system = file_system
		self.step_info = step_info
		self.sensitive_data = sensitive_data
		self.available_file_paths = available_file_paths
	
	def get_context_name(self) -> str:
		return "agent_state"
	
	def get_context_description(self) -> str:
		if self.step_info:
			step_info_description = f'Step {self.step_info.step_number + 1} of {self.step_info.max_steps} max possible steps\n'
		else:
			step_info_description = ''
		time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
		step_info_description += f'Current date and time: {time_str}'
		
		_todo_contents = self.file_system.get_todo_contents() if self.file_system else ''
		if not len(_todo_contents):
			_todo_contents = '[Current todo.md is empty, fill it with your plan when applicable]'
		
		agent_state = f"""
<user_request>
{self.task}
</user_request>
<file_system>
{self.file_system.describe() if self.file_system else 'No file system available'}
</file_system>
<todo_contents>
{_todo_contents}
</todo_contents>
"""
		if self.sensitive_data:
			agent_state += f'<sensitive_data>\n{self.sensitive_data}\n</sensitive_data>\n'
		
		agent_state += f'<step_info>\n{step_info_description}\n</step_info>\n'
		if self.available_file_paths:
			agent_state += '<available_file_paths>\n' + '\n'.join(self.available_file_paths) + '\n</available_file_paths>\n'
		return agent_state.strip()
	
	def get_input_description(self) -> str:
		return "<agent_state>: Current <user_request>, summary of <file_system>, <todo_contents>, and <step_info>."


class HistoryContextProvider(ContextProvider):
	"""Provides agent history context"""
	
	def __init__(self, history_description: str | None):
		self.history_description = history_description
	
	def get_context_name(self) -> str:
		return "agent_history"
	
	def get_context_description(self) -> str:
		return self.history_description.strip('\n') if self.history_description else ''
	
	def get_input_description(self) -> str:
		return "<agent_history>: A chronological event stream including your previous actions and their results."


class ReadStateContextProvider(ContextProvider):
	"""Provides extracted/read data context"""
	
	def __init__(self, read_state_description: str | None):
		self.read_state_description = read_state_description
	
	def get_context_name(self) -> str:
		return "read_state"
	
	def get_context_description(self) -> str:
		return self.read_state_description.strip('\n') if self.read_state_description else ''
	
	def get_input_description(self) -> str:
		return "<read_state>: This will be displayed only if your previous action was extract_data or read_file. This data is only shown in the current step."


class EnvironmentContextProvider(ContextProvider):
	"""Base class for environment-specific context providers"""
	
	def __init__(self, environment_state: Any, available_actions: list[str] | None = None):
		self.environment_state = environment_state
		self.available_actions = available_actions
	
	@abstractmethod
	def get_context_name(self) -> str:
		pass
	
	@abstractmethod
	def get_context_description(self) -> str:
		pass
	
	def get_input_description(self) -> str:
		return f"<{self.get_context_name()}>: Current environment state and available actions."