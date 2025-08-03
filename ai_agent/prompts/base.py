from __future__ import annotations

import importlib.resources
from abc import ABC, abstractmethod
from typing import Any, Optional

from ai_agent.llm.messages import SystemMessage


class BaseSystemPrompt(ABC):
	"""Base class for system prompts"""
	
	def __init__(
		self,
		action_description: str,
		max_actions_per_step: int = 10,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
		use_thinking: bool = True,
		template_name: str | None = None,
		**kwargs: Any,
	):
		self.default_action_description = action_description
		self.max_actions_per_step = max_actions_per_step
		self.use_thinking = use_thinking
		self.template_name = template_name or self._get_default_template_name()
		self.kwargs = kwargs
		
		prompt = ''
		if override_system_message:
			prompt = override_system_message
		else:
			self._load_prompt_template()
			prompt = self._format_template()
		
		if extend_system_message:
			prompt += f'\n{extend_system_message}'
		
		self.system_message = SystemMessage(content=prompt, cache=True)
	
	@abstractmethod
	def _get_default_template_name(self) -> str:
		"""Get the default template name for this prompt type"""
		pass
	
	@abstractmethod
	def _format_template(self) -> str:
		"""Format the loaded template with specific variables"""
		pass
	
	def _load_prompt_template(self) -> None:
		"""Load the prompt template from the markdown file."""
		try:
			# This works both in development and when installed as a package
			with importlib.resources.files('ai_agent.prompts.templates').joinpath(self.template_name).open('r', encoding='utf-8') as f:
				self.prompt_template = f.read()
		except Exception as e:
			raise RuntimeError(f'Failed to load system prompt template {self.template_name}: {e}')
	
	def get_system_message(self) -> SystemMessage:
		"""Get the system prompt for the agent."""
		return self.system_message


class GenericSystemPrompt(BaseSystemPrompt):
	"""Generic system prompt for any agent type"""
	
	def _get_default_template_name(self) -> str:
		if self.use_thinking:
			return 'generic_system_prompt.md'
		else:
			return 'generic_system_prompt_no_thinking.md'
	
	def _format_template(self) -> str:
		"""Format the template with generic variables"""
		return self.prompt_template.format(
			max_actions=self.max_actions_per_step,
			action_description=self.default_action_description,
			**self.kwargs
		)