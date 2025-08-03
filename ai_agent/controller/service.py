from __future__ import annotations

import logging
from typing import Generic, TypeVar

from pydantic import BaseModel

from ai_agent.controller.registry.service import Registry
from ai_agent.common.models import ActionResult

logger = logging.getLogger(__name__)

Context = TypeVar('Context')
T = TypeVar('T', bound=BaseModel)


class Controller(Generic[Context]):
	"""Generic controller for action execution"""
	
	def __init__(
		self,
		exclude_actions: list[str] | None = None,
		output_model: type[T] | None = None,
		context_param_types: dict[str, type] | None = None,
	):
		"""
		Initialize controller with optional configuration.
		
		Args:
			exclude_actions: List of action names to exclude from registry
			output_model: Optional structured output model for completion actions
			context_param_types: Additional context parameter types for injection
		"""
		self.registry = Registry[Context](
			exclude_actions=exclude_actions or [],
			context_param_types=context_param_types or {},
		)
		self.output_model = output_model
		
		# Register default actions
		self._register_default_actions()
	
	def _register_default_actions(self):
		"""Register default actions that are useful for any agent"""
		
		# Register done action
		if self.output_model:
			self._register_done_with_output()
		else:
			self._register_done()
		
		# Register wait action
		self._register_wait()
	
	def _register_done(self):
		"""Register simple done action"""
		from ai_agent.controller.actions import DoneAction
		
		@self.registry.action(
			'Mark task as completed',
			param_model=DoneAction,
		)
		async def done(params: DoneAction) -> ActionResult:
			return ActionResult(
				success=True,
				is_done=True,
				extracted_content=params.text or 'Task completed',
			)
	
	def _register_done_with_output(self):
		"""Register done action with structured output"""
		from ai_agent.controller.actions import StructuredOutputAction
		
		@self.registry.action(
			f'Mark task as completed and return structured output of type {self.output_model.__name__}',
			param_model=StructuredOutputAction[self.output_model],
		)
		async def done(params: StructuredOutputAction[T]) -> ActionResult:
			return ActionResult(
				success=True,
				is_done=True,
				output_data=params.structured_output,
				extracted_content=f'Task completed with {self.output_model.__name__}',
			)
	
	def _register_wait(self):
		"""Register wait action"""
		from ai_agent.controller.actions import WaitAction
		
		@self.registry.action(
			'Wait for specified duration',
			param_model=WaitAction,
		)
		async def wait(params: WaitAction) -> ActionResult:
			import asyncio
			await asyncio.sleep(params.seconds)
			return ActionResult(
				success=True,
				extracted_content=f'Waited for {params.seconds} seconds',
			)
	
	async def act(self, action: ActionModel, **kwargs) -> ActionResult:
		"""
		Execute an action through the registry.
		
		Args:
			action: The action model to execute
			**kwargs: Additional context parameters for injection
			
		Returns:
			ActionResult from the executed action
		"""
		try:
			print(f"DEBUG: Controller.act() called with action: {action}")
			print(f"DEBUG: Controller.act() kwargs: {kwargs}")
			result = await self.registry.execute_action(action, **kwargs)
			print(f"DEBUG: Controller.act() registry returned: {result}")
			return result
		except Exception as e:
			print(f"DEBUG: Controller.act() exception: {type(e).__name__}: {e}")
			logger.error(f'Action execution failed: {e}')
			return ActionResult(
				success=False,
				error=str(e),
			)