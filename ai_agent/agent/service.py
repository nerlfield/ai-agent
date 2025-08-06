from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Generic, Type, TypeVar

from bubus import EventBus
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_agent.agent.message_manager.interfaces import AgentStepInfo, ContextState
from ai_agent.agent.message_manager.service import MessageManager
from ai_agent.agent.message_manager.utils import save_conversation
from ai_agent.agent.views import (
	ActionResult,
	AgentError,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	AgentSettings,
	AgentState,
	AgentStructuredOutput,
	ContextStateHistory,
	StepMetadata,
)
from ai_agent.llm.base import BaseChatModel
from ai_agent.llm.exceptions import (
	ModelError,
	ModelRateLimitError,
	ModelProviderError,
)
from ai_agent.llm.messages import AssistantMessage
from ai_agent.llm.views import ChatInvokeCompletion
from ai_agent.prompts.base import GenericSystemPrompt
from ai_agent.prompts.context import ContextProvider
from ai_agent.filesystem import FileSystem
from ai_agent.prompts.prompts import AgentMessagePrompt
from ai_agent.registry import ActionModel, ActionRegistry
from ai_agent.tokens.service import TokenCost
from ai_agent.utils.core import SignalHandler, time_execution_async

# Type variables for generic agent
Context = TypeVar('Context')
State = TypeVar('State', bound=ContextState)
AgentStructuredOutputT = TypeVar('AgentStructuredOutputT', bound=BaseModel)

logger = logging.getLogger(__name__)


class ExecutionContext(Generic[Context, State]):
	"""Generic execution context for any tool/environment"""
	
	async def get_current_state(self, cache_elements: bool = True, include_media: bool = True) -> State:
		"""Get current environment state"""
		raise NotImplementedError
	
	async def execute_action(self, action: ActionModel, **kwargs) -> ActionResult:
		"""Execute a single action in the environment"""
		raise NotImplementedError
	
	async def recover_from_error(self, error: Exception) -> None:
		"""Attempt to recover from execution errors"""
		pass
	
	def get_action_registry(self) -> ActionRegistry:
		"""Get available actions for current context"""
		raise NotImplementedError
	
	def get_context_providers(self, **kwargs) -> list[ContextProvider]:
		"""Get context providers for prompt building"""
		return []


class Controller(Generic[Context]):
	"""Generic controller for action execution"""
	
	def __init__(self, execution_context: ExecutionContext[Context, Any]):
		self.execution_context = execution_context
		self.registry = execution_context.get_action_registry()
	
	async def act(self, action: ActionModel, **kwargs) -> ActionResult:
		"""Execute an action through the execution context"""
		return await self.execution_context.execute_action(action, **kwargs)


class GenericAgent(Generic[Context, State, AgentStructuredOutputT]):
	"""Generic agent that can work with any execution context"""
	
	execution_context: ExecutionContext[Context, State] | None = None
	controller: Controller[Context] | None = None
	
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		execution_context: ExecutionContext[Context, State] | None = None,
		controller: Controller[Context] | None = None,
		screenshot_service: Any | None = None,
		planner_llm: BaseChatModel | None = None,
		settings: AgentSettings = AgentSettings(),
		event_bus: EventBus | None = None,
		log_dir: str | None = None,
		structured_output_type: Type[AgentStructuredOutputT] | None = None,
		**kwargs,
	):
		self.task = task
		self.llm = llm
		self.planner_llm = planner_llm
		self.logger = logging.getLogger(f'ai_agent.{self.__class__.__name__}')
		self.event_bus = event_bus
		self.structured_output_type = structured_output_type
		
		# Core components
		self.execution_context = execution_context
		self.controller = controller or (Controller(execution_context) if execution_context else None)
		self.screenshot_service = screenshot_service
		
		# Agent settings
		self.settings = settings
		
		# Agent state
		self.id = str(uuid.uuid4())
		self.started = False
		self.state = AgentState()
		
		# Initialize message manager state if not set
		if not self.state.message_manager_state:
			from ai_agent.agent.message_manager.views import MessageManagerState
			self.state.message_manager_state = MessageManagerState()
			
		self.history = AgentHistoryList()
		self.token_cost_service = TokenCost() if settings.calculate_cost else None
		
		# File system
		self.file_system: FileSystem | None = kwargs.get('file_system')
		
		# Step tracking
		self.last_step = 0
		self.step_start_time: float | None = None
		
		# Action models
		self._action_models: dict[str, Type[ActionModel]] = {}
		self._last_actions: str | None = None
		
		# System prompt
		action_desc = self._get_action_descriptions()
		self.system_prompt = GenericSystemPrompt(
			action_description=action_desc,
			max_actions_per_step=settings.max_actions_per_step,
			override_system_message=settings.override_system_message,
			extend_system_message=settings.extend_system_message,
			use_thinking=settings.use_thinking,
		).get_system_message()
		
		# Message manager
		self.msg_manager = MessageManager(
			task=task,
			system_message=self.system_prompt,
			file_system=self.file_system,
			state=self.state.message_manager_state,
			use_thinking=settings.use_thinking,
			message_context=settings.message_context,
			max_history_items=settings.max_history_items,
			vision_detail_level='low' if settings.flash_mode else 'auto',
			include_tool_call_examples=settings.include_tool_call_examples,
		)
		
		# Interruption handler
		self.sigint_handler = SignalHandler(
			loop=asyncio.get_event_loop(),
			interruptible_task_patterns=['step', 'multi_act', 'execute'],
		)
		
		# Telemetry
		self._init_telemetry()
		
		self.logger.info(f'🌟 Agent initialized with ID: {self.id}')
	
	def _init_telemetry(self):
		"""Initialize telemetry tracking"""
		pass  # Placeholder for telemetry
	
	def _get_action_descriptions(self) -> str:
		"""Get descriptions of all available actions"""
		if not self.controller:
			return "No actions available"
		
		actions = []
		for action_name, action in self.controller.registry.registry.actions.items():
			actions.append(action.prompt_description())
		
		action_descriptions = '\n'.join(actions)
		self._log_debug(f"📋 Available actions: {len(actions)} total")
		
		return action_descriptions
	
	def _log_debug(self, message: str, level: str = "INFO") -> None:
		"""Clean debug logging with emojis"""
		# Check if message already starts with an emoji, if so don't add another
		if len(message) > 0 and ord(message[0]) > 127:  # Unicode emoji check
			print(message)
		else:
			emoji_map = {
				"INFO": "ℹ️",
				"SUCCESS": "✅", 
				"ERROR": "❌",
				"WARNING": "⚠️",
				"EXEC": "🔧"
			}
			emoji = emoji_map.get(level, "📝")
			print(f"{emoji} {message}")
	
	async def run(self, max_steps: int = 50) -> AgentHistoryList:
		"""Execute agent until completion or max steps reached"""
		if self.started:
			raise RuntimeError('Agent has already been started. Create a new agent to run again.')
		
		self.started = True
		self.logger.info(f'⏺ Starting AI agent execution')
		self.logger.info(f'  ⎿ Task: {self.task}')
		self.logger.info(f'  ⎿ Max steps: {max_steps}')
		self.logger.info(f'  ⎿ Agent settings: {self.settings.__class__.__name__}')
		
		try:
			step_count = 0
			while step_count < max_steps:
				if self.state.stopped:
					self.logger.info('⏺ Agent execution stopped by user request')
					break
				
				self.logger.info(f'⏺ Step {step_count + 1}/{max_steps}: Beginning agent reasoning cycle')
				
				step_info = AgentStepInfo(step_number=step_count, max_steps=max_steps)
				
				await self.step(step_info)
				step_count += 1
				
				if self.state.last_model_output and self.state.last_model_output.is_done:
					self.logger.info(f'⏺ Task completed successfully in {step_count} steps')
					self.logger.info(f'  ⎿ Result: {self.state.last_model_output.result if hasattr(self.state.last_model_output, "result") else "Task marked as done"}')
					break
			else:
				self.logger.warning(f'⏺ Reached maximum steps ({max_steps}) without task completion')
		
		except Exception as e:
			self.logger.error(f'⏺ Agent execution failed: {type(e).__name__}: {e}')
			raise
		finally:
			await self._cleanup()
		
		return self.history
	
	async def step(self, step_info: AgentStepInfo | None = None) -> None:
		"""Execute a single agent step"""
		self.step_start_time = time.time()
		context_state = None
		
		try:
			# Phase 1: Prepare execution context
			self.logger.info('⏺ Preparing execution context for reasoning')
			context_state = await self._prepare_execution_context(step_info)
			self.logger.info('  ⎿ Context prepared successfully')
			
			# Phase 2: Get model output and execute actions
			self.logger.info('⏺ Querying LLM for next actions')
			await self._get_next_action(context_state, step_info)
			self.logger.info('  ⎿ LLM response received, parsing actions')
			
			self.logger.info('⏺ Executing planned actions')
			await self._execute_actions()
			self.logger.info('  ⎿ Action execution completed')
			
			# Phase 3: Post-processing
			self.logger.info('⏺ Post-processing step results')
			await self._post_process()
			self.logger.info('  ⎿ Step completed successfully')
			
		except Exception as e:
			self.logger.error(f'⏺ Step execution failed: {type(e).__name__}: {e}')
			await self._handle_step_error(e)
		finally:
			step_duration = time.time() - self.step_start_time
			self.logger.info(f'  ⎿ Step duration: {step_duration:.2f}s')
			await self._finalize_step(context_state, step_info)
	
	async def _prepare_execution_context(self, step_info: AgentStepInfo | None = None) -> State | None:
		"""Prepare the execution context for the current step"""
		if not self.execution_context:
			return None
		
		self.logger.debug(f'🌐 Step {self.state.n_steps}: Getting context state...')
		
		try:
			context_state = await self.execution_context.get_current_state(
				cache_elements=True,
				include_media=self.settings.use_vision
			)
			
			# Update action models if needed
			if self.controller:
				current_actions = self.controller.registry.get_prompt_description()
				if current_actions != self._last_actions:
					self._last_actions = current_actions
					self._update_action_models()
			
			# Add state message to message manager
			custom_providers = self.execution_context.get_context_providers(
				context_state=context_state,
				step_info=step_info,
			)
			
			self.msg_manager.add_state_message(
				context_state=context_state,
				model_output=self.state.last_model_output,
				result=self.state.last_result,
				step_info=step_info,
				use_vision=self.settings.use_vision,
				page_filtered_actions=self._last_actions,
				custom_context_providers=custom_providers,
			)
			
			return context_state
			
		except Exception as e:
			self.logger.error(f'Failed to prepare execution context: {e}')
			raise
	
	async def _get_next_action(self, context_state: State | None, step_info: AgentStepInfo | None) -> None:
		"""Get next action from LLM"""
		# Check for pause
		if self.state.paused:
			self.logger.info('⏺ Agent execution paused')
			return
		
		# Get messages
		input_messages = self.msg_manager.get_messages()
		self.logger.info(f'  ⎿ Sending {len(input_messages)} messages to LLM')
		
		# Invoke LLM
		llm_start = time.time()
		model_output_completion = await self._invoke_llm(input_messages, step_info)
		llm_duration = time.time() - llm_start
		
		if not model_output_completion.completion:
			raise ValueError('Model output is empty')
		
		# Clean logging of LLM output
		num_actions = len(model_output_completion.completion.action) if hasattr(model_output_completion.completion, 'action') else 0
		is_done = getattr(model_output_completion.completion, 'is_done', False)
		
		self.logger.info(f'  ⎿ LLM responded in {llm_duration:.2f}s: {num_actions} actions, done={is_done}')
		
		# Show thinking/reasoning if available
		if hasattr(model_output_completion.completion, 'current_state') and model_output_completion.completion.current_state:
			current_state = model_output_completion.completion.current_state
			
			if hasattr(current_state, 'thinking') and current_state.thinking:
				self.logger.info(f'⏺ Agent reasoning process')
				self.logger.info(f'  ⎿ {current_state.thinking}')
			
			if hasattr(current_state, 'next_goal') and current_state.next_goal:
				self.logger.info(f'⏺ Planning next actions')  
				self.logger.info(f'  ⎿ Goal: {current_state.next_goal}')
		
		# Show actions planned
		if hasattr(model_output_completion.completion, 'action') and model_output_completion.completion.action:
			self.logger.info(f'⏺ LLM planned {num_actions} actions')
			for i, action in enumerate(model_output_completion.completion.action):
				action_data = action.model_dump(exclude_unset=True)
				action_name = next(iter(action_data.keys())) if action_data else 'unknown'
				self.logger.info(f'  ⎿ Action {i+1}: {action_name}')
		
		self.state.last_model_output = model_output_completion.completion
		
		# Save conversation if configured
		if self.settings.save_conversation_path:
			await save_conversation(
				input_messages,
				model_output_completion.completion,
				self.settings.save_conversation_path,
				self.settings.save_conversation_path_encoding,
			)
		
		# Add assistant message
		self.msg_manager.add_assistant_message(model_output_completion.completion)
		
		# Update token usage
		if self.token_cost_service and model_output_completion.usage:
			self.token_cost_service.update_usage(
				model_output_completion.usage,
				self.llm.name,
			)
	
	async def _invoke_llm(
		self,
		input_messages: list[Any],
		step_info: AgentStepInfo | None = None,
	) -> ChatInvokeCompletion[AgentOutput | AgentStructuredOutputT]:
		"""Invoke the LLM with retry logic"""
		retry_count = 0
		max_retries = 3
		
		while retry_count <= max_retries:
			try:
				# Determine output type
				if self.structured_output_type:
					output_type = AgentStructuredOutput[self.structured_output_type]
				else:
					output_type = AgentOutput
				
				# Call LLM
				completion = await self.llm.ainvoke(
					input_messages,
					output_format=output_type,
				)
				
				return completion
				
			except ModelRateLimitError as e:
				retry_count += 1
				self.logger.error(f'Rate limit error (attempt {retry_count}/{max_retries + 1}): {e}')
				self.logger.error(f'Full exception details: {type(e).__name__}: {str(e)}')
				
				if retry_count > max_retries:
					self.logger.error(f'Max retries ({max_retries}) exceeded for rate limit')
					raise
				
				wait_time = min(60, 10 * retry_count)
				self.logger.warning(f'Rate limit hit, retrying in {wait_time}s... ({retry_count}/{max_retries})')
				await asyncio.sleep(wait_time)
				
			except ModelProviderError as e:
				retry_count += 1
				self.logger.error(f'Model provider error (attempt {retry_count}/{max_retries + 1}): {e}')
				self.logger.error(f'Full exception details: {type(e).__name__}: {str(e)}')
				
				if retry_count > max_retries:
					self.logger.error(f'Max retries ({max_retries}) exceeded for model provider error')
					raise
				
				wait_time = min(30, 5 * retry_count)
				self.logger.warning(f'Model provider error, retrying in {wait_time}s... ({retry_count}/{max_retries})')
				await asyncio.sleep(wait_time)
				
			except ModelError as e:
				retry_count += 1
				self.logger.error(f'Model error (attempt {retry_count}/{max_retries + 1}): {e}')
				self.logger.error(f'Full exception details: {type(e).__name__}: {str(e)}')
				
				if retry_count > max_retries:
					self.logger.error(f'Max retries ({max_retries}) exceeded for model error')
					raise
				
				wait_time = min(30, 5 * retry_count)
				self.logger.warning(f'Model error, retrying in {wait_time}s... ({retry_count}/{max_retries})')
				await asyncio.sleep(wait_time)
				
			except ValidationError as e:
				self.logger.error(f'Validation error: {e}')
				self.logger.error(f'Full exception details: {type(e).__name__}: {str(e)}')
				raise
			except Exception as e:
				self.logger.error(f'Unexpected error during LLM invocation: {e}')
				self.logger.error(f'Full exception details: {type(e).__name__}: {str(e)}')
				import traceback
				self.logger.error(f'Traceback: {traceback.format_exc()}')
				raise
	
	async def _execute_actions(self) -> None:
		"""Execute the actions from model output"""
		if not self.state.last_model_output:
			return
		
		# Check if last_model_output has the expected structure
		if not hasattr(self.state.last_model_output, 'action'):
			self.logger.error(f"Model output has no 'action' attribute. Type: {type(self.state.last_model_output)}")
			self.logger.error(f"Model output content: {self.state.last_model_output}")
			return
		
		if not self.state.last_model_output.action:
			return
		
		# Filter out empty ActionModels
		valid_actions = []
		for action in self.state.last_model_output.action:
			action_dict = action.model_dump(exclude_unset=True)
			if action_dict:  # Only include non-empty actions
				valid_actions.append(action)
		
		if not valid_actions:
			return
		
		if self.state.paused or self.state.stopped:
			return
		
		# Execute actions
		self.logger.info(f'  ⎿ Found {len(valid_actions)} valid actions to execute')
		results = await self.multi_act(valid_actions)
		self.state.last_result = results
		self.state.consecutive_failures = 0
		self.logger.info(f'  ⎿ All actions executed, {sum(1 for r in results if r.success)} succeeded')
	
	async def multi_act(self, actions: list[ActionModel]) -> list[ActionResult]:
		"""Execute multiple actions in sequence"""
		if not self.controller:
			return [ActionResult(success=False, error='No controller available')]
		
		results = []
		
		for i, action in enumerate(actions):
			if self.state.stopped:
				break
			
			# Clean action name extraction
			action_data = action.model_dump(exclude_unset=True)
			action_name = next(iter(action_data.keys()), 'unknown') if action_data else 'unknown'
			action_params = action_data.get(action_name, {}) if action_data else {}
			
			self.logger.info(f'⏺ Execute action {i+1}/{len(actions)}: {action_name}')
			if action_params:
				# Log parameters in a clean format
				param_summary = ', '.join(f'{k}={str(v)[:50]}{"..." if len(str(v)) > 50 else ""}' 
					for k, v in action_params.items())
				self.logger.info(f'  ⎿ Parameters: {param_summary}')
			
			try:
				# Check for index changes between actions
				if i > 0 and hasattr(action, 'get_index'):
					await self._check_index_changes(action)
				
				# Execute action
				action_start = time.time()
				result = await self.controller.act(action, file_system=self.file_system)
				action_duration = time.time() - action_start
				
				if result.success:
					content = result.extracted_content or 'completed'
					self.logger.info(f'  ⎿ Action succeeded in {action_duration:.2f}s: {content[:100]}{"..." if len(content) > 100 else ""}')
				else:
					error = result.error or 'unknown error'
					self.logger.error(f'  ⎿ Action failed in {action_duration:.2f}s: {error}')
				
				results.append(result)
				
				# Check if we should stop
				if result.is_done or (hasattr(result, 'stop_execution') and result.stop_execution):
					self.logger.info("⏺ Task completion detected")
					self.logger.info("  ⎿ Agent will finish after this step")
					# Also mark the model output as done to prevent repeat steps
					if self.state.last_model_output:
						self.state.last_model_output.is_done = True
					break
					
			except Exception as e:
				error_msg = AgentError.format_error(e, action)
				self.logger.error(f'  ⎿ Action execution error: {type(e).__name__}: {str(e)[:100]}{"..." if len(str(e)) > 100 else ""}')
				results.append(ActionResult(success=False, error=error_msg))
				
				# Decide whether to continue after error
				if not self._should_continue_after_error(e):
					break
		
		return results
	
	async def _check_index_changes(self, action: ActionModel) -> None:
		"""Check if indices have changed between actions"""
		# This is a placeholder - specific implementations can override
		pass
	
	def _should_continue_after_error(self, error: Exception) -> bool:
		"""Determine if execution should continue after an error"""
		# Stop on critical errors
		critical_errors = (
			ValidationError,
			ModelProviderError,
			ModelError,
		)
		return not isinstance(error, critical_errors)
	
	async def _post_process(self) -> None:
		"""Post-process after action execution"""
		# Update consecutive failures
		if self.state.last_result:
			if any(not r.success for r in self.state.last_result):
				self.state.consecutive_failures += 1
			else:
				self.state.consecutive_failures = 0
	
	async def _handle_step_error(self, error: Exception) -> None:
		"""Handle errors during step execution"""
		self.logger.error(f'Step failed: {error}')
		self.state.consecutive_failures += 1
		
		# Check if we should stop
		if self.state.consecutive_failures >= self.settings.max_failures:
			self.logger.error(f'Max failures ({self.settings.max_failures}) reached, stopping agent')
			self.state.stopped = True
	
	async def _finalize_step(self, context_state: State | None, step_info: AgentStepInfo | None) -> None:
		"""Finalize the step"""
		# Calculate step duration
		duration = time.time() - self.step_start_time if self.step_start_time else 0
		
		# Create step metadata
		metadata = StepMetadata(
			step_number=step_info.step_number if step_info else self.state.n_steps,
			timestamp=self.step_start_time or time.time(),
			duration=duration,
			token_usage=self.token_cost_service.get_current_usage() if self.token_cost_service else None,
		)
		
		# Create history entry
		history_state = ContextStateHistory(
			context_type=type(context_state).__name__ if context_state else 'Unknown',
			context_data=context_state.model_dump() if context_state else {},
			attachments={},
		)
		
		history_item = AgentHistory(
			model_output=self.state.last_model_output,
			result=self.state.last_result or [],
			state=history_state,
			metadata=metadata,
		)
		
		self.history.append(history_item)
		self.state.n_steps += 1
		
		# Log step summary
		self._log_step_summary(step_info, duration)
	
	def _log_step_summary(self, step_info: AgentStepInfo | None, duration: float) -> None:
		"""Log a summary of the step"""
		step_num = step_info.step_number if step_info else self.state.n_steps
		actions_taken = len(self.state.last_model_output.action) if self.state.last_model_output else 0
		
		self.logger.info(
			f'📊 Step {step_num} completed in {duration:.2f}s | '
			f'Actions: {actions_taken} | '
			f'Failures: {self.state.consecutive_failures}'
		)
	
	def _update_action_models(self) -> None:
		"""Update action models from controller registry"""
		if not self.controller:
			return
		
		self._action_models = self.controller.registry.action_models.copy()
	
	async def _cleanup(self) -> None:
		"""Cleanup resources"""
		self.logger.info('🧹 Cleaning up agent resources...')
		
		# Save final conversation if configured
		if self.settings.save_conversation_path and self.msg_manager.last_input_messages:
			await save_conversation(
				self.msg_manager.last_input_messages,
				self.state.last_model_output or AgentOutput(is_done=True, action=[], current_state=None),
				self.settings.save_conversation_path,
				self.settings.save_conversation_path_encoding,
			)
		
		# Close token cost service
		if self.token_cost_service:
			final_cost = self.token_cost_service.get_current_cost()
			self.logger.info(f'💰 Total cost: ${final_cost:.4f}')
	
	# Control methods
	def pause(self) -> None:
		"""Pause the agent"""
		self.state.paused = True
		self.logger.info('⏸️ Agent paused')
	
	def resume(self) -> None:
		"""Resume the agent"""
		self.state.paused = False
		self.logger.info('▶️ Agent resumed')
	
	def stop(self) -> None:
		"""Stop the agent"""
		self.state.stopped = True
		self.logger.info('🛑 Agent stopped')