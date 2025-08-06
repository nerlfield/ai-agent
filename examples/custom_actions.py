#!/usr/bin/env python3
"""
Example showing how to create and register custom actions.

This demonstrates:
1. Creating custom action classes
2. Registering actions with the controller
3. Using custom context and capabilities
"""

import asyncio
import random
from datetime import datetime

from pydantic import Field

from ai_agent.actions.base import ActionContext, ActionParameter, BaseAction
from ai_agent.actions.categories import ActionCategory, ActionTag
from ai_agent.agent.context import SimpleExecutionContext
from ai_agent.agent.service import GenericAgent
from ai_agent.controller.service import Controller
from ai_agent.filesystem import FileSystem
from ai_agent.llm.factory import create_llm
from ai_agent.llm.views import LLMType
from ai_agent.common.models import ActionResult


# Custom action parameters
class RollDiceParams(ActionParameter):
	"""Parameters for rolling dice"""
	sides: int = Field(default=6, description='Number of sides on the die', ge=2, le=100)
	count: int = Field(default=1, description='Number of dice to roll', ge=1, le=10)


class GetTimeParams(ActionParameter):
	"""Parameters for getting current time"""
	format: str = Field(
		default='%Y-%m-%d %H:%M:%S',
		description='Time format string (Python strftime format)',
	)
	timezone: str = Field(default='local', description='Timezone (only "local" supported currently)')


class CalculateParams(ActionParameter):
	"""Parameters for simple calculations"""
	expression: str = Field(description='Mathematical expression to evaluate (basic operations only)')


# Custom actions
class RollDiceAction(BaseAction[RollDiceParams, ActionResult, dict]):
	"""Action to roll dice"""
	
	name = 'roll_dice'
	description = 'Roll one or more dice and get the results'
	category = ActionCategory.AUTOMATION
	tags = {ActionTag.SAFE, ActionTag.OFFLINE_CAPABLE, ActionTag.FAST}
	
	async def execute(self, parameters: RollDiceParams, context: ActionContext[dict]) -> ActionResult:
		rolls = [random.randint(1, parameters.sides) for _ in range(parameters.count)]
		total = sum(rolls)
		
		if parameters.count == 1:
			content = f"Rolled a d{parameters.sides}: {rolls[0]}"
		else:
			content = f"Rolled {parameters.count}d{parameters.sides}: {rolls} (Total: {total})"
		
		return ActionResult(
			success=True,
			extracted_content=content,
			output_data={'rolls': rolls, 'total': total, 'sides': parameters.sides},
		)


class GetTimeAction(BaseAction[GetTimeParams, ActionResult, dict]):
	"""Action to get current time"""
	
	name = 'get_time'
	description = 'Get the current date and time'
	category = ActionCategory.MONITORING
	tags = {ActionTag.SAFE, ActionTag.READ_ONLY, ActionTag.OFFLINE_CAPABLE, ActionTag.FAST}
	
	async def execute(self, parameters: GetTimeParams, context: ActionContext[dict]) -> ActionResult:
		try:
			current_time = datetime.now()
			formatted_time = current_time.strftime(parameters.format)
			
			return ActionResult(
				success=True,
				extracted_content=formatted_time,
				output_data={
					'formatted': formatted_time,
					'timestamp': current_time.timestamp(),
					'format': parameters.format,
				},
			)
		except ValueError as e:
			return ActionResult(
				success=False,
				error=f'Invalid time format: {str(e)}',
			)


class CalculateAction(BaseAction[CalculateParams, ActionResult, dict]):
	"""Action to perform simple calculations"""
	
	name = 'calculate'
	description = 'Perform simple mathematical calculations'
	category = ActionCategory.DATA_MANIPULATION
	tags = {ActionTag.SAFE, ActionTag.OFFLINE_CAPABLE, ActionTag.FAST}
	
	async def execute(self, parameters: CalculateParams, context: ActionContext[dict]) -> ActionResult:
		try:
			# Safety: only allow basic math operations
			allowed_chars = '0123456789+-*/()., '
			expression = parameters.expression.replace(' ', '')
			
			if not all(c in allowed_chars for c in expression):
				return ActionResult(
					success=False,
					error='Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed.',
				)
			
			# Evaluate the expression
			result = eval(expression)
			
			return ActionResult(
				success=True,
				extracted_content=f"{parameters.expression} = {result}",
				output_data={'expression': parameters.expression, 'result': result},
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Calculation failed: {str(e)}',
			)


def register_custom_actions(controller: Controller):
	"""Register custom actions with the controller"""
	
	# Roll dice
	@controller.registry.action(
		description='Roll dice and get random results',
		param_model=RollDiceAction().get_parameter_model(),
	)
	async def roll_dice(params):
		action = RollDiceAction()
		context = ActionContext(data={})
		return await action.execute(params, context)
	
	# Get time
	@controller.registry.action(
		description='Get the current date and time',
		param_model=GetTimeAction().get_parameter_model(),
	)
	async def get_time(params):
		action = GetTimeAction()
		context = ActionContext(data={})
		return await action.execute(params, context)
	
	# Calculate
	@controller.registry.action(
		description='Perform mathematical calculations',
		param_model=CalculateAction().get_parameter_model(),
	)
	async def calculate(params):
		action = CalculateAction()
		context = ActionContext(data={})
		return await action.execute(params, context)


async def main():
	"""Run example with custom actions"""
	# Set up components
	llm = create_llm(LLMType.ANTHROPIC, model='claude-3-5-sonnet-20241022')
	file_system = FileSystem()
	
	# Create controller and register custom actions
	controller = Controller()
	register_custom_actions(controller)
	
	# Also register file actions for saving results
	from ai_agent.registry_helper import register_file_actions
	register_file_actions(controller.registry, file_system)
	
	# Create execution context
	execution_context = SimpleExecutionContext(
		file_system=file_system,
		llm=llm,
		registry=controller.registry,
	)
	
	# Define a fun task using custom actions
	task = """
	Let's play a dice game! Please:
	1. Roll 3 six-sided dice and tell me the results
	2. Calculate the average of the rolls
	3. Get the current time and date
	4. Save a game summary to 'dice_game.txt' that includes:
	   - The dice rolls and total
	   - The calculated average
	   - The timestamp when the game was played
	   - Whether the total was above or below 10 (with appropriate celebration or commiseration)
	"""
	
	# Create and run agent
	agent = GenericAgent(
		task=task,
		llm=llm,
		execution_context=execution_context,
		controller=controller,
		file_system=file_system,
	)
	
	print("🎲 Starting dice game agent...")
	history = await agent.run(max_steps=10)
	
	# Check results
	if history.last and history.last.model_output and history.last.model_output.is_done:
		print("✅ Game completed! Check dice_game.txt for results.")
	else:
		print("❌ Game didn't complete fully.")


async def calculation_example():
	"""Example focused on calculations"""
	llm = create_llm(LLMType.ANTHROPIC, model='claude-3-5-sonnet-20241022')
	controller = Controller()
	register_custom_actions(controller)
	
	execution_context = SimpleExecutionContext(
		llm=llm,
		registry=controller.registry,
	)
	
	task = """
	Please help me with some calculations:
	1. Calculate 15 * 23
	2. Calculate (100 + 50) / 3
	3. Calculate 2 * 3 + 4 * 5
	Tell me all the results.
	"""
	
	agent = GenericAgent(
		task=task,
		llm=llm,
		execution_context=execution_context,
		controller=controller,
	)
	
	print("🧮 Running calculation agent...")
	await agent.run(max_steps=5)


if __name__ == "__main__":
	# Run the dice game example
	asyncio.run(main())
	
	# Or run the calculation example
	# asyncio.run(calculation_example())