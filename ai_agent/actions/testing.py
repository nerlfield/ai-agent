from __future__ import annotations

import asyncio
import time
from typing import Any, Type

from pydantic import BaseModel, Field

from ai_agent.actions.base import ActionContext, BaseAction
from ai_agent.registry.views import ActionResult


class MockActionContext(ActionContext):
	"""Mock context for testing actions"""
	
	def __init__(
		self,
		data: Any = None,
		capabilities: set[str] | None = None,
		metadata: dict[str, Any] | None = None,
	):
		super().__init__(
			data=data or {},
			capabilities=capabilities or set(),
			metadata=metadata or {},
		)


class ActionTestCase(BaseModel):
	"""Test case for an action"""
	
	name: str
	description: str | None = None
	parameters: dict[str, Any]
	context: dict[str, Any] = Field(default_factory=dict)
	expected_success: bool = True
	expected_error: str | None = None
	expected_result: dict[str, Any] | None = None
	timeout_seconds: float = 30.0


class ActionTestResult(BaseModel):
	"""Result of running an action test"""
	
	test_name: str
	passed: bool
	duration_ms: float
	error: str | None = None
	actual_result: ActionResult | None = None
	expected_result: dict[str, Any] | None = None


class ActionTestHarness:
	"""Test harness for running action tests"""
	
	def __init__(self, action_class: Type[BaseAction]):
		self.action_class = action_class
		self.action = action_class()
	
	async def run_test(self, test_case: ActionTestCase) -> ActionTestResult:
		"""Run a single test case"""
		start_time = time.time()
		
		try:
			# Create context
			context = MockActionContext(
				data=test_case.context.get('data'),
				capabilities=set(test_case.context.get('capabilities', [])),
				metadata=test_case.context.get('metadata', {}),
			)
			
			# Create parameters
			param_model = self.action.get_parameter_model()
			params = param_model(**test_case.parameters)
			
			# Run action with timeout
			result = await asyncio.wait_for(
				self.action.execute(params, context),
				timeout=test_case.timeout_seconds
			)
			
			duration_ms = (time.time() - start_time) * 1000
			
			# Check result
			if test_case.expected_success and not result.success:
				return ActionTestResult(
					test_name=test_case.name,
					passed=False,
					duration_ms=duration_ms,
					error=f'Expected success but got failure: {result.error}',
					actual_result=result,
				)
			
			if not test_case.expected_success and result.success:
				return ActionTestResult(
					test_name=test_case.name,
					passed=False,
					duration_ms=duration_ms,
					error='Expected failure but got success',
					actual_result=result,
				)
			
			if test_case.expected_error and result.error != test_case.expected_error:
				return ActionTestResult(
					test_name=test_case.name,
					passed=False,
					duration_ms=duration_ms,
					error=f'Expected error "{test_case.expected_error}" but got "{result.error}"',
					actual_result=result,
				)
			
			if test_case.expected_result:
				# Compare result fields
				for key, expected_value in test_case.expected_result.items():
					actual_value = getattr(result, key, None)
					if actual_value != expected_value:
						return ActionTestResult(
							test_name=test_case.name,
							passed=False,
							duration_ms=duration_ms,
							error=f'Expected {key}={expected_value} but got {key}={actual_value}',
							actual_result=result,
							expected_result=test_case.expected_result,
						)
			
			return ActionTestResult(
				test_name=test_case.name,
				passed=True,
				duration_ms=duration_ms,
				actual_result=result,
			)
			
		except asyncio.TimeoutError:
			duration_ms = (time.time() - start_time) * 1000
			return ActionTestResult(
				test_name=test_case.name,
				passed=False,
				duration_ms=duration_ms,
				error=f'Test timed out after {test_case.timeout_seconds} seconds',
			)
		except Exception as e:
			duration_ms = (time.time() - start_time) * 1000
			return ActionTestResult(
				test_name=test_case.name,
				passed=False,
				duration_ms=duration_ms,
				error=f'{type(e).__name__}: {str(e)}',
			)
	
	async def run_tests(self, test_cases: list[ActionTestCase]) -> list[ActionTestResult]:
		"""Run multiple test cases"""
		results = []
		for test_case in test_cases:
			result = await self.run_test(test_case)
			results.append(result)
		return results
	
	def validate_parameters(self, test_cases: list[ActionTestCase]) -> list[str]:
		"""Validate test case parameters without running tests"""
		errors = []
		param_model = self.action.get_parameter_model()
		
		for test_case in test_cases:
			try:
				param_model(**test_case.parameters)
			except Exception as e:
				errors.append(f'{test_case.name}: {str(e)}')
		
		return errors
	
	async def test_context_validation(self) -> bool:
		"""Test that the action properly validates context"""
		# Test with empty context
		empty_context = MockActionContext()
		if self.action.requires_capabilities:
			# Should fail validation
			if self.action.validate_context(empty_context):
				return False
		
		# Test with required capabilities
		full_context = MockActionContext(capabilities=self.action.requires_capabilities)
		return self.action.validate_context(full_context)
	
	async def performance_test(
		self,
		parameters: dict[str, Any],
		iterations: int = 100,
		context: dict[str, Any] | None = None,
	) -> dict[str, float]:
		"""Run performance tests"""
		context_obj = MockActionContext(**(context or {}))
		param_model = self.action.get_parameter_model()
		params = param_model(**parameters)
		
		durations = []
		for _ in range(iterations):
			start = time.time()
			await self.action.execute(params, context_obj)
			durations.append(time.time() - start)
		
		return {
			'min_ms': min(durations) * 1000,
			'max_ms': max(durations) * 1000,
			'avg_ms': (sum(durations) / len(durations)) * 1000,
			'p50_ms': sorted(durations)[len(durations) // 2] * 1000,
			'p95_ms': sorted(durations)[int(len(durations) * 0.95)] * 1000,
			'p99_ms': sorted(durations)[int(len(durations) * 0.99)] * 1000,
		}


class ActionTestGenerator:
	"""Generate test cases for actions"""
	
	@staticmethod
	def generate_parameter_tests(action: BaseAction) -> list[ActionTestCase]:
		"""Generate test cases for parameter validation"""
		param_model = action.get_parameter_model()
		schema = param_model.model_json_schema()
		properties = schema.get('properties', {})
		required = set(schema.get('required', []))
		
		test_cases = []
		
		# Test missing required parameters
		for field in required:
			params = {f: None for f in properties if f != field}
			test_cases.append(ActionTestCase(
				name=f'missing_required_{field}',
				description=f'Test missing required field: {field}',
				parameters=params,
				expected_success=False,
			))
		
		# Test invalid types
		for field, prop in properties.items():
			field_type = prop.get('type')
			if field_type == 'string':
				invalid_value = 123
			elif field_type == 'number':
				invalid_value = 'not a number'
			elif field_type == 'boolean':
				invalid_value = 'not a bool'
			elif field_type == 'array':
				invalid_value = 'not an array'
			else:
				continue
			
			params = {f: None for f in properties}
			params[field] = invalid_value
			
			test_cases.append(ActionTestCase(
				name=f'invalid_type_{field}',
				description=f'Test invalid type for field: {field}',
				parameters=params,
				expected_success=False,
			))
		
		return test_cases