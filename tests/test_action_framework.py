"""
Comprehensive integration tests for the action framework.

These tests validate the entire action framework including:
- Action creation and registration
- Parameter validation
- Result handling
- Category management
- Documentation generation
- Testing utilities
"""

import asyncio
import pytest
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from ai_agent.actions.base import ActionParameter, BaseAction, ActionContext
from ai_agent.actions.categories import ActionCategory, ActionTag, CategoryManager
from ai_agent.actions.documentation import ActionDocGenerator
from ai_agent.registry import ActionRegistry
from ai_agent.actions.results import EnhancedActionResult
from ai_agent.actions.testing import (
	ActionTestHarness,
	ActionTestCase,
	MockActionContext,
	create_parameter_validation_tests,
	run_action_test_suite
)
from ai_agent.actions.validation import ValidationError, RangeRule, LengthRule


# Test action for integration testing
class TestActionParameters(ActionParameter):
	"""Test parameters with various validation rules"""
	
	name: str = Field(
		description="Name parameter with length constraints",
		min_length=2,
		max_length=50
	)
	
	value: int = Field(
		description="Numeric value with range constraints",
		ge=0,
		le=100
	)
	
	optional_text: str | None = Field(
		default=None,
		description="Optional text parameter"
	)


class TestActionContext(ActionContext):
	"""Test context for integration testing"""
	
	def __init__(self, **kwargs):
		super().__init__(context_type="test", **kwargs)
		self.capabilities.update({"test_capability"})
		self.resources["test_resource"] = "test_value"


class TestAction(BaseAction[TestActionParameters, EnhancedActionResult, TestActionContext]):
	"""Test action for integration testing"""
	
	name: ClassVar[str] = "test_action"
	description: ClassVar[str] = "A test action for integration testing"
	category: ClassVar[str] = ActionCategory.UTILITY
	tags: ClassVar[set[str]] = {ActionTag.SAFE, ActionTag.STATELESS}
	
	async def execute(
		self,
		parameters: TestActionParameters,
		context: TestActionContext,
		**kwargs
	) -> EnhancedActionResult:
		"""Execute the test action"""
		# Simulate some processing
		result_data = {
			"processed_name": parameters.name.upper(),
			"doubled_value": parameters.value * 2,
			"has_optional_text": parameters.optional_text is not None
		}
		
		if parameters.optional_text:
			result_data["optional_text_length"] = len(parameters.optional_text)
		
		return EnhancedActionResult.success_with_data(
			data=result_data,
			result_type="test_result",
			summary=f"Successfully processed {parameters.name} with value {parameters.value}"
		)


class TestActionFramework:
	"""Integration tests for the complete action framework"""
	
	def setup_method(self):
		"""Set up test fixtures"""
		self.action = TestAction()
		self.context = TestActionContext()
		self.registry = ActionRegistry()
		self.category_manager = CategoryManager()
		self.doc_generator = ActionDocGenerator()
	
	@pytest.mark.asyncio
	async def test_action_lifecycle(self):
		"""Test complete action lifecycle from creation to execution"""
		# Create parameters
		params = {
			"name": "test",
			"value": 42,
			"optional_text": "hello world"
		}
		
		# Execute action
		result = await self.action.execute_with_lifecycle(params, self.context)
		
		# Verify result
		assert result.success is True
		assert result.data["processed_name"] == "TEST"
		assert result.data["doubled_value"] == 84
		assert result.data["has_optional_text"] is True
		assert result.data["optional_text_length"] == 11
	
	@pytest.mark.asyncio
	async def test_parameter_validation(self):
		"""Test parameter validation with various invalid inputs"""
		# Test empty name (violates min_length)
		with pytest.raises(ValueError):
			self.action.validate_parameters({"name": "", "value": 50})
		
		# Test name too long (violates max_length)
		with pytest.raises(ValueError):
			self.action.validate_parameters({
				"name": "a" * 51,
				"value": 50
			})
		
		# Test negative value (violates ge constraint)
		with pytest.raises(ValueError):
			self.action.validate_parameters({"name": "test", "value": -1})
		
		# Test value too large (violates le constraint)
		with pytest.raises(ValueError):
			self.action.validate_parameters({"name": "test", "value": 101})
		
		# Test valid parameters
		validated = self.action.validate_parameters({
			"name": "valid",
			"value": 50,
			"optional_text": "optional"
		})
		assert validated.name == "valid"
		assert validated.value == 50
		assert validated.optional_text == "optional"
	
	def test_action_registration(self):
		"""Test action registration in the registry"""
		# Register action
		registered = self.registry.register_action(self.action)
		
		# Verify registration
		assert registered.name == "test_action"
		assert registered.description == "A test action for integration testing"
		assert self.action.name in self.registry.actions
		
		# Test retrieval
		retrieved = self.registry.get_action_by_name("test_action")
		assert retrieved is not None
		assert retrieved.name == "test_action"
	
	def test_category_management(self):
		"""Test action categorization and discovery"""
		# Register action with category manager
		self.category_manager.register_action(self.action)
		
		# Test category discovery
		utility_actions = self.category_manager.find_actions_by_category(ActionCategory.UTILITY)
		assert "TestAction" in utility_actions
		
		# Test tag discovery
		safe_actions = self.category_manager.find_actions_by_tags({ActionTag.SAFE})
		assert "TestAction" in safe_actions
		
		# Test capability discovery
		# (This would work if we had actions with specific capabilities)
	
	def test_documentation_generation(self):
		"""Test automatic documentation generation"""
		# Generate documentation
		doc = self.doc_generator.generate_documentation(self.action)
		
		# Verify documentation content
		assert doc.name == "test_action"
		assert doc.description == "A test action for integration testing"
		assert doc.category == ActionCategory.UTILITY
		assert ActionTag.SAFE.value in doc.tags
		
		# Verify parameter documentation
		assert len(doc.parameters) == 3  # name, value, optional_text
		
		name_param = next((p for p in doc.parameters if p.name == "name"), None)
		assert name_param is not None
		assert name_param.required is True
		assert name_param.type == "string"
		
		value_param = next((p for p in doc.parameters if p.name == "value"), None)
		assert value_param is not None
		assert value_param.required is True
		assert value_param.type == "integer"
		
		optional_param = next((p for p in doc.parameters if p.name == "optional_text"), None)
		assert optional_param is not None
		assert optional_param.required is False
	
	@pytest.mark.asyncio
	async def test_action_test_harness(self):
		"""Test the action testing framework"""
		# Create test cases
		test_cases = [
			ActionTestCase(
				name="valid_execution",
				description="Test with valid parameters",
				parameters={"name": "test", "value": 42},
				expected_success=True,
				tags=["positive"]
			),
			ActionTestCase(
				name="invalid_name_empty",
				description="Test with empty name",
				parameters={"name": "", "value": 42},
				expected_success=False,
				tags=["negative", "validation"]
			),
			ActionTestCase(
				name="invalid_value_negative",
				description="Test with negative value",
				parameters={"name": "test", "value": -1},
				expected_success=False,
				tags=["negative", "validation"]
			)
		]
		
		# Run test suite
		result = await run_action_test_suite(
			self.action,
			test_cases,
			lambda: TestActionContext()
		)
		
		# Verify test results
		summary = result["summary"]
		assert summary["total_tests"] == 3
		assert summary["successful_tests"] == 1  # Only the valid test should pass
		assert summary["failed_tests"] == 2
		assert summary["success_rate"] == 1/3
		
		# Check specific test results
		reports = result["reports"]
		valid_report = next((r for r in reports if r.test_name == "valid_execution"), None)
		assert valid_report is not None
		assert valid_report.success is True
		assert valid_report.actual_result.success is True
	
	def test_validation_rules(self):
		"""Test custom validation rules"""
		# Test range rule
		range_rule = RangeRule(min_value=0, max_value=100)
		
		# Valid values
		assert range_rule.validate(50) is True
		assert range_rule.validate(0) is True
		assert range_rule.validate(100) is True
		
		# Invalid values
		assert range_rule.validate(-1) is False
		assert range_rule.validate(101) is False
		
		# Test length rule
		length_rule = LengthRule(min_length=2, max_length=10)
		
		# Valid lengths
		assert length_rule.validate("ab") is True
		assert length_rule.validate("hello") is True
		assert length_rule.validate("1234567890") is True
		
		# Invalid lengths
		assert length_rule.validate("a") is False
		assert length_rule.validate("12345678901") is False
	
	def test_result_formatting(self):
		"""Test result formatting for different outputs"""
		# Create a test result
		result = EnhancedActionResult.success_with_data(
			data={"key": "value"},
			result_type="test",
			summary="Test result"
		)
		
		# Test LLM formatting
		from ai_agent.actions.results import ResultFormatter
		llm_format = ResultFormatter.format_for_llm(result)
		
		assert "Status: SUCCESS" in llm_format
		assert "Summary: Test result" in llm_format
		assert "Data:" in llm_format
		
		# Test human formatting
		human_format = ResultFormatter.format_for_human(result)
		
		assert "✅ Action completed successfully" in human_format
		assert "📋 Test result" in human_format
		
		# Test JSON formatting
		json_format = ResultFormatter.format_for_json(result)
		import json
		parsed = json.loads(json_format)
		assert parsed["success"] is True
		assert parsed["data"]["key"] == "value"
	
	@pytest.mark.asyncio
	async def test_error_handling(self):
		"""Test comprehensive error handling"""
		# Create an action that always fails
		class FailingAction(BaseAction[TestActionParameters, EnhancedActionResult, TestActionContext]):
			name: ClassVar[str] = "failing_action"
			description: ClassVar[str] = "An action that always fails"
			
			async def execute(self, parameters, context, **kwargs):
				raise ValueError("This action always fails")
		
		failing_action = FailingAction()
		
		# Execute and verify error handling
		result = await failing_action.execute_with_lifecycle(
			{"name": "test", "value": 42},
			self.context
		)
		
		assert result.success is False
		assert "This action always fails" in result.error_message
		assert result.status.value == "failed"
	
	def test_prompt_generation(self):
		"""Test LLM prompt generation"""
		# Generate prompt description
		prompt = self.action.get_prompt_description()
		
		assert "A test action for integration testing" in prompt
		assert "Parameters:" in prompt
		
		# Test registry prompt generation
		self.registry.register_action(self.action)
		registry_prompt = self.registry.get_prompt_description()
		
		assert "test_action" in registry_prompt
		assert "A test action for integration testing" in registry_prompt
	
	def test_registry_statistics(self):
		"""Test registry statistics and analytics"""
		# Register action and simulate some usage
		registered = self.registry.register_action(self.action)
		
		# Simulate execution statistics
		registered.update_execution_stats(success=True, execution_time=0.1)
		registered.update_execution_stats(success=True, execution_time=0.2)
		registered.update_execution_stats(success=False, execution_time=0.15)
		
		# Get statistics
		stats = self.registry.get_registry_statistics()
		
		assert stats["registry_info"]["total_actions"] == 1
		assert stats["registry_info"]["total_executions"] == 3
		
		# Check action statistics
		assert registered.execution_count == 3
		assert registered.success_count == 2
		assert registered.failure_count == 1
		assert registered.get_success_rate() == 200/3  # 66.67%
		assert registered.get_average_execution_time() == 0.15
	
	@pytest.mark.asyncio
	async def test_complete_workflow(self):
		"""Test a complete workflow using the action framework"""
		# 1. Register action
		self.registry.register_action(self.action)
		
		# 2. Generate documentation
		doc = self.doc_generator.generate_documentation(self.action)
		
		# 3. Create test cases
		test_cases = create_parameter_validation_tests(
			self.action,
			{"name": "valid", "value": 50},
			[
				{"name": "", "value": 50},  # Invalid name
				{"name": "valid", "value": -1},  # Invalid value
			]
		)
		
		# 4. Run tests
		test_result = await run_action_test_suite(
			self.action,
			test_cases,
			lambda: TestActionContext()
		)
		
		# 5. Verify everything works together
		assert self.registry.get_action_by_name("test_action") is not None
		assert doc.name == "test_action"
		assert test_result["summary"]["total_tests"] == 3
		assert test_result["summary"]["successful_tests"] == 1
		
		# 6. Generate final documentation
		markdown_docs = self.registry.generate_documentation("markdown")
		assert "test_action" in markdown_docs
		assert "A test action for integration testing" in markdown_docs


if __name__ == "__main__":
	# Run tests if executed directly
	pytest.main([__file__, "-v"])