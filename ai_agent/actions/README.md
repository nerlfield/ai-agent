# Base Action Framework

A comprehensive framework for creating, validating, and executing actions in the generic agent library. This framework provides type-safe, well-validated, discoverable, and self-documenting actions that are compatible with LLM tool calling.

## Key Features

- **Type-safe action definitions** with Pydantic validation
- **Comprehensive parameter validation** with custom rules
- **Enhanced result handling** with structured error reporting
- **Action categorization and discovery** system
- **Auto-generated documentation** optimized for LLMs
- **Complete testing utilities** for validation and integration tests
- **Plugin architecture** for extensible action systems

## Quick Start

### 1. Define Your Action

```python
from ai_agent.actions import BaseAction, ActionParameter, EnhancedActionResult, ActionContext
from pydantic import Field

class CalculateParameters(ActionParameter):
    """Parameters for calculation"""
    x: float = Field(description="First number")
    y: float = Field(description="Second number")
    operation: str = Field(description="Operation to perform", regex="^(add|subtract|multiply|divide)$")

class CalculateAction(BaseAction[CalculateParameters, EnhancedActionResult, ActionContext]):
    name = "calculate"
    description = "Perform basic mathematical operations"
    category = "utility"
    
    async def execute(self, parameters: CalculateParameters, context: ActionContext) -> EnhancedActionResult:
        if parameters.operation == "add":
            result = parameters.x + parameters.y
        elif parameters.operation == "subtract":
            result = parameters.x - parameters.y
        elif parameters.operation == "multiply":
            result = parameters.x * parameters.y
        elif parameters.operation == "divide":
            if parameters.y == 0:
                return EnhancedActionResult.error_with_details("Division by zero")
            result = parameters.x / parameters.y
        
        return EnhancedActionResult.success_with_data(
            data={"result": result},
            summary=f"Successfully calculated {parameters.x} {parameters.operation} {parameters.y} = {result}"
        )
```

### 2. Register and Use Your Action

```python
from ai_agent.actions import ActionRegistry

# Create registry and register action
registry = ActionRegistry()
registry.register_action(CalculateAction())

# Execute action
action = registry.get_action_by_name("calculate")
context = ActionContext(context_type="math")

result = await action.function({
    "x": 10,
    "y": 5,
    "operation": "multiply"
}, context)

print(result.data)  # {"result": 50}
```

### 3. Generate Documentation

```python
from ai_agent.actions import ActionDocGenerator

doc_generator = ActionDocGenerator()
documentation = doc_generator.generate_documentation(CalculateAction())

# Generate markdown documentation
markdown = doc_generator.generate_markdown_docs(documentation)
print(markdown)
```

### 4. Test Your Action

```python
from ai_agent.actions import ActionTestHarness, ActionTestCase

# Create test cases
test_cases = [
    ActionTestCase(
        name="test_addition",
        description="Test basic addition",
        parameters={"x": 5, "y": 3, "operation": "add"},
        expected_success=True,
        expected_data={"result": 8}
    ),
    ActionTestCase(
        name="test_division_by_zero",
        description="Test division by zero error",
        parameters={"x": 5, "y": 0, "operation": "divide"},
        expected_success=False
    )
]

# Run tests
harness = ActionTestHarness()
for test_case in test_cases:
    harness.add_test_case(test_case)

reports = await harness.run_tests(CalculateAction())
summary = harness.get_summary()
print(f"Success rate: {summary['success_rate']:.1%}")
```

## Framework Components

### Base Interfaces (`base.py`)
- `ActionProtocol` - Interface all actions must implement
- `BaseAction` - Base class with full lifecycle management
- `ActionParameter` - Base for parameter validation
- `ActionResult` - Base for result handling
- `ActionContext` - Base for execution context

### Parameter Validation (`validation.py`)
- `ValidationRule` - Custom validation rules
- `ParameterValidator` - Comprehensive validation system
- Pre-built validators for common patterns (URL, email, etc.)

### Result Handling (`results.py`)
- `EnhancedActionResult` - Advanced result with error details
- `ResultFormatter` - Format results for different outputs
- `ResultAnalyzer` - Analyze result patterns and performance

### Categorization (`categories.py`)
- `ActionCategory` - Standard action categories
- `ActionTag` - Fine-grained classification tags
- `CategoryManager` - Discovery and organization

### Documentation (`documentation.py`)
- `ActionDocGenerator` - Auto-generate comprehensive docs
- `ActionDocumentation` - Structured documentation model
- LLM-optimized descriptions and examples

### Testing (`testing.py`)
- `ActionTestHarness` - Comprehensive test runner
- `MockActionContext` - Mock contexts for testing
- `ActionTestCase` - Test case definitions
- Utilities for parameter validation and performance testing

### Registry (`registry.py`)
- `ActionRegistry` - Enhanced action management
- `RegisteredAction` - Action with metadata and statistics
- `ActionDiscoveryFilter` - Advanced action discovery
- Usage statistics and performance tracking

## Advanced Features

### Custom Validation Rules

```python
from ai_agent.actions.validation import ValidationRule, RangeRule, LengthRule

class PositiveNumberRule(ValidationRule[float]):
    def validate(self, value: float) -> bool:
        return value > 0

# Use in parameter definition
class MyParameters(ActionParameter):
    amount: float = Field(annotation=PositiveNumberRule())
```

### Action Discovery

```python
from ai_agent.actions import ActionDiscoveryFilter, ActionCategory, ActionTag

# Find actions by criteria
filter_criteria = ActionDiscoveryFilter(
    category=ActionCategory.DATA,
    tags={ActionTag.SAFE, ActionTag.READ_ONLY},
    min_success_rate=0.9
)

actions = registry.discover_actions(filter_criteria)
```

### Performance Analysis

```python
from ai_agent.actions.results import ResultAnalyzer

# Analyze action performance
results = [...]  # List of action results
analysis = ResultAnalyzer.analyze_performance(results)
print(f"Average duration: {analysis['avg_duration']:.3f}s")
print(f"Success rate: {analysis['success_rate']:.1%}")
```

## Example Actions

The framework includes example actions in the `examples/` directory:

- **File Actions** (`file_actions.py`) - File system operations
- **Web Actions** (`web_actions.py`) - Browser interactions  
- **Data Actions** (`data_actions.py`) - Data processing and transformation

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_action_framework.py -v
```

The test suite covers:
- Action lifecycle management
- Parameter validation
- Result handling
- Registry operations
- Documentation generation
- Testing utilities
- Integration scenarios

## Best Practices

1. **Always use type hints** for parameters, results, and context
2. **Validate parameters thoroughly** using Pydantic and custom rules
3. **Provide comprehensive error handling** with structured error information
4. **Document your actions** with clear descriptions and examples
5. **Test extensively** using the provided testing utilities
6. **Use appropriate categories and tags** for discoverability
7. **Follow the principle of least surprise** in parameter naming and behavior

## Integration with Browser-Use

This framework is designed to be a drop-in replacement for the existing action system in browser-use, while providing much more functionality and type safety. Actions created with this framework can be easily integrated into the existing agent architecture.