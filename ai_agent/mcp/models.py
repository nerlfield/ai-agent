"""
MCP (Model Context Protocol) data models and schemas.

This module defines the core data structures for representing MCP tools,
parameters, and results, providing type-safe interfaces for the MCP adapter system.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid_extensions import uuid7str

from pydantic import BaseModel, ConfigDict, Field, validator


class MCPDataType(str, Enum):
	"""MCP parameter data types"""
	STRING = "string"
	NUMBER = "number"
	INTEGER = "integer"
	BOOLEAN = "boolean"
	ARRAY = "array"
	OBJECT = "object"
	NULL = "null"


class MCPParameterConstraints(BaseModel):
	"""Constraints for MCP parameters"""
	model_config = ConfigDict(extra='forbid')
	
	# String constraints
	min_length: int | None = None
	max_length: int | None = None
	pattern: str | None = None
	
	# Numeric constraints  
	minimum: float | None = None
	maximum: float | None = None
	exclusive_minimum: float | None = None
	exclusive_maximum: float | None = None
	multiple_of: float | None = None
	
	# Array constraints
	min_items: int | None = None
	max_items: int | None = None
	unique_items: bool | None = None
	
	# Object constraints
	min_properties: int | None = None
	max_properties: int | None = None
	
	# Enum constraints
	enum: list[Any] | None = None


class MCPParameter(BaseModel):
	"""MCP tool parameter definition"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str
	type: MCPDataType
	description: str | None = None
	required: bool = True
	default: Any | None = None
	constraints: MCPParameterConstraints | None = None
	
	# JSON Schema extensions
	json_schema: dict[str, Any] | None = None
	
	@validator('default')
	def validate_default_type(cls, v, values):
		"""Validate that default value matches parameter type"""
		if v is None:
			return v
		
		param_type = values.get('type')
		if param_type == MCPDataType.STRING and not isinstance(v, str):
			raise ValueError(f'Default value must be string for parameter type {param_type}')
		elif param_type == MCPDataType.NUMBER and not isinstance(v, (int, float)):
			raise ValueError(f'Default value must be number for parameter type {param_type}')
		elif param_type == MCPDataType.INTEGER and not isinstance(v, int):
			raise ValueError(f'Default value must be integer for parameter type {param_type}')
		elif param_type == MCPDataType.BOOLEAN and not isinstance(v, bool):
			raise ValueError(f'Default value must be boolean for parameter type {param_type}')
		elif param_type == MCPDataType.ARRAY and not isinstance(v, list):
			raise ValueError(f'Default value must be array for parameter type {param_type}')
		elif param_type == MCPDataType.OBJECT and not isinstance(v, dict):
			raise ValueError(f'Default value must be object for parameter type {param_type}')
		
		return v


class MCPToolMetadata(BaseModel):
	"""Metadata about an MCP tool"""
	model_config = ConfigDict(extra='allow')  # Allow extra fields for extensibility
	
	name: str
	description: str
	version: str | None = None
	author: str | None = None
	homepage: str | None = None
	documentation: str | None = None
	categories: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)
	
	# Execution constraints
	timeout_ms: int | None = None
	max_retries: int = 3
	requires_auth: bool = False
	
	# Resource usage hints
	estimated_duration_ms: int | None = None
	memory_usage_mb: int | None = None
	
	# Compatibility info
	mcp_version: str = "1.0"
	supported_features: list[str] = Field(default_factory=list)


class MCPTool(BaseModel):
	"""Complete MCP tool definition"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	parameters: list[MCPParameter] = Field(default_factory=list)
	metadata: MCPToolMetadata | None = None
	
	# Server connection info
	server_name: str | None = None
	server_url: str | None = None
	
	def get_required_parameters(self) -> list[MCPParameter]:
		"""Get list of required parameters"""
		return [param for param in self.parameters if param.required]
	
	def get_optional_parameters(self) -> list[MCPParameter]:
		"""Get list of optional parameters"""
		return [param for param in self.parameters if not param.required]
	
	def get_parameter_by_name(self, name: str) -> MCPParameter | None:
		"""Get parameter by name"""
		for param in self.parameters:
			if param.name == name:
				return param
		return None
	
	def validate_parameter_values(self, values: dict[str, Any]) -> list[str]:
		"""Validate parameter values against schema, return list of errors"""
		errors = []
		
		# Check required parameters
		for param in self.get_required_parameters():
			if param.name not in values:
				errors.append(f'Required parameter "{param.name}" is missing')
		
		# Validate provided values
		for name, value in values.items():
			param = self.get_parameter_by_name(name)
			if param is None:
				errors.append(f'Unknown parameter "{name}"')
				continue
			
			# Type validation
			type_errors = self._validate_parameter_type(param, value)
			errors.extend(type_errors)
			
			# Constraint validation
			if param.constraints:
				constraint_errors = self._validate_parameter_constraints(param, value)
				errors.extend(constraint_errors)
		
		return errors
	
	def _validate_parameter_type(self, param: MCPParameter, value: Any) -> list[str]:
		"""Validate parameter type"""
		errors = []
		
		if param.type == MCPDataType.STRING and not isinstance(value, str):
			errors.append(f'Parameter "{param.name}" must be a string')
		elif param.type == MCPDataType.NUMBER and not isinstance(value, (int, float)):
			errors.append(f'Parameter "{param.name}" must be a number')
		elif param.type == MCPDataType.INTEGER and not isinstance(value, int):
			errors.append(f'Parameter "{param.name}" must be an integer')
		elif param.type == MCPDataType.BOOLEAN and not isinstance(value, bool):
			errors.append(f'Parameter "{param.name}" must be a boolean')
		elif param.type == MCPDataType.ARRAY and not isinstance(value, list):
			errors.append(f'Parameter "{param.name}" must be an array')
		elif param.type == MCPDataType.OBJECT and not isinstance(value, dict):
			errors.append(f'Parameter "{param.name}" must be an object')
		
		return errors
	
	def _validate_parameter_constraints(self, param: MCPParameter, value: Any) -> list[str]:
		"""Validate parameter constraints"""
		errors = []
		constraints = param.constraints
		
		if param.type == MCPDataType.STRING and isinstance(value, str):
			if constraints.min_length is not None and len(value) < constraints.min_length:
				errors.append(f'Parameter "{param.name}" must be at least {constraints.min_length} characters')
			if constraints.max_length is not None and len(value) > constraints.max_length:
				errors.append(f'Parameter "{param.name}" must be at most {constraints.max_length} characters')
			if constraints.pattern is not None:
				import re
				if not re.match(constraints.pattern, value):
					errors.append(f'Parameter "{param.name}" does not match required pattern')
		
		elif param.type in (MCPDataType.NUMBER, MCPDataType.INTEGER) and isinstance(value, (int, float)):
			if constraints.minimum is not None and value < constraints.minimum:
				errors.append(f'Parameter "{param.name}" must be >= {constraints.minimum}')
			if constraints.maximum is not None and value > constraints.maximum:
				errors.append(f'Parameter "{param.name}" must be <= {constraints.maximum}')
			if constraints.exclusive_minimum is not None and value <= constraints.exclusive_minimum:
				errors.append(f'Parameter "{param.name}" must be > {constraints.exclusive_minimum}')
			if constraints.exclusive_maximum is not None and value >= constraints.exclusive_maximum:
				errors.append(f'Parameter "{param.name}" must be < {constraints.exclusive_maximum}')
			if constraints.multiple_of is not None and value % constraints.multiple_of != 0:
				errors.append(f'Parameter "{param.name}" must be a multiple of {constraints.multiple_of}')
		
		elif param.type == MCPDataType.ARRAY and isinstance(value, list):
			if constraints.min_items is not None and len(value) < constraints.min_items:
				errors.append(f'Parameter "{param.name}" must have at least {constraints.min_items} items')
			if constraints.max_items is not None and len(value) > constraints.max_items:
				errors.append(f'Parameter "{param.name}" must have at most {constraints.max_items} items')
			if constraints.unique_items and len(value) != len(set(str(item) for item in value)):
				errors.append(f'Parameter "{param.name}" must have unique items')
		
		elif param.type == MCPDataType.OBJECT and isinstance(value, dict):
			if constraints.min_properties is not None and len(value) < constraints.min_properties:
				errors.append(f'Parameter "{param.name}" must have at least {constraints.min_properties} properties')
			if constraints.max_properties is not None and len(value) > constraints.max_properties:
				errors.append(f'Parameter "{param.name}" must have at most {constraints.max_properties} properties')
		
		# Enum validation
		if constraints.enum is not None and value not in constraints.enum:
			errors.append(f'Parameter "{param.name}" must be one of: {constraints.enum}')
		
		return errors


class MCPResultStatus(str, Enum):
	"""MCP tool execution result status"""
	SUCCESS = "success"
	ERROR = "error" 
	TIMEOUT = "timeout"
	CANCELLED = "cancelled"


class MCPContent(BaseModel):
	"""MCP content block"""
	model_config = ConfigDict(extra='forbid')
	
	type: Literal["text", "image", "resource"]
	data: str | dict[str, Any]
	metadata: dict[str, Any] = Field(default_factory=dict)


class MCPResult(BaseModel):
	"""Result from MCP tool execution"""
	model_config = ConfigDict(extra='forbid')
	
	status: MCPResultStatus
	content: list[MCPContent] = Field(default_factory=list)
	
	# Result data
	value: Any | None = None
	error_message: str | None = None
	error_code: str | None = None
	
	# Execution metadata
	execution_time_ms: int | None = None
	server_name: str | None = None
	tool_name: str | None = None
	
	# Additional context
	metadata: dict[str, Any] = Field(default_factory=dict)
	
	@property
	def is_success(self) -> bool:
		"""Check if result indicates success"""
		return self.status == MCPResultStatus.SUCCESS
	
	@property
	def is_error(self) -> bool:
		"""Check if result indicates error"""
		return self.status == MCPResultStatus.ERROR
	
	@property
	def text_content(self) -> str:
		"""Get concatenated text content"""
		text_parts = []
		for content in self.content:
			if content.type == "text" and isinstance(content.data, str):
				text_parts.append(content.data)
		return "\n".join(text_parts)
	
	def add_text_content(self, text: str, metadata: dict[str, Any] | None = None) -> None:
		"""Add text content to result"""
		content = MCPContent(
			type="text",
			data=text,
			metadata=metadata or {}
		)
		self.content.append(content)
	
	def add_error(self, message: str, code: str | None = None) -> None:
		"""Add error information to result"""
		self.status = MCPResultStatus.ERROR
		self.error_message = message
		self.error_code = code


class MCPServerInfo(BaseModel):
	"""Information about an MCP server"""
	model_config = ConfigDict(extra='forbid')
	
	name: str
	version: str | None = None
	protocol_version: str = "1.0"
	
	# Connection info
	endpoint: str | None = None
	transport: Literal["stdio", "http", "ws"] = "stdio"
	
	# Capabilities
	supports_tools: bool = True
	supports_resources: bool = False
	supports_prompts: bool = False
	
	# Available tools
	tools: list[MCPTool] = Field(default_factory=list)
	
	# Server metadata
	description: str | None = None
	author: str | None = None
	homepage: str | None = None


__all__ = [
	'MCPDataType',
	'MCPParameterConstraints', 
	'MCPParameter',
	'MCPToolMetadata',
	'MCPTool',
	'MCPResultStatus',
	'MCPContent',
	'MCPResult',
	'MCPServerInfo',
]