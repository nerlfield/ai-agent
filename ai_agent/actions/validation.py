from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import Field, field_validator
from pydantic.types import conint, constr

T = TypeVar('T')


class ValidationRule(ABC, Generic[T]):
	"""Base class for custom validation rules"""
	
	@abstractmethod
	def validate(self, value: T) -> T:
		"""Validate and potentially transform the value"""
		pass
	
	@abstractmethod
	def error_message(self) -> str:
		"""Get error message for validation failure"""
		pass


class RangeRule(ValidationRule[float]):
	"""Validate numeric values are within a range"""
	
	def __init__(self, min_value: float | None = None, max_value: float | None = None):
		self.min_value = min_value
		self.max_value = max_value
	
	def validate(self, value: float) -> float:
		if self.min_value is not None and value < self.min_value:
			raise ValueError(self.error_message())
		if self.max_value is not None and value > self.max_value:
			raise ValueError(self.error_message())
		return value
	
	def error_message(self) -> str:
		if self.min_value is not None and self.max_value is not None:
			return f'Value must be between {self.min_value} and {self.max_value}'
		elif self.min_value is not None:
			return f'Value must be at least {self.min_value}'
		else:
			return f'Value must be at most {self.max_value}'


class LengthRule(ValidationRule[str]):
	"""Validate string length"""
	
	def __init__(self, min_length: int | None = None, max_length: int | None = None):
		self.min_length = min_length
		self.max_length = max_length
	
	def validate(self, value: str) -> str:
		if self.min_length is not None and len(value) < self.min_length:
			raise ValueError(self.error_message())
		if self.max_length is not None and len(value) > self.max_length:
			raise ValueError(self.error_message())
		return value
	
	def error_message(self) -> str:
		if self.min_length is not None and self.max_length is not None:
			return f'Length must be between {self.min_length} and {self.max_length} characters'
		elif self.min_length is not None:
			return f'Length must be at least {self.min_length} characters'
		else:
			return f'Length must be at most {self.max_length} characters'


class RegexRule(ValidationRule[str]):
	"""Validate string matches a regex pattern"""
	
	def __init__(self, pattern: str, error_msg: str | None = None):
		self.pattern = pattern
		self.regex = re.compile(pattern)
		self.error_msg = error_msg
	
	def validate(self, value: str) -> str:
		if not self.regex.match(value):
			raise ValueError(self.error_message())
		return value
	
	def error_message(self) -> str:
		return self.error_msg or f'Value must match pattern: {self.pattern}'


class ParameterValidator:
	"""Utility class for common parameter validations"""
	
	@staticmethod
	def validate_url(url: str) -> str:
		"""Validate URL format"""
		url_pattern = re.compile(
			r'^https?://'  # http:// or https://
			r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
			r'localhost|'  # localhost...
			r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
			r'(?::\d+)?'  # optional port
			r'(?:/?|[/?]\S+)$', re.IGNORECASE
		)
		if not url_pattern.match(url):
			raise ValueError('Invalid URL format')
		return url
	
	@staticmethod
	def validate_email(email: str) -> str:
		"""Validate email format"""
		email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
		if not email_pattern.match(email):
			raise ValueError('Invalid email format')
		return email
	
	@staticmethod
	def validate_file_path(path: str, must_exist: bool = False) -> str:
		"""Validate file path"""
		if '..' in path:
			raise ValueError('Path traversal not allowed')
		if must_exist:
			from pathlib import Path
			if not Path(path).exists():
				raise ValueError(f'Path does not exist: {path}')
		return path


# Pre-configured field types for common use cases

def UrlField(description: str = 'Valid URL', **kwargs) -> Any:
	"""Field type for URLs with automatic validation"""
	return Field(
		description=description,
		**kwargs,
		json_schema_extra={
			'format': 'uri',
			'examples': ['https://example.com', 'http://localhost:8080/path'],
		}
	)


def EmailField(description: str = 'Valid email address', **kwargs) -> Any:
	"""Field type for emails with automatic validation"""
	return Field(
		description=description,
		**kwargs,
		json_schema_extra={
			'format': 'email',
			'examples': ['user@example.com'],
		}
	)


def PositiveNumberField(description: str = 'Positive number', **kwargs) -> Any:
	"""Field type for positive numbers"""
	return Field(description=description, gt=0, **kwargs)


def PercentageField(description: str = 'Percentage (0-100)', **kwargs) -> Any:
	"""Field type for percentages"""
	return Field(description=description, ge=0, le=100, **kwargs)


def NonEmptyStringField(description: str = 'Non-empty string', **kwargs) -> Any:
	"""Field type for non-empty strings"""
	return Field(description=description, min_length=1, **kwargs)