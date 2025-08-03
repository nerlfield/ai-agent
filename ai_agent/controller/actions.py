from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar('T', bound=BaseModel)


class DoneAction(BaseModel):
	"""Action to mark task as completed"""
	text: str | None = Field(None, description='Optional completion message')


class WaitAction(BaseModel):
	"""Action to wait for a specified duration"""
	seconds: float = Field(description='Number of seconds to wait', ge=0, le=300)


class StructuredOutputAction(BaseModel, Generic[T]):
	"""Action to complete task with structured output"""
	structured_output: T = Field(description='The structured output to return')