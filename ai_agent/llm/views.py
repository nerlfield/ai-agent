from typing import Generic, TypeVar, Union

from pydantic import BaseModel

# Re-export LLMType for backward compatibility
from ai_agent.llm.types import LLMType

T = TypeVar('T', bound=Union[BaseModel, str])


class ChatInvokeUsage(BaseModel):
	prompt_tokens: int
	prompt_cached_tokens: int | None
	prompt_cache_creation_tokens: int | None
	prompt_image_tokens: int | None
	completion_tokens: int
	total_tokens: int


class ChatInvokeCompletion(BaseModel, Generic[T]):
	completion: T
	thinking: str | None = None
	redacted_thinking: str | None = None
	usage: ChatInvokeUsage | None


__all__ = [
	'LLMType',
	'ChatInvokeUsage',
	'ChatInvokeCompletion',
]