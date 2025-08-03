from dataclasses import dataclass
from typing import TypeVar, overload

from pydantic import BaseModel

from ai_agent.llm.base import BaseChatModel
from ai_agent.llm.exceptions import ModelProviderError
from ai_agent.llm.messages import BaseMessage
from ai_agent.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatGroq(BaseChatModel):
	model: str
	api_key: str | None = None

	@property
	def provider(self) -> str:
		return 'groq'

	@property
	def name(self) -> str:
		return self.model

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		raise ModelProviderError("Groq provider not yet implemented", model=self.name)