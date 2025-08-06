import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from anthropic import (
	NOT_GIVEN,
	APIConnectionError,
	APIStatusError,
	AsyncAnthropic,
	NotGiven,
	RateLimitError,
)
from anthropic.types import CacheControlEphemeralParam, Message, ToolParam
from anthropic.types.model_param import ModelParam
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_choice_tool_param import ToolChoiceToolParam
from httpx import Timeout
from pydantic import BaseModel

from ai_agent.llm.anthropic.serializer import AnthropicMessageSerializer
from ai_agent.llm.base import BaseChatModel
from ai_agent.llm.exceptions import ModelProviderError, ModelRateLimitError
from ai_agent.llm.messages import BaseMessage
from ai_agent.llm.schema import SchemaOptimizer
from ai_agent.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatAnthropic(BaseChatModel):
	model: str | ModelParam
	max_tokens: int = 8192
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None

	api_key: str | None = None
	auth_token: str | None = None
	base_url: str | httpx.URL | None = None
	timeout: float | Timeout | None | NotGiven = NotGiven()
	max_retries: int = 10
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None

	@property
	def provider(self) -> str:
		return 'anthropic'

	def _get_client_params(self) -> dict[str, Any]:
		base_params = {
			'api_key': self.api_key,
			'auth_token': self.auth_token,
			'base_url': self.base_url,
			'timeout': self.timeout,
			'max_retries': self.max_retries,
			'default_headers': self.default_headers,
			'default_query': self.default_query,
		}

		client_params = {}
		for k, v in base_params.items():
			if v is not None and v is not NotGiven():
				client_params[k] = v

		return client_params

	def _get_client_params_for_invoke(self):
		client_params = {}

		if self.temperature is not None:
			client_params['temperature'] = self.temperature

		if self.max_tokens is not None:
			client_params['max_tokens'] = self.max_tokens

		if self.top_p is not None:
			client_params['top_p'] = self.top_p

		if self.seed is not None:
			client_params['seed'] = self.seed

		return client_params

	def get_client(self) -> AsyncAnthropic:
		client_params = self._get_client_params()
		return AsyncAnthropic(**client_params)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: Message) -> ChatInvokeUsage | None:
		usage = ChatInvokeUsage(
			prompt_tokens=response.usage.input_tokens
			+ (
				response.usage.cache_read_input_tokens or 0
			),
			completion_tokens=response.usage.output_tokens,
			total_tokens=response.usage.input_tokens + response.usage.output_tokens,
			prompt_cached_tokens=response.usage.cache_read_input_tokens,
			prompt_cache_creation_tokens=response.usage.cache_creation_input_tokens,
			prompt_image_tokens=None,
		)
		return usage

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		anthropic_messages, system_prompt = AnthropicMessageSerializer.serialize_messages(messages)

		try:
			if output_format is None:
				response = await self.get_client().messages.create(
					model=self.model,
					messages=anthropic_messages,
					system=system_prompt or NOT_GIVEN,
					**self._get_client_params_for_invoke(),
				)

				if not isinstance(response, Message):
					raise ModelProviderError(
						message=f'Unexpected response type from Anthropic API: {type(response).__name__}. Response: {str(response)[:200]}',
						status_code=502,
						model=self.name,
					)

				usage = self._get_usage(response)

				first_content = response.content[0]
				if isinstance(first_content, TextBlock):
					response_text = first_content.text
				else:
					response_text = str(first_content)

				return ChatInvokeCompletion(
					completion=response_text,
					usage=usage,
				)

			else:
				tool_name = output_format.__name__
				schema = SchemaOptimizer.create_optimized_json_schema(output_format)

				if 'title' in schema:
					del schema['title']

				tool = ToolParam(
					name=tool_name,
					description=f'Extract information in the format of {tool_name}',
					input_schema=schema,
					cache_control=CacheControlEphemeralParam(type='ephemeral'),
				)

				tool_choice = ToolChoiceToolParam(type='tool', name=tool_name)

				response = await self.get_client().messages.create(
					model=self.model,
					messages=anthropic_messages,
					tools=[tool],
					system=system_prompt or NOT_GIVEN,
					tool_choice=tool_choice,
					**self._get_client_params_for_invoke(),
				)

				if not isinstance(response, Message):
					raise ModelProviderError(
						message=f'Unexpected response type from Anthropic API: {type(response).__name__}. Response: {str(response)[:200]}',
						status_code=502,
						model=self.name,
					)

				usage = self._get_usage(response)

				for content_block in response.content:
					if hasattr(content_block, 'type') and content_block.type == 'tool_use':
						try:
							# Parse LLM response  
							result = output_format.model_validate(content_block.input)
							return ChatInvokeCompletion(completion=result, usage=usage)
						except Exception as e:
							print(f"⏺ LLM response parsing failed: {type(e).__name__}: {e}")
							if isinstance(content_block.input, str):
								data = json.loads(content_block.input)
								result = output_format.model_validate(data) 
								return ChatInvokeCompletion(
									completion=result,
									usage=usage,
								)
							raise e

				raise ValueError('Expected tool use in response but none found')

		except APIConnectionError as e:
			raise ModelProviderError(message=e.message, model=self.name) from e
		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e
		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e