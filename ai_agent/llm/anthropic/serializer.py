import json
from typing import overload

from anthropic.types import (
	Base64ImageSourceParam,
	CacheControlEphemeralParam,
	ImageBlockParam,
	MessageParam,
	TextBlockParam,
	ToolUseBlockParam,
	URLImageSourceParam,
)

from ai_agent.llm.messages import (
	AssistantMessage,
	BaseMessage,
	ContentPartImageParam,
	ContentPartTextParam,
	SupportedImageMediaType,
	SystemMessage,
	UserMessage,
)

NonSystemMessage = UserMessage | AssistantMessage


class AnthropicMessageSerializer:
	@staticmethod
	def _is_base64_image(url: str) -> bool:
		return url.startswith('data:image/')

	@staticmethod
	def _parse_base64_url(url: str) -> tuple[SupportedImageMediaType, str]:
		if not url.startswith('data:'):
			raise ValueError(f'Invalid base64 URL: {url}')

		header, data = url.split(',', 1)
		media_type = header.split(';')[0].replace('data:', '')

		supported_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
		if media_type not in supported_types:
			media_type = 'image/png'

		return media_type, data  # type: ignore

	@staticmethod
	def _serialize_cache_control(use_cache: bool) -> CacheControlEphemeralParam | None:
		if use_cache:
			return CacheControlEphemeralParam(type='ephemeral')
		return None

	@staticmethod
	def _serialize_content_part_text(part: ContentPartTextParam, use_cache: bool) -> TextBlockParam:
		return TextBlockParam(
			text=part.text, type='text', cache_control=AnthropicMessageSerializer._serialize_cache_control(use_cache)
		)

	@staticmethod
	def _serialize_content_part_image(part: ContentPartImageParam) -> ImageBlockParam:
		url = part.image_url.url

		if AnthropicMessageSerializer._is_base64_image(url):
			media_type, data = AnthropicMessageSerializer._parse_base64_url(url)
			return ImageBlockParam(
				source=Base64ImageSourceParam(
					data=data,
					media_type=media_type,
					type='base64',
				),
				type='image',
			)
		else:
			return ImageBlockParam(source=URLImageSourceParam(url=url, type='url'), type='image')

	@staticmethod
	def _serialize_content_to_str(
		content: str | list[ContentPartTextParam], use_cache: bool = False
	) -> list[TextBlockParam] | str:
		cache_control = AnthropicMessageSerializer._serialize_cache_control(use_cache)

		if isinstance(content, str):
			if cache_control:
				return [TextBlockParam(text=content, type='text', cache_control=cache_control)]
			else:
				return content

		serialized_blocks: list[TextBlockParam] = []
		for part in content:
			if part.type == 'text':
				serialized_blocks.append(AnthropicMessageSerializer._serialize_content_part_text(part, use_cache))

		return serialized_blocks

	@staticmethod
	def _serialize_content(
		content: str | list[ContentPartTextParam | ContentPartImageParam],
		use_cache: bool = False,
	) -> str | list[TextBlockParam | ImageBlockParam]:
		if isinstance(content, str):
			if use_cache:
				return [TextBlockParam(text=content, type='text', cache_control=CacheControlEphemeralParam(type='ephemeral'))]
			else:
				return content

		serialized_blocks: list[TextBlockParam | ImageBlockParam] = []
		for part in content:
			if part.type == 'text':
				serialized_blocks.append(AnthropicMessageSerializer._serialize_content_part_text(part, use_cache))
			elif part.type == 'image_url':
				serialized_blocks.append(AnthropicMessageSerializer._serialize_content_part_image(part))

		return serialized_blocks

	@staticmethod
	def _serialize_tool_calls_to_content(tool_calls, use_cache: bool = False) -> list[ToolUseBlockParam]:
		blocks: list[ToolUseBlockParam] = []
		for tool_call in tool_calls:
			try:
				input_obj = json.loads(tool_call.function.arguments)
			except json.JSONDecodeError:
				input_obj = {'arguments': tool_call.function.arguments}

			blocks.append(
				ToolUseBlockParam(
					id=tool_call.id,
					input=input_obj,
					name=tool_call.function.name,
					type='tool_use',
					cache_control=AnthropicMessageSerializer._serialize_cache_control(use_cache),
				)
			)
		return blocks

	@overload
	@staticmethod
	def serialize(message: UserMessage) -> MessageParam: ...

	@overload
	@staticmethod
	def serialize(message: SystemMessage) -> SystemMessage: ...

	@overload
	@staticmethod
	def serialize(message: AssistantMessage) -> MessageParam: ...

	@staticmethod
	def serialize(message: BaseMessage) -> MessageParam | SystemMessage:
		if isinstance(message, UserMessage):
			content = AnthropicMessageSerializer._serialize_content(message.content, use_cache=message.cache)
			return MessageParam(role='user', content=content)

		elif isinstance(message, SystemMessage):
			return message

		elif isinstance(message, AssistantMessage):
			blocks: list[TextBlockParam | ToolUseBlockParam] = []

			if message.content is not None:
				if isinstance(message.content, str):
					blocks.append(
						TextBlockParam(
							text=message.content,
							type='text',
							cache_control=AnthropicMessageSerializer._serialize_cache_control(message.cache),
						)
					)
				else:
					for part in message.content:
						if part.type == 'text':
							blocks.append(AnthropicMessageSerializer._serialize_content_part_text(part, use_cache=message.cache))

			if message.tool_calls:
				tool_blocks = AnthropicMessageSerializer._serialize_tool_calls_to_content(
					message.tool_calls, use_cache=message.cache
				)
				blocks.extend(tool_blocks)

			if not blocks:
				blocks.append(
					TextBlockParam(
						text='', type='text', cache_control=AnthropicMessageSerializer._serialize_cache_control(message.cache)
					)
				)

			if message.cache or len(blocks) > 1:
				content = blocks
			else:
				single_block = blocks[0]
				if single_block['type'] == 'text' and not single_block.get('cache_control'):
					content = single_block['text']
				else:
					content = blocks

			return MessageParam(
				role='assistant',
				content=content,
			)

		else:
			raise ValueError(f'Unknown message type: {type(message)}')

	@staticmethod
	def _clean_cache_messages(messages: list[NonSystemMessage]) -> list[NonSystemMessage]:
		if not messages:
			return messages

		cleaned_messages = [msg.model_copy(deep=True) for msg in messages]

		last_cache_index = -1
		for i in range(len(cleaned_messages) - 1, -1, -1):
			if cleaned_messages[i].cache:
				last_cache_index = i
				break

		if last_cache_index != -1:
			for i, msg in enumerate(cleaned_messages):
				if i != last_cache_index and msg.cache:
					msg.cache = False

		return cleaned_messages

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> tuple[list[MessageParam], list[TextBlockParam] | str | None]:
		messages = [m.model_copy(deep=True) for m in messages]

		normal_messages: list[NonSystemMessage] = []
		system_message: SystemMessage | None = None

		for message in messages:
			if isinstance(message, SystemMessage):
				system_message = message
			else:
				normal_messages.append(message)

		normal_messages = AnthropicMessageSerializer._clean_cache_messages(normal_messages)

		serialized_messages: list[MessageParam] = []
		for message in normal_messages:
			serialized_messages.append(AnthropicMessageSerializer.serialize(message))

		serialized_system_message: list[TextBlockParam] | str | None = None
		if system_message:
			serialized_system_message = AnthropicMessageSerializer._serialize_content_to_str(
				system_message.content, use_cache=system_message.cache
			)

		return serialized_messages, serialized_system_message