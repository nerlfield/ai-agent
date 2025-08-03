from typing import Literal, Union

from openai import BaseModel


def _truncate(text: str, max_length: int = 50) -> str:
	if len(text) <= max_length:
		return text
	return text[: max_length - 3] + '...'


def _format_image_url(url: str, max_length: int = 50) -> str:
	if url.startswith('data:'):
		media_type = url.split(';')[0].split(':')[1] if ';' in url else 'image'
		return f'<base64 {media_type}>'
	else:
		return _truncate(url, max_length)


class ContentPartTextParam(BaseModel):
	text: str
	type: Literal['text'] = 'text'

	def __str__(self) -> str:
		return f'Text: {_truncate(self.text)}'

	def __repr__(self) -> str:
		return f'ContentPartTextParam(text={_truncate(self.text)})'


class ContentPartRefusalParam(BaseModel):
	refusal: str
	type: Literal['refusal'] = 'refusal'

	def __str__(self) -> str:
		return f'Refusal: {_truncate(self.refusal)}'

	def __repr__(self) -> str:
		return f'ContentPartRefusalParam(refusal={_truncate(repr(self.refusal), 50)})'


SupportedImageMediaType = Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']


class ImageURL(BaseModel):
	url: str
	detail: Literal['auto', 'low', 'high'] = 'auto'
	media_type: SupportedImageMediaType = 'image/png'

	def __str__(self) -> str:
		url_display = _format_image_url(self.url)
		return f'🖼️  Image[{self.media_type}, detail={self.detail}]: {url_display}'

	def __repr__(self) -> str:
		url_repr = _format_image_url(self.url, 30)
		return f'ImageURL(url={repr(url_repr)}, detail={repr(self.detail)}, media_type={repr(self.media_type)})'


class ContentPartImageParam(BaseModel):
	image_url: ImageURL
	type: Literal['image_url'] = 'image_url'

	def __str__(self) -> str:
		return str(self.image_url)

	def __repr__(self) -> str:
		return f'ContentPartImageParam(image_url={repr(self.image_url)})'


class Function(BaseModel):
	arguments: str
	name: str

	def __str__(self) -> str:
		args_preview = _truncate(self.arguments, 80)
		return f'{self.name}({args_preview})'

	def __repr__(self) -> str:
		args_repr = _truncate(repr(self.arguments), 50)
		return f'Function(name={repr(self.name)}, arguments={args_repr})'


class ToolCall(BaseModel):
	id: str
	function: Function
	type: Literal['function'] = 'function'

	def __str__(self) -> str:
		return f'ToolCall[{self.id}]: {self.function}'

	def __repr__(self) -> str:
		return f'ToolCall(id={repr(self.id)}, function={repr(self.function)})'


class _MessageBase(BaseModel):
	role: Literal['user', 'system', 'assistant']
	cache: bool = False


class UserMessage(_MessageBase):
	role: Literal['user'] = 'user'
	content: str | list[ContentPartTextParam | ContentPartImageParam]
	name: str | None = None

	@property
	def text(self) -> str:
		if isinstance(self.content, str):
			return self.content
		elif isinstance(self.content, list):
			return '\n'.join([part.text for part in self.content if part.type == 'text'])
		else:
			return ''

	def __str__(self) -> str:
		return f'UserMessage(content={self.text})'

	def __repr__(self) -> str:
		return f'UserMessage(content={repr(self.text)})'


class SystemMessage(_MessageBase):
	role: Literal['system'] = 'system'
	content: str | list[ContentPartTextParam]
	name: str | None = None

	@property
	def text(self) -> str:
		if isinstance(self.content, str):
			return self.content
		elif isinstance(self.content, list):
			return '\n'.join([part.text for part in self.content if part.type == 'text'])
		else:
			return ''

	def __str__(self) -> str:
		return f'SystemMessage(content={self.text})'

	def __repr__(self) -> str:
		return f'SystemMessage(content={repr(self.text)})'


class AssistantMessage(_MessageBase):
	role: Literal['assistant'] = 'assistant'
	content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None
	name: str | None = None
	refusal: str | None = None
	tool_calls: list[ToolCall] = []

	@property
	def text(self) -> str:
		if isinstance(self.content, str):
			return self.content
		elif isinstance(self.content, list):
			text = ''
			for part in self.content:
				if part.type == 'text':
					text += part.text
				elif part.type == 'refusal':
					text += f'[Refusal] {part.refusal}'
			return text
		else:
			return ''

	def __str__(self) -> str:
		return f'AssistantMessage(content={self.text})'

	def __repr__(self) -> str:
		return f'AssistantMessage(content={repr(self.text)})'


BaseMessage = Union[UserMessage, SystemMessage, AssistantMessage]