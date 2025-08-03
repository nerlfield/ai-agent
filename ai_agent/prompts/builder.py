from __future__ import annotations

from typing import Any, Literal

from ai_agent.llm.messages import (
	ContentPartImageParam,
	ContentPartTextParam,
	ImageURL,
	UserMessage,
)
from ai_agent.prompts.context import ContextProvider


class GenericMessagePrompt:
	"""Generic message prompt builder that works with context providers"""
	
	vision_detail_level: Literal['auto', 'low', 'high']
	
	def __init__(
		self,
		context_providers: list[ContextProvider],
		media_items: list[str | bytes] | None = None,
		page_filtered_actions: str | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		**kwargs: Any,
	):
		self.context_providers = context_providers
		self.media_items = media_items or []
		self.page_filtered_actions = page_filtered_actions
		self.vision_detail_level = vision_detail_level
		self.kwargs = kwargs
	
	def _build_context_sections(self) -> str:
		"""Build all context sections from providers"""
		sections = []
		
		for provider in self.context_providers:
			context_name = provider.get_context_name()
			context_desc = provider.get_context_description()
			
			if context_desc:  # Only add if there's content
				section = f'<{context_name}>\n{context_desc}\n</{context_name}>'
				sections.append(section)
		
		return '\n'.join(sections)
	
	def _prepare_media_for_message(self, media_data: str | bytes) -> ImageURL:
		"""Convert media data to proper format for message"""
		if isinstance(media_data, str):
			# Assume it's a file path or URL
			return ImageURL(
				url=media_data,
				detail=self.vision_detail_level,
			)
		else:
			# Assume it's raw bytes - convert to base64
			import base64
			base64_data = base64.b64encode(media_data).decode('utf-8')
			return ImageURL(
				url=f"data:image/png;base64,{base64_data}",
				detail=self.vision_detail_level,
			)
	
	def get_user_message(self, use_vision: bool = True) -> UserMessage:
		"""Build the user message from all context providers"""
		
		# Build text content from context providers
		text_content = self._build_context_sections()
		
		# Add page-specific actions if available
		if self.page_filtered_actions:
			text_content += '\n\nFor this context, these additional actions are available:\n'
			text_content += self.page_filtered_actions
		
		# Create message content
		if use_vision and self.media_items:
			# Start with text description
			content_parts: list[ContentPartTextParam | ContentPartImageParam] = [
				ContentPartTextParam(text=text_content)
			]
			
			# Add all media items
			for media_item in self.media_items:
				image_url = self._prepare_media_for_message(media_item)
				content_parts.append(
					ContentPartImageParam(
						image_url=image_url
					)
				)
			
			return UserMessage(content=content_parts)
		else:
			# Text-only message
			return UserMessage(content=text_content)