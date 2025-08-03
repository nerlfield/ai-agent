from openai.types.chat import ChatCompletionMessageParam

from ai_agent.llm.messages import BaseMessage


class OpenAIMessageSerializer:
	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> list[ChatCompletionMessageParam]:
		serialized_messages: list[ChatCompletionMessageParam] = []
		
		for message in messages:
			if message.role == 'user':
				serialized_messages.append({
					'role': 'user',
					'content': message.text,
				})
			elif message.role == 'system':
				serialized_messages.append({
					'role': 'system', 
					'content': message.text,
				})
			elif message.role == 'assistant':
				serialized_messages.append({
					'role': 'assistant',
					'content': message.text,
				})
				
		return serialized_messages