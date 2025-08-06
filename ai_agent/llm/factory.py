"""Factory for creating LLM instances"""
from ai_agent.llm.base import BaseChatModel
from ai_agent.llm.types import LLMType


def create_llm(llm_type: LLMType, model: str, **kwargs) -> BaseChatModel:
	"""Create an LLM instance based on type"""
	
	if llm_type == LLMType.ANTHROPIC:
		from ai_agent.llm.anthropic.chat import ChatAnthropic
		return ChatAnthropic(model=model, **kwargs)
	
	elif llm_type == LLMType.OPENAI:
		from ai_agent.llm.openai.chat import ChatOpenAI
		return ChatOpenAI(model=model, **kwargs)
	else:
		raise ValueError(f"Unsupported LLM type: {llm_type}")


__all__ = ['LLMType', 'create_llm']