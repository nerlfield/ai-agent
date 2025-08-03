from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from ai_agent.llm.anthropic.chat import ChatAnthropic
	from ai_agent.llm.openai.chat import ChatOpenAI
	from ai_agent.llm.google.chat import ChatGoogle
	from ai_agent.llm.groq.chat import ChatGroq
	from ai_agent.llm.ollama.chat import ChatOllama

_LAZY_IMPORTS = {
	'ChatAnthropic': ('ai_agent.llm.anthropic.chat', 'ChatAnthropic'),
	'ChatOpenAI': ('ai_agent.llm.openai.chat', 'ChatOpenAI'),
	'ChatGoogle': ('ai_agent.llm.google.chat', 'ChatGoogle'),
	'ChatGroq': ('ai_agent.llm.groq.chat', 'ChatGroq'),
	'ChatOllama': ('ai_agent.llm.ollama.chat', 'ChatOllama'),
}

def __getattr__(name: str):
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module
			module = import_module(module_path)
			attr = getattr(module, attr_name)
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e
	
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
	'ChatAnthropic',
	'ChatOpenAI', 
	'ChatGoogle',
	'ChatGroq',
	'ChatOllama',
]