"""LLM type definitions"""
from enum import Enum


class LLMType(Enum):
	"""Supported LLM types"""
	ANTHROPIC = "anthropic"
	OPENAI = "openai"
	GOOGLE = "google"
	GROQ = "groq"
	OLLAMA = "ollama"


__all__ = ['LLMType']