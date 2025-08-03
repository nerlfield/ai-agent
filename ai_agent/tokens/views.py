from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from ai_agent.llm.views import ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


class TokenUsageEntry(BaseModel):
	model: str
	timestamp: datetime
	usage: ChatInvokeUsage


class TokenCostCalculated(BaseModel):
	new_prompt_tokens: int
	new_prompt_cost: float

	prompt_read_cached_tokens: int | None
	prompt_read_cached_cost: float | None

	prompt_cached_creation_tokens: int | None
	prompt_cache_creation_cost: float | None

	completion_tokens: int
	completion_cost: float

	@property
	def prompt_cost(self) -> float:
		return self.new_prompt_cost + (self.prompt_read_cached_cost or 0) + (self.prompt_cache_creation_cost or 0)

	@property
	def total_cost(self) -> float:
		return (
			self.new_prompt_cost
			+ (self.prompt_read_cached_cost or 0)
			+ (self.prompt_cache_creation_cost or 0)
			+ self.completion_cost
		)


class ModelPricing(BaseModel):
	model: str
	input_cost_per_token: float | None
	output_cost_per_token: float | None

	cache_read_input_token_cost: float | None
	cache_creation_input_token_cost: float | None

	max_tokens: int | None
	max_input_tokens: int | None
	max_output_tokens: int | None


class CachedPricingData(BaseModel):
	timestamp: datetime
	data: dict[str, Any]


class ModelUsageStats(BaseModel):
	model: str
	prompt_tokens: int = 0
	completion_tokens: int = 0
	total_tokens: int = 0
	cost: float = 0.0
	invocations: int = 0
	average_tokens_per_invocation: float = 0.0


class ModelUsageTokens(BaseModel):
	model: str
	prompt_tokens: int
	prompt_cached_tokens: int
	completion_tokens: int
	total_tokens: int


class UsageSummary(BaseModel):
	total_prompt_tokens: int
	total_prompt_cost: float

	total_prompt_cached_tokens: int
	total_prompt_cached_cost: float

	total_completion_tokens: int
	total_completion_cost: float
	total_tokens: int
	total_cost: float
	entry_count: int

	by_model: dict[str, ModelUsageStats] = Field(default_factory=dict)