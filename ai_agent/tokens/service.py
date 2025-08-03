import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from dotenv import load_dotenv

from ai_agent.llm.base import BaseChatModel
from ai_agent.llm.views import ChatInvokeUsage
from ai_agent.tokens.views import (
	CachedPricingData,
	ModelPricing,
	ModelUsageStats,
	ModelUsageTokens,
	TokenCostCalculated,
	TokenUsageEntry,
	UsageSummary,
)
from ai_agent.utils.config import CONFIG

load_dotenv()

logger = logging.getLogger(__name__)
cost_logger = logging.getLogger('cost')


def xdg_cache_home() -> Path:
	default = Path.home() / '.cache'
	xdg_cache = os.getenv('XDG_CACHE_HOME')
	if xdg_cache and (path := Path(xdg_cache)).is_absolute():
		return path
	return default


class TokenCost:
	CACHE_DIR_NAME = 'ai_agent/token_cost'
	CACHE_DURATION = timedelta(days=1)
	PRICING_URL = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'

	def __init__(self, include_cost: bool = False):
		self.include_cost = include_cost or os.getenv('AI_AGENT_CALCULATE_COST', 'false').lower() == 'true'

		self.usage_history: list[TokenUsageEntry] = []
		self.registered_llms: dict[str, BaseChatModel] = {}
		self._pricing_data: dict[str, Any] | None = None
		self._initialized = False
		self._cache_dir = xdg_cache_home() / self.CACHE_DIR_NAME

	async def initialize(self) -> None:
		if not self._initialized:
			if self.include_cost:
				await self._load_pricing_data()
			self._initialized = True

	async def _load_pricing_data(self) -> None:
		cache_file = await self._find_valid_cache()

		if cache_file:
			await self._load_from_cache(cache_file)
		else:
			await self._fetch_and_cache_pricing_data()

	async def _find_valid_cache(self) -> Path | None:
		try:
			self._cache_dir.mkdir(parents=True, exist_ok=True)
			cutoff_time = datetime.now() - self.CACHE_DURATION
			
			valid_files = []
			for file_path in self._cache_dir.glob('pricing_*.json'):
				file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
				if file_time > cutoff_time:
					valid_files.append((file_path, file_time))
			
			if valid_files:
				return max(valid_files, key=lambda x: x[1])[0]
			
		except Exception as e:
			logger.debug(f'Error finding cache files: {e}')
		
		return None

	async def _load_from_cache(self, cache_file: Path) -> None:
		try:
			async with aiofiles.open(cache_file, 'r') as f:
				content = await f.read()
				cached_data = CachedPricingData.model_validate_json(content)
				self._pricing_data = cached_data.data
				logger.debug(f'Loaded pricing data from cache: {cache_file}')
		except Exception as e:
			logger.warning(f'Failed to load pricing data from cache: {e}')
			await self._fetch_and_cache_pricing_data()

	async def _fetch_and_cache_pricing_data(self) -> None:
		try:
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.get(self.PRICING_URL)
				response.raise_for_status()
				self._pricing_data = response.json()

			self._cache_dir.mkdir(parents=True, exist_ok=True)
			cache_file = self._cache_dir / f'pricing_{int(datetime.now().timestamp())}.json'
			
			cached_data = CachedPricingData(
				timestamp=datetime.now(),
				data=self._pricing_data
			)
			
			async with aiofiles.open(cache_file, 'w') as f:
				await f.write(cached_data.model_dump_json())
			
			logger.debug(f'Fetched and cached pricing data to: {cache_file}')
			
		except Exception as e:
			logger.warning(f'Failed to fetch pricing data: {e}')
			self._pricing_data = {}

	def register_llm(self, llm: BaseChatModel) -> None:
		self.registered_llms[llm.name] = llm

	def track_usage(self, model: str, usage: ChatInvokeUsage) -> None:
		entry = TokenUsageEntry(
			model=model,
			timestamp=datetime.now(),
			usage=usage
		)
		self.usage_history.append(entry)

	def get_model_pricing(self, model: str) -> ModelPricing | None:
		if not self._pricing_data:
			return None
		
		model_data = self._pricing_data.get(model)
		if not model_data:
			return None
		
		return ModelPricing(
			model=model,
			input_cost_per_token=model_data.get('input_cost_per_token'),
			output_cost_per_token=model_data.get('output_cost_per_token'),
			cache_read_input_token_cost=model_data.get('cache_read_input_token_cost'),
			cache_creation_input_token_cost=model_data.get('cache_creation_input_token_cost'),
			max_tokens=model_data.get('max_tokens'),
			max_input_tokens=model_data.get('max_input_tokens'),
			max_output_tokens=model_data.get('max_output_tokens'),
		)

	def calculate_cost(self, model: str, usage: ChatInvokeUsage) -> TokenCostCalculated | None:
		pricing = self.get_model_pricing(model)
		if not pricing:
			return None

		input_cost = pricing.input_cost_per_token or 0
		output_cost = pricing.output_cost_per_token or 0
		cache_read_cost = pricing.cache_read_input_token_cost or input_cost
		cache_creation_cost = pricing.cache_creation_input_token_cost or input_cost

		new_prompt_tokens = usage.prompt_tokens - (usage.prompt_cached_tokens or 0)
		
		return TokenCostCalculated(
			new_prompt_tokens=new_prompt_tokens,
			new_prompt_cost=new_prompt_tokens * input_cost,
			prompt_read_cached_tokens=usage.prompt_cached_tokens,
			prompt_read_cached_cost=(usage.prompt_cached_tokens or 0) * cache_read_cost,
			prompt_cached_creation_tokens=usage.prompt_cache_creation_tokens,
			prompt_cache_creation_cost=(usage.prompt_cache_creation_tokens or 0) * cache_creation_cost,
			completion_tokens=usage.completion_tokens,
			completion_cost=usage.completion_tokens * output_cost,
		)

	def get_usage_summary(self) -> UsageSummary:
		if not self.usage_history:
			return UsageSummary(
				total_prompt_tokens=0,
				total_prompt_cost=0.0,
				total_prompt_cached_tokens=0,
				total_prompt_cached_cost=0.0,
				total_completion_tokens=0,
				total_completion_cost=0.0,
				total_tokens=0,
				total_cost=0.0,
				entry_count=0,
			)

		model_stats: dict[str, ModelUsageStats] = {}
		
		total_prompt_tokens = 0
		total_prompt_cost = 0.0
		total_prompt_cached_tokens = 0
		total_prompt_cached_cost = 0.0
		total_completion_tokens = 0
		total_completion_cost = 0.0

		for entry in self.usage_history:
			usage = entry.usage
			model = entry.model
			
			cost_calc = self.calculate_cost(model, usage) if self.include_cost else None
			
			if model not in model_stats:
				model_stats[model] = ModelUsageStats(model=model)
			
			stats = model_stats[model]
			stats.prompt_tokens += usage.prompt_tokens
			stats.completion_tokens += usage.completion_tokens
			stats.total_tokens += usage.total_tokens
			stats.invocations += 1
			
			if cost_calc:
				stats.cost += cost_calc.total_cost
				total_prompt_cost += cost_calc.prompt_cost
				total_prompt_cached_cost += cost_calc.prompt_read_cached_cost or 0
				total_completion_cost += cost_calc.completion_cost
			
			total_prompt_tokens += usage.prompt_tokens
			total_prompt_cached_tokens += usage.prompt_cached_tokens or 0
			total_completion_tokens += usage.completion_tokens

		for stats in model_stats.values():
			if stats.invocations > 0:
				stats.average_tokens_per_invocation = stats.total_tokens / stats.invocations

		return UsageSummary(
			total_prompt_tokens=total_prompt_tokens,
			total_prompt_cost=total_prompt_cost,
			total_prompt_cached_tokens=total_prompt_cached_tokens,
			total_prompt_cached_cost=total_prompt_cached_cost,
			total_completion_tokens=total_completion_tokens,
			total_completion_cost=total_completion_cost,
			total_tokens=total_prompt_tokens + total_completion_tokens,
			total_cost=total_prompt_cost + total_prompt_cached_cost + total_completion_cost,
			entry_count=len(self.usage_history),
			by_model=model_stats,
		)