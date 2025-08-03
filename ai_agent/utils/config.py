import json
import logging
import os
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


@cache
def is_running_in_docker() -> bool:
	try:
		if Path('/.dockerenv').exists():
			return True
	except Exception:
		pass

	try:
		cgroup_content = Path('/proc/1/cgroup').read_text().lower()
		if 'docker' in cgroup_content:
			return True
	except Exception:
		pass

	return False


class CoreConfig:
	@property
	def AI_AGENT_LOGGING_LEVEL(self) -> str:
		return os.getenv('AI_AGENT_LOGGING_LEVEL', 'info').lower()

	@property
	def ANONYMIZED_TELEMETRY(self) -> bool:
		return os.getenv('ANONYMIZED_TELEMETRY', 'true').lower()[:1] in 'ty1'

	@property
	def AI_AGENT_CONFIG_DIR(self) -> Path:
		xdg_config_home = Path(os.getenv('XDG_CONFIG_HOME', Path.home() / '.config'))
		path = Path(os.getenv('AI_AGENT_CONFIG_DIR', str(xdg_config_home / 'ai-agent'))).expanduser().resolve()
		
		if not self._dirs_created:
			path.mkdir(parents=True, exist_ok=True)
			self._dirs_created = True
		
		return path

	@property
	def AI_AGENT_DATA_DIR(self) -> Path:
		return self.AI_AGENT_CONFIG_DIR / 'data'

	_dirs_created = False


class AgentSettings(BaseModel):
	model_config = ConfigDict(extra='ignore')

	AI_AGENT_LOGGING_LEVEL: str = Field(default='info')
	ANONYMIZED_TELEMETRY: bool = Field(default=True)


class DBStyleEntry(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=lambda: str(uuid4()))
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)
	default: bool = Field(default=False)


class LLMEntry(DBStyleEntry):
	provider: str = 'openai'
	model: str = 'gpt-4'
	api_key: str | None = None
	base_url: str | None = None
	temperature: float | None = None


class AgentEntry(DBStyleEntry):
	max_steps: int = 50
	max_failures: int = 3
	retry_delay: int = 10
	use_vision: bool = True
	save_conversation_path: str | None = None


class ConfigFile(BaseModel):
	model_config = ConfigDict(extra='forbid')

	version: str = '1.0'
	llm: dict[str, LLMEntry] = Field(default_factory=dict)
	agent: dict[str, AgentEntry] = Field(default_factory=dict)

	@classmethod
	def create_default(cls) -> 'ConfigFile':
		new_config = cls()
		
		llm_id = str(uuid4())
		agent_id = str(uuid4())
		
		new_config.llm[llm_id] = LLMEntry(id=llm_id, default=True)
		new_config.agent[agent_id] = AgentEntry(id=agent_id, default=True)
		
		return new_config


def load_ai_agent_config() -> ConfigFile:
	config_dir = CoreConfig().AI_AGENT_CONFIG_DIR
	config_path = config_dir / 'config.json'
	
	if not config_path.exists():
		logger.info(f'Creating default config at {config_path}')
		config = ConfigFile.create_default()
		config_path.write_text(config.model_dump_json(indent=2))
		return config
	
	try:
		data = json.loads(config_path.read_text())
		return ConfigFile.model_validate(data)
	except Exception as e:
		logger.warning(f'Failed to load config from {config_path}: {e}')
		logger.info('Creating new default config')
		config = ConfigFile.create_default()
		config_path.write_text(config.model_dump_json(indent=2))
		return config


def get_default_llm() -> LLMEntry:
	config = load_ai_agent_config()
	for entry in config.llm.values():
		if entry.default:
			return entry
	
	if config.llm:
		return next(iter(config.llm.values()))
	
	return LLMEntry()


CONFIG = CoreConfig()