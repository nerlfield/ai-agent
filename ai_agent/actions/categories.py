from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionCategory(str, Enum):
	"""Hierarchical categories for actions"""
	
	# Core categories
	NAVIGATION = 'navigation'
	DATA_EXTRACTION = 'data_extraction'
	DATA_MANIPULATION = 'data_manipulation'
	FILE_SYSTEM = 'file_system'
	COMMUNICATION = 'communication'
	AUTOMATION = 'automation'
	MONITORING = 'monitoring'
	SECURITY = 'security'
	
	# Subcategories
	WEB_NAVIGATION = 'navigation.web'
	API_NAVIGATION = 'navigation.api'
	WEB_SCRAPING = 'data_extraction.web'
	FILE_READING = 'data_extraction.file'
	DATABASE_QUERY = 'data_extraction.database'
	TEXT_PROCESSING = 'data_manipulation.text'
	DATA_TRANSFORMATION = 'data_manipulation.transform'
	FILE_OPERATIONS = 'file_system.operations'
	FILE_SEARCH = 'file_system.search'
	EMAIL = 'communication.email'
	MESSAGING = 'communication.messaging'
	TASK_AUTOMATION = 'automation.tasks'
	WORKFLOW = 'automation.workflow'
	HEALTH_CHECK = 'monitoring.health'
	METRICS = 'monitoring.metrics'
	AUTH = 'security.auth'
	ENCRYPTION = 'security.encryption'


class ActionTag(str, Enum):
	"""Fine-grained tags for action characteristics"""
	
	# Safety
	SAFE = 'safe'
	DESTRUCTIVE = 'destructive'
	REVERSIBLE = 'reversible'
	IDEMPOTENT = 'idempotent'
	
	SIDE_EFFECTS = 'side_effects'
	REQUIRES_FILE_SYSTEM = 'requires_file_system'
	
	# Performance
	FAST = 'fast'
	SLOW = 'slow'
	RESOURCE_INTENSIVE = 'resource_intensive'
	
	# Data access
	READ_ONLY = 'read_only'
	WRITE = 'write'
	CREATE = 'create'
	UPDATE = 'update'
	DELETE = 'delete'
	
	# Network
	NETWORK_REQUIRED = 'network_required'
	OFFLINE_CAPABLE = 'offline_capable'
	
	# Authentication
	AUTH_REQUIRED = 'auth_required'
	PUBLIC = 'public'
	
	# Visibility
	USER_VISIBLE = 'user_visible'
	BACKGROUND = 'background'
	
	# Reliability
	RETRY_SAFE = 'retry_safe'
	CRITICAL = 'critical'
	BEST_EFFORT = 'best_effort'


class CategoryInfo(BaseModel):
	"""Information about an action category"""
	
	name: str
	description: str
	parent: str | None = None
	typical_tags: set[ActionTag] = Field(default_factory=set)
	required_capabilities: set[str] = Field(default_factory=set)


class CategoryManager:
	"""Manages action categories and provides discovery features"""
	
	def __init__(self):
		self.categories: dict[str, CategoryInfo] = self._build_category_tree()
	
	def _build_category_tree(self) -> dict[str, CategoryInfo]:
		"""Build the default category tree"""
		return {
			ActionCategory.NAVIGATION: CategoryInfo(
				name='Navigation',
				description='Actions for navigating between resources',
				typical_tags={ActionTag.SAFE, ActionTag.READ_ONLY},
			),
			ActionCategory.WEB_NAVIGATION: CategoryInfo(
				name='Web Navigation',
				description='Navigate web pages and URLs',
				parent=ActionCategory.NAVIGATION,
				typical_tags={ActionTag.SAFE, ActionTag.NETWORK_REQUIRED},
				required_capabilities={'web_browser'},
			),
			ActionCategory.DATA_EXTRACTION: CategoryInfo(
				name='Data Extraction',
				description='Extract data from various sources',
				typical_tags={ActionTag.SAFE, ActionTag.READ_ONLY},
			),
			ActionCategory.WEB_SCRAPING: CategoryInfo(
				name='Web Scraping',
				description='Extract data from web pages',
				parent=ActionCategory.DATA_EXTRACTION,
				typical_tags={ActionTag.SAFE, ActionTag.NETWORK_REQUIRED},
				required_capabilities={'web_browser', 'html_parser'},
			),
			ActionCategory.FILE_SYSTEM: CategoryInfo(
				name='File System',
				description='File and directory operations',
				typical_tags={ActionTag.OFFLINE_CAPABLE},
				required_capabilities={'file_system'},
			),
			ActionCategory.FILE_OPERATIONS: CategoryInfo(
				name='File Operations',
				description='Create, read, update, delete files',
				parent=ActionCategory.FILE_SYSTEM,
				typical_tags={ActionTag.OFFLINE_CAPABLE},
			),
		}
	
	def get_category_info(self, category: str) -> CategoryInfo | None:
		"""Get information about a category"""
		return self.categories.get(category)
	
	def get_subcategories(self, parent: str) -> list[str]:
		"""Get all subcategories of a parent category"""
		return [
			cat for cat, info in self.categories.items()
			if info.parent == parent
		]
	
	def get_category_path(self, category: str) -> list[str]:
		"""Get the full path from root to this category"""
		path = []
		current = category
		
		while current:
			path.append(current)
			info = self.categories.get(current)
			if info and info.parent:
				current = info.parent
			else:
				break
		
		return list(reversed(path))
	
	def is_subcategory_of(self, category: str, parent: str) -> bool:
		"""Check if a category is a subcategory of parent"""
		path = self.get_category_path(category)
		return parent in path


class ActionDiscoveryFilter(BaseModel):
	"""Filter criteria for discovering actions"""
	
	category: ActionCategory | None = None
	include_subcategories: bool = True
	tags: set[ActionTag] = Field(default_factory=set)
	exclude_tags: set[ActionTag] = Field(default_factory=set)
	required_capabilities: set[str] = Field(default_factory=set)
	name_pattern: str | None = None
	min_priority: int | None = None
	max_priority: int | None = None
	min_success_rate: float | None = Field(None, ge=0.0, le=1.0)
	
	def matches(self, action_metadata: dict[str, Any]) -> bool:
		"""Check if an action matches this filter"""
		# Category check
		if self.category:
			action_cat = action_metadata.get('category')
			if not action_cat:
				return False
			
			if self.include_subcategories:
				manager = CategoryManager()
				if not manager.is_subcategory_of(action_cat, self.category):
					return False
			elif action_cat != self.category:
				return False
		
		# Tags check
		action_tags = set(action_metadata.get('tags', []))
		if self.tags and not self.tags.issubset(action_tags):
			return False
		if self.exclude_tags and self.exclude_tags.intersection(action_tags):
			return False
		
		# Capabilities check
		action_caps = set(action_metadata.get('requires_capabilities', []))
		if self.required_capabilities and not self.required_capabilities.issubset(action_caps):
			return False
		
		# Name pattern check
		if self.name_pattern:
			import re
			if not re.search(self.name_pattern, action_metadata.get('name', ''), re.IGNORECASE):
				return False
		
		# Priority check
		priority = action_metadata.get('priority', 0)
		if self.min_priority is not None and priority < self.min_priority:
			return False
		if self.max_priority is not None and priority > self.max_priority:
			return False
		
		# Success rate check
		success_rate = action_metadata.get('success_rate')
		if self.min_success_rate is not None and success_rate is not None:
			if success_rate < self.min_success_rate:
				return False
		
		return True