from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.registry.views import ActionResult

# Type variables for generic actions
TParams = TypeVar('TParams', bound=BaseModel)
TResult = TypeVar('TResult', bound=ActionResult)
TContext = TypeVar('TContext')


@runtime_checkable
class ActionProtocol(Protocol[TParams, TResult]):
	"""Protocol defining the interface for all actions"""
	
	name: str
	description: str
	category: str | None
	tags: set[str]
	
	async def execute(self, parameters: TParams, context: Any) -> TResult:
		"""Execute the action with given parameters and context"""
		...
	
	def validate_context(self, context: Any) -> bool:
		"""Validate if the action can be executed in the given context"""
		...


class ActionParameter(BaseModel):
	"""Base class for action parameters with validation"""
	model_config = ConfigDict(extra='ignore', validate_assignment=True)
	
	def validate_constraints(self) -> None:
		"""Override to add custom validation logic"""
		pass
	
	def model_post_init(self, __context) -> None:
		"""Run custom validation after model initialization"""
		self.validate_constraints()


class ActionMetadata(BaseModel):
	"""Metadata about an action for discovery and documentation"""
	name: str
	description: str
	category: str | None = None
	tags: set[str] = Field(default_factory=set)
	priority: int = Field(default=0, ge=0, le=100)
	requires_capabilities: set[str] = Field(default_factory=set)
	example_usage: str | None = None
	success_rate: float | None = Field(None, ge=0.0, le=1.0)
	average_duration_ms: float | None = Field(None, ge=0)


class ActionContext(BaseModel, Generic[TContext]):
	"""Context provided to actions during execution"""
	model_config = ConfigDict(extra='allow')
	
	data: TContext
	metadata: dict[str, Any] = Field(default_factory=dict)
	capabilities: set[str] = Field(default_factory=set)
	
	def has_capability(self, capability: str) -> bool:
		"""Check if a capability is available in this context"""
		return capability in self.capabilities


class BaseAction(ABC, Generic[TParams, TResult, TContext]):
	"""Base class for implementing actions"""
	
	# Class attributes to be overridden
	name: str
	description: str
	category: str | None = None
	tags: set[str] = set()
	priority: int = 0
	requires_capabilities: set[str] = set()
	
	def __init__(self):
		if not hasattr(self, 'name'):
			raise ValueError(f'{self.__class__.__name__} must define a "name" attribute')
		if not hasattr(self, 'description'):
			raise ValueError(f'{self.__class__.__name__} must define a "description" attribute')
	
	@abstractmethod
	async def execute(self, parameters: TParams, context: ActionContext[TContext]) -> TResult:
		"""
		Execute the action with given parameters and context.
		
		Args:
			parameters: Validated parameters for the action
			context: Execution context with capabilities and metadata
			
		Returns:
			Result of the action execution
		"""
		pass
	
	def validate_context(self, context: ActionContext[TContext]) -> bool:
		"""
		Validate if the action can be executed in the given context.
		
		Default implementation checks required capabilities.
		Override for custom validation logic.
		"""
		return all(context.has_capability(cap) for cap in self.requires_capabilities)
	
	async def pre_execute(self, parameters: TParams, context: ActionContext[TContext]) -> None:
		"""Hook called before action execution"""
		pass
	
	async def post_execute(
		self,
		parameters: TParams,
		context: ActionContext[TContext],
		result: TResult
	) -> None:
		"""Hook called after action execution"""
		pass
	
	def get_metadata(self) -> ActionMetadata:
		"""Get metadata about this action"""
		return ActionMetadata(
			name=self.name,
			description=self.description,
			category=self.category,
			tags=self.tags,
			priority=self.priority,
			requires_capabilities=self.requires_capabilities,
		)
	
	def get_parameter_model(self) -> type[TParams]:
		"""Get the parameter model for this action"""
		# Extract from generic type hints
		import inspect
		from typing import get_args
		
		for base in inspect.getmro(self.__class__):
			if hasattr(base, '__orig_bases__'):
				for orig_base in base.__orig_bases__:
					if hasattr(orig_base, '__origin__') and orig_base.__origin__ is BaseAction:
						args = get_args(orig_base)
						if args and len(args) >= 1:
							return args[0]
		
		raise ValueError(f'Could not determine parameter model for {self.__class__.__name__}')