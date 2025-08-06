"""Unified registry system for action management."""

from ai_agent.registry.models import (
	ActionDiscoveryFilter,
	ActionModel,
	ActionRegistry,
	ActionResult,
	RegisteredAction,
	SpecialActionParameters,
)
from ai_agent.registry.service import Registry

# Backward compatibility aliases
ActionRegistryModel = ActionRegistry

__all__ = [
	# Core registry
	'Registry',
	
	# Models
	'RegisteredAction',
	'ActionModel',
	'ActionRegistry',
	'ActionDiscoveryFilter',
	'SpecialActionParameters',
	'ActionResult',
	
	# Aliases for backward compatibility
	'ActionRegistryModel',
]