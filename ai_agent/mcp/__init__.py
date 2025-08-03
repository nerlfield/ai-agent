from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from ai_agent.mcp.client import MCPClient

_LAZY_IMPORTS = {
	'MCPClient': ('ai_agent.mcp.client', 'MCPClient'),
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
	'MCPClient',
]