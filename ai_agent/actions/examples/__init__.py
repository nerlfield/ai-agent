"""
Example action implementations demonstrating the base action framework.

This module contains concrete examples of how to create actions using the framework,
showcasing different patterns and capabilities.
"""

from ai_agent.actions.examples.file_actions import (
	ReadFileAction,
	WriteFileAction,
	DeleteFileAction,
	ReadFileParameters,
	WriteFileParameters,
	DeleteFileParameters,
	FileActionContext,
)

from ai_agent.actions.examples.web_actions import (
	NavigateAction,
	ClickElementAction,
	TypeTextAction,
	NavigateParameters,
	ClickElementParameters,
	TypeTextParameters,
	WebActionContext,
)

from ai_agent.actions.examples.data_actions import (
	ProcessTextAction,
	QueryDatabaseAction,
	TransformDataAction,
	ProcessTextParameters,
	QueryDatabaseParameters,
	TransformDataParameters,
	DataActionContext,
)

__all__ = [
	# File actions
	'ReadFileAction',
	'WriteFileAction',
	'DeleteFileAction',
	'ReadFileParameters',
	'WriteFileParameters',
	'DeleteFileParameters',
	'FileActionContext',
	
	# Web actions
	'NavigateAction',
	'ClickElementAction',
	'TypeTextAction',
	'NavigateParameters',
	'ClickElementParameters',
	'TypeTextParameters',
	'WebActionContext',
	
	# Data actions
	'ProcessTextAction',
	'QueryDatabaseAction',
	'TransformDataAction',
	'ProcessTextParameters',
	'QueryDatabaseParameters',
	'TransformDataParameters',
	'DataActionContext',
]