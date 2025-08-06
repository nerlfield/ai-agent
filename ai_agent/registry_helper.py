from __future__ import annotations

from ai_agent.actions.data_actions import (
	CountWordsAction,
	ExtractTextAction,
	FormatTextAction,
	ParseJsonAction,
	ReplaceTextAction,
)
from ai_agent.actions.file_actions import (
	CreateDirectoryAction,
	DeleteFileAction,
	ListFilesAction,
	ReadFileAction,
	WriteFileAction,
)
from ai_agent.actions.web_actions import HttpGetAction, HttpRequestAction
from ai_agent.filesystem import FileSystem
from ai_agent.registry import Registry


def register_file_actions(registry: Registry, file_system: FileSystem) -> None:
	"""Register all file system actions with the registry"""
	
	# Read file
	@registry.action(
		description='Read the contents of a file',
		param_model=ReadFileAction().get_parameter_model(),
	)
	async def read_file(params, file_system=None):
		action = ReadFileAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data=file_system)
		return await action.execute(params, context)
	
	# Write file
	@registry.action(
		description='Write content to a file',
		param_model=WriteFileAction().get_parameter_model(),
	)
	async def write_file(params, file_system=None):
		action = WriteFileAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data=file_system)
		return await action.execute(params, context)
	
	# Delete file
	@registry.action(
		description='Delete a file',
		param_model=DeleteFileAction().get_parameter_model(),
	)
	async def delete_file(params, file_system=None):
		action = DeleteFileAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data=file_system)
		return await action.execute(params, context)
	
	# List files
	@registry.action(
		description='List files in the working directory',
		param_model=ListFilesAction().get_parameter_model(),
	)
	async def list_files(params, file_system=None):
		action = ListFilesAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data=file_system)
		return await action.execute(params, context)
	
	# Create directory
	@registry.action(
		description='Create a new directory',
		param_model=CreateDirectoryAction().get_parameter_model(),
	)
	async def create_directory(params, file_system=None):
		action = CreateDirectoryAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data=file_system)
		return await action.execute(params, context)


def register_data_actions(registry: Registry) -> None:
	"""Register all data manipulation actions with the registry"""
	
	# Extract text
	@registry.action(
		description='Extract text patterns using regular expressions',
		param_model=ExtractTextAction().get_parameter_model(),
	)
	async def extract_text(params):
		action = ExtractTextAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data={})
		return await action.execute(params, context)
	
	# Parse JSON
	@registry.action(
		description='Parse JSON text and optionally extract specific values',
		param_model=ParseJsonAction().get_parameter_model(),
	)
	async def parse_json(params):
		action = ParseJsonAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data={})
		return await action.execute(params, context)
	
	# Format text
	@registry.action(
		description='Format text using template strings',
		param_model=FormatTextAction().get_parameter_model(),
	)
	async def format_text(params):
		action = FormatTextAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data={})
		return await action.execute(params, context)
	
	# Count words
	@registry.action(
		description='Count words, lines, and characters in text',
		param_model=CountWordsAction().get_parameter_model(),
	)
	async def count_words(params):
		action = CountWordsAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data={})
		return await action.execute(params, context)
	
	# Replace text
	@registry.action(
		description='Find and replace text',
		param_model=ReplaceTextAction().get_parameter_model(),
	)
	async def replace_text(params):
		action = ReplaceTextAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data={})
		return await action.execute(params, context)


def register_web_actions(registry: Registry) -> None:
	"""Register all web/HTTP actions with the registry"""
	
	# HTTP request
	@registry.action(
		description='Make HTTP requests to APIs or web services',
		param_model=HttpRequestAction().get_parameter_model(),
	)
	async def http_request(params):
		action = HttpRequestAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data={})
		return await action.execute(params, context)
	
	# HTTP GET
	@registry.action(
		description='Fetch content from a URL using GET request',
		param_model=HttpGetAction().get_parameter_model(),
	)
	async def http_get(params):
		action = HttpGetAction()
		from ai_agent.actions.base import ActionContext
		context = ActionContext(data={})
		return await action.execute(params, context)


def register_all_actions(registry: Registry, file_system: FileSystem | None = None) -> None:
	"""Register all available actions with the registry"""
	if file_system:
		register_file_actions(registry, file_system)
	register_data_actions(registry)
	register_web_actions(registry)