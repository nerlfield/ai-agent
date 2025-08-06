from __future__ import annotations

from pathlib import Path

from pydantic import Field

from ai_agent.actions.base import ActionContext, ActionParameter, BaseAction
from ai_agent.actions.categories import ActionCategory, ActionTag
from ai_agent.actions.validation import NonEmptyStringField
from ai_agent.filesystem import FileSystem
from ai_agent.common.models import ActionResult


class ReadFileParams(ActionParameter):
	"""Parameters for reading a file"""
	path: str = NonEmptyStringField(description='Path to the file to read')


class WriteFileParams(ActionParameter):
	"""Parameters for writing a file"""
	path: str = NonEmptyStringField(description='Path to the file to write')
	content: str = Field(description='Content to write to the file')
	append: bool = Field(default=False, description='Whether to append to existing file')


class DeleteFileParams(ActionParameter):
	"""Parameters for deleting a file"""
	path: str = NonEmptyStringField(description='Path to the file to delete')


class ListFilesParams(ActionParameter):
	"""Parameters for listing files"""
	pattern: str = Field(default='*', description='Glob pattern for file matching')


class CreateDirectoryParams(ActionParameter):
	"""Parameters for creating a directory"""
	path: str = NonEmptyStringField(description='Path to the directory to create')


class ReadFileAction(BaseAction[ReadFileParams, ActionResult, FileSystem]):
	"""Action to read file contents"""
	
	name = 'read_file'
	description = 'Read the contents of a file'
	category = ActionCategory.FILE_READING
	tags = {ActionTag.SAFE, ActionTag.READ_ONLY, ActionTag.OFFLINE_CAPABLE}
	requires_capabilities = {'file_system'}
	
	async def execute(self, parameters: ReadFileParams, context: ActionContext[FileSystem]) -> ActionResult:
		try:
			file_system = context.data
			content = file_system.read_file(parameters.path)
			
			return ActionResult(
				success=True,
				extracted_content=content,
				metadata={'path': parameters.path, 'size': len(content)},
			)
		except FileNotFoundError:
			return ActionResult(
				success=False,
				error=f'File not found: {parameters.path}',
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Failed to read file: {str(e)}',
			)


class WriteFileAction(BaseAction[WriteFileParams, ActionResult, FileSystem]):
	"""Action to write file contents"""
	
	name = 'write_file'
	description = 'Write content to a file'
	category = ActionCategory.FILE_OPERATIONS
	tags = {ActionTag.WRITE, ActionTag.OFFLINE_CAPABLE, ActionTag.REVERSIBLE}
	requires_capabilities = {'file_system'}
	
	async def execute(self, parameters: WriteFileParams, context: ActionContext[FileSystem]) -> ActionResult:
		try:
			file_system = context.data
			
			if parameters.append:
				file_system.append_to_file(parameters.path, parameters.content)
				action = 'appended to'
			else:
				file_system.write_file(parameters.path, parameters.content)
				action = 'written to'
			
			return ActionResult(
				success=True,
				extracted_content=f'Successfully {action} file: {parameters.path}',
				metadata={'path': parameters.path, 'size': len(parameters.content), 'append': parameters.append},
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Failed to write file: {str(e)}',
			)


class DeleteFileAction(BaseAction[DeleteFileParams, ActionResult, FileSystem]):
	"""Action to delete a file"""
	
	name = 'delete_file'
	description = 'Delete a file'
	category = ActionCategory.FILE_OPERATIONS
	tags = {ActionTag.DELETE, ActionTag.DESTRUCTIVE, ActionTag.OFFLINE_CAPABLE}
	requires_capabilities = {'file_system'}
	priority = 10  # Lower priority for destructive actions
	
	async def execute(self, parameters: DeleteFileParams, context: ActionContext[FileSystem]) -> ActionResult:
		try:
			file_system = context.data
			
			# Check if file exists first
			if not file_system.file_exists(parameters.path):
				return ActionResult(
					success=False,
					error=f'File does not exist: {parameters.path}',
				)
			
			file_system.delete_file(parameters.path)
			
			return ActionResult(
				success=True,
				extracted_content=f'Successfully deleted file: {parameters.path}',
				metadata={'path': parameters.path},
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Failed to delete file: {str(e)}',
			)


class ListFilesAction(BaseAction[ListFilesParams, ActionResult, FileSystem]):
	"""Action to list files in working directory"""
	
	name = 'list_files'
	description = 'List files in the working directory'
	category = ActionCategory.FILE_SEARCH
	tags = {ActionTag.SAFE, ActionTag.READ_ONLY, ActionTag.OFFLINE_CAPABLE}
	requires_capabilities = {'file_system'}
	
	async def execute(self, parameters: ListFilesParams, context: ActionContext[FileSystem]) -> ActionResult:
		try:
			file_system = context.data
			files = file_system.list_files(parameters.pattern)
			
			# Format file list
			file_list = []
			for file_info in files:
				if file_info.is_directory:
					file_list.append(f"[DIR] {file_info.path}")
				else:
					file_list.append(f"{file_info.path} ({file_info.size} bytes)")
			
			content = '\n'.join(file_list) if file_list else 'No files found'
			
			return ActionResult(
				success=True,
				extracted_content=content,
				metadata={'count': len(files), 'pattern': parameters.pattern},
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Failed to list files: {str(e)}',
			)


class CreateDirectoryAction(BaseAction[CreateDirectoryParams, ActionResult, FileSystem]):
	"""Action to create a directory"""
	
	name = 'create_directory'
	description = 'Create a new directory'
	category = ActionCategory.FILE_OPERATIONS
	tags = {ActionTag.CREATE, ActionTag.OFFLINE_CAPABLE, ActionTag.IDEMPOTENT}
	requires_capabilities = {'file_system'}
	
	async def execute(self, parameters: CreateDirectoryParams, context: ActionContext[FileSystem]) -> ActionResult:
		try:
			file_system = context.data
			file_system.create_directory(parameters.path)
			
			return ActionResult(
				success=True,
				extracted_content=f'Successfully created directory: {parameters.path}',
				metadata={'path': parameters.path},
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'Failed to create directory: {str(e)}',
			)