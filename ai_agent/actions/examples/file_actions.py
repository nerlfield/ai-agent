"""
Example file system actions demonstrating the action framework.

These actions showcase file operations with proper validation, error handling,
and result formatting.
"""

import os
from pathlib import Path
from typing import ClassVar

from pydantic import Field, validator

from ai_agent.actions.base import ActionContext, ActionParameter, BaseAction
from ai_agent.actions.categories import ActionCategory, ActionTag
from ai_agent.actions.results import EnhancedActionResult, ResultCategory
from ai_agent.actions.validation import NonEmptyStringField, validate_file_path


class FileActionContext(ActionContext):
	"""Context for file system operations"""
	
	def __init__(self, **kwargs):
		super().__init__(context_type="file_system", **kwargs)
		self.capabilities.update({"file_read", "file_write", "file_delete"})
		
		# Add file system resources
		self.resources["current_directory"] = Path.cwd()
		self.resources["permissions"] = {"read": True, "write": True, "delete": True}
	
	def get_safe_path(self, file_path: str) -> Path:
		"""Get a safe, absolute path within allowed directories"""
		path = Path(file_path).resolve()
		
		# Basic security check - prevent path traversal
		current_dir = self.resources["current_directory"]
		try:
			path.relative_to(current_dir)
		except ValueError:
			raise ValueError(f"Path {path} is outside allowed directory {current_dir}")
		
		return path


class ReadFileParameters(ActionParameter):
	"""Parameters for reading a file"""
	
	file_path: str = NonEmptyStringField(
		description="Path to the file to read",
		examples=["/path/to/file.txt", "data/input.json", "documents/readme.md"]
	)
	
	encoding: str = Field(
		default="utf-8",
		description="Text encoding to use when reading the file",
		examples=["utf-8", "latin-1", "ascii"]
	)
	
	max_size_mb: float = Field(
		default=10.0,
		description="Maximum file size in MB to read",
		ge=0.1,
		le=100.0
	)
	
	def validate_constraints(self) -> None:
		"""Additional validation for file parameters"""
		validate_file_path(self.file_path)


class WriteFileParameters(ActionParameter):
	"""Parameters for writing a file"""
	
	file_path: str = NonEmptyStringField(
		description="Path to the file to write",
		examples=["/path/to/output.txt", "data/results.json"]
	)
	
	content: str = Field(
		description="Content to write to the file",
		examples=["Hello, world!", '{"key": "value"}', "# Markdown content"]
	)
	
	encoding: str = Field(
		default="utf-8",
		description="Text encoding to use when writing the file"
	)
	
	overwrite: bool = Field(
		default=False,
		description="Whether to overwrite existing files"
	)
	
	create_directories: bool = Field(
		default=True,
		description="Whether to create parent directories if they don't exist"
	)


class DeleteFileParameters(ActionParameter):
	"""Parameters for deleting a file"""
	
	file_path: str = NonEmptyStringField(
		description="Path to the file to delete",
		examples=["/path/to/unwanted.txt", "temp/cache.dat"]
	)
	
	confirm_delete: bool = Field(
		default=False,
		description="Confirmation that the file should be deleted"
	)
	
	def validate_constraints(self) -> None:
		"""Require explicit confirmation for deletion"""
		if not self.confirm_delete:
			raise ValueError("confirm_delete must be True to delete files")


class ReadFileAction(BaseAction[ReadFileParameters, EnhancedActionResult, FileActionContext]):
	"""Action to read content from a file"""
	
	name: ClassVar[str] = "read_file"
	description: ClassVar[str] = "Read text content from a file"
	category: ClassVar[str] = ActionCategory.FILE_SYSTEM
	tags: ClassVar[set[str]] = {ActionTag.READ_ONLY, ActionTag.SAFE, ActionTag.REQUIRES_FILE_SYSTEM}
	
	async def execute(
		self,
		parameters: ReadFileParameters,
		context: FileActionContext,
		**kwargs
	) -> EnhancedActionResult:
		"""Execute file reading"""
		try:
			# Get safe path
			file_path = context.get_safe_path(parameters.file_path)
			
			# Check if file exists
			if not file_path.exists():
				return EnhancedActionResult.error_with_details(
					f"File not found: {file_path}",
					category=ResultCategory.FILE_SYSTEM,
					context={"file_path": str(file_path)}
				)
			
			# Check file size
			file_size_mb = file_path.stat().st_size / (1024 * 1024)
			if file_size_mb > parameters.max_size_mb:
				return EnhancedActionResult.error_with_details(
					f"File too large: {file_size_mb:.1f}MB > {parameters.max_size_mb}MB",
					category=ResultCategory.VALIDATION,
					context={"file_size_mb": file_size_mb, "max_size_mb": parameters.max_size_mb}
				)
			
			# Read file content
			content = file_path.read_text(encoding=parameters.encoding)
			
			return EnhancedActionResult.success_with_data(
				data=content,
				result_type="file_content",
				summary=f"Successfully read {len(content)} characters from {file_path.name}",
				attachments={
					"file_info": {
						"path": str(file_path),
						"size_bytes": file_path.stat().st_size,
						"encoding": parameters.encoding,
						"line_count": content.count('\n') + 1
					}
				}
			)
		
		except UnicodeDecodeError as e:
			return EnhancedActionResult.error_with_details(
				f"Unable to decode file with encoding {parameters.encoding}: {str(e)}",
				category=ResultCategory.VALIDATION,
				context={"encoding": parameters.encoding, "file_path": parameters.file_path}
			)
		
		except PermissionError:
			return EnhancedActionResult.error_with_details(
				f"Permission denied reading file: {parameters.file_path}",
				category=ResultCategory.FILE_SYSTEM,
				context={"file_path": parameters.file_path}
			)
		
		except Exception as e:
			return EnhancedActionResult.error_with_details(
				e,
				category=ResultCategory.FILE_SYSTEM,
				context={"file_path": parameters.file_path}
			)


class WriteFileAction(BaseAction[WriteFileParameters, EnhancedActionResult, FileActionContext]):
	"""Action to write content to a file"""
	
	name: ClassVar[str] = "write_file"
	description: ClassVar[str] = "Write text content to a file"
	category: ClassVar[str] = ActionCategory.FILE_SYSTEM
	tags: ClassVar[set[str]] = {ActionTag.SIDE_EFFECTS, ActionTag.REQUIRES_FILE_SYSTEM}
	
	async def execute(
		self,
		parameters: WriteFileParameters,
		context: FileActionContext,
		**kwargs
	) -> EnhancedActionResult:
		"""Execute file writing"""
		try:
			# Get safe path
			file_path = context.get_safe_path(parameters.file_path)
			
			# Check if file exists and overwrite is not allowed
			if file_path.exists() and not parameters.overwrite:
				return EnhancedActionResult.error_with_details(
					f"File already exists and overwrite=False: {file_path}",
					category=ResultCategory.VALIDATION,
					context={"file_path": str(file_path), "overwrite": parameters.overwrite}
				)
			
			# Create parent directories if needed
			if parameters.create_directories:
				file_path.parent.mkdir(parents=True, exist_ok=True)
			
			# Write content to file
			file_path.write_text(parameters.content, encoding=parameters.encoding)
			
			# Get file info for response
			file_stat = file_path.stat()
			
			return EnhancedActionResult.success_with_data(
				data={"bytes_written": len(parameters.content.encode(parameters.encoding))},
				result_type="file_write",
				summary=f"Successfully wrote {len(parameters.content)} characters to {file_path.name}",
				attachments={
					"file_info": {
						"path": str(file_path),
						"size_bytes": file_stat.st_size,
						"encoding": parameters.encoding,
						"created": not file_path.exists() if not parameters.overwrite else False
					}
				}
			)
		
		except PermissionError:
			return EnhancedActionResult.error_with_details(
				f"Permission denied writing file: {parameters.file_path}",
				category=ResultCategory.FILE_SYSTEM,
				context={"file_path": parameters.file_path}
			)
		
		except OSError as e:
			return EnhancedActionResult.error_with_details(
				f"OS error writing file: {str(e)}",
				category=ResultCategory.FILE_SYSTEM,
				context={"file_path": parameters.file_path, "os_error": str(e)}
			)
		
		except Exception as e:
			return EnhancedActionResult.error_with_details(
				e,
				category=ResultCategory.FILE_SYSTEM,
				context={"file_path": parameters.file_path}
			)


class DeleteFileAction(BaseAction[DeleteFileParameters, EnhancedActionResult, FileActionContext]):
	"""Action to delete a file"""
	
	name: ClassVar[str] = "delete_file"
	description: ClassVar[str] = "Delete a file from the file system"
	category: ClassVar[str] = ActionCategory.FILE_SYSTEM
	tags: ClassVar[set[str]] = {ActionTag.DESTRUCTIVE, ActionTag.SIDE_EFFECTS, ActionTag.REQUIRES_FILE_SYSTEM}
	
	async def execute(
		self,
		parameters: DeleteFileParameters,
		context: FileActionContext,
		**kwargs
	) -> EnhancedActionResult:
		"""Execute file deletion"""
		try:
			# Get safe path
			file_path = context.get_safe_path(parameters.file_path)
			
			# Check if file exists
			if not file_path.exists():
				return EnhancedActionResult.error_with_details(
					f"File not found: {file_path}",
					category=ResultCategory.FILE_SYSTEM,
					context={"file_path": str(file_path)}
				)
			
			# Get file info before deletion
			file_stat = file_path.stat()
			file_info = {
				"path": str(file_path),
				"size_bytes": file_stat.st_size,
				"modification_time": file_stat.st_mtime
			}
			
			# Delete the file
			file_path.unlink()
			
			return EnhancedActionResult.success_with_data(
				data={"deleted": True},
				result_type="file_delete",
				summary=f"Successfully deleted file {file_path.name}",
				attachments={"deleted_file_info": file_info}
			)
		
		except PermissionError:
			return EnhancedActionResult.error_with_details(
				f"Permission denied deleting file: {parameters.file_path}",
				category=ResultCategory.FILE_SYSTEM,
				context={"file_path": parameters.file_path}
			)
		
		except OSError as e:
			return EnhancedActionResult.error_with_details(
				f"OS error deleting file: {str(e)}",
				category=ResultCategory.FILE_SYSTEM,
				context={"file_path": parameters.file_path, "os_error": str(e)}
			)
		
		except Exception as e:
			return EnhancedActionResult.error_with_details(
				e,
				category=ResultCategory.FILE_SYSTEM,
				context={"file_path": parameters.file_path}
			)