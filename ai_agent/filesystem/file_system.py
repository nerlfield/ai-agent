from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FileInfo(BaseModel):
	"""Information about a file"""
	path: str
	size: int
	is_directory: bool
	exists: bool


class FileSystem:
	"""File system operations for the agent"""
	
	def __init__(self, working_directory: str | Path | None = None):
		self.working_directory = Path(working_directory) if working_directory else Path.cwd()
		self.working_directory = self.working_directory.resolve()
		
		# Ensure working directory exists
		self.working_directory.mkdir(parents=True, exist_ok=True)
		
		# Create standard files
		self.todo_file = self.working_directory / 'todo.md'
		self.index_file = self.working_directory / 'index.md'
		
		# Initialize files if they don't exist
		if not self.todo_file.exists():
			self.todo_file.write_text('# TODOs\n\n')
		if not self.index_file.exists():
			self.index_file.write_text('# Index\n\n')
	
	def describe(self) -> str:
		"""Get a description of the file system state"""
		files = list(self.working_directory.glob('*'))
		file_count = len([f for f in files if f.is_file()])
		dir_count = len([f for f in files if f.is_dir()])
		
		description = f"Working directory: {self.working_directory}\n"
		description += f"Files: {file_count}, Directories: {dir_count}\n"
		description += "Key files:\n"
		description += f"  - todo.md: {self.todo_file.stat().st_size} bytes\n"
		description += f"  - index.md: {self.index_file.stat().st_size} bytes"
		
		return description
	
	def get_todo_contents(self) -> str:
		"""Get the contents of the todo file"""
		if self.todo_file.exists():
			return self.todo_file.read_text()
		return ''
	
	def update_todo(self, content: str) -> None:
		"""Update the todo file"""
		self.todo_file.write_text(content)
	
	def read_file(self, path: str) -> str:
		"""Read a file relative to working directory"""
		file_path = self._resolve_path(path)
		if not file_path.exists():
			raise FileNotFoundError(f"File not found: {path}")
		return file_path.read_text()
	
	def write_file(self, path: str, content: str) -> None:
		"""Write a file relative to working directory"""
		file_path = self._resolve_path(path)
		file_path.parent.mkdir(parents=True, exist_ok=True)
		file_path.write_text(content)
	
	def append_to_file(self, path: str, content: str) -> None:
		"""Append to a file"""
		file_path = self._resolve_path(path)
		file_path.parent.mkdir(parents=True, exist_ok=True)
		
		current = file_path.read_text() if file_path.exists() else ''
		file_path.write_text(current + content)
	
	def delete_file(self, path: str) -> None:
		"""Delete a file"""
		file_path = self._resolve_path(path)
		if file_path.exists():
			file_path.unlink()
	
	def list_files(self, pattern: str = '*') -> list[FileInfo]:
		"""List files in working directory"""
		files = []
		for path in self.working_directory.glob(pattern):
			relative_path = path.relative_to(self.working_directory)
			files.append(FileInfo(
				path=str(relative_path),
				size=path.stat().st_size if path.exists() else 0,
				is_directory=path.is_dir(),
				exists=path.exists(),
			))
		return files
	
	def file_exists(self, path: str) -> bool:
		"""Check if a file exists"""
		file_path = self._resolve_path(path)
		return file_path.exists()
	
	def create_directory(self, path: str) -> None:
		"""Create a directory"""
		dir_path = self._resolve_path(path)
		dir_path.mkdir(parents=True, exist_ok=True)
	
	def save_json(self, path: str, data: Any) -> None:
		"""Save data as JSON"""
		file_path = self._resolve_path(path)
		file_path.parent.mkdir(parents=True, exist_ok=True)
		file_path.write_text(json.dumps(data, indent=2))
	
	def load_json(self, path: str) -> Any:
		"""Load JSON data"""
		file_path = self._resolve_path(path)
		if not file_path.exists():
			raise FileNotFoundError(f"File not found: {path}")
		return json.loads(file_path.read_text())
	
	def _resolve_path(self, path: str) -> Path:
		"""Resolve path relative to working directory"""
		# Prevent path traversal attacks
		if '..' in path:
			raise ValueError("Path traversal not allowed")
		
		file_path = Path(path)
		if file_path.is_absolute():
			# Ensure absolute paths are within working directory
			try:
				file_path.relative_to(self.working_directory)
			except ValueError:
				raise ValueError(f"Path must be within working directory: {path}")
			return file_path
		else:
			# Make relative paths relative to working directory
			return self.working_directory / path