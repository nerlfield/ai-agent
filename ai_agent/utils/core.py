import asyncio
import logging
import os
import platform
import signal
import time
from collections.abc import Callable, Coroutine
from fnmatch import fnmatch
from functools import cache, wraps
from pathlib import Path
from sys import stderr
from typing import Any, ParamSpec, TypeVar
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_exiting = False

R = TypeVar('R')
T = TypeVar('T')
P = ParamSpec('P')


class SignalHandler:
	def __init__(
		self,
		loop: asyncio.AbstractEventLoop | None = None,
		pause_callback: Callable[[], None] | None = None,
		resume_callback: Callable[[], None] | None = None,
		custom_exit_callback: Callable[[], None] | None = None,
		exit_on_second_int: bool = True,
		interruptible_task_patterns: list[str] | None = None,
	):
		self.loop = loop or asyncio.get_event_loop()
		self.pause_callback = pause_callback
		self.resume_callback = resume_callback
		self.custom_exit_callback = custom_exit_callback
		self.exit_on_second_int = exit_on_second_int
		self.interruptible_task_patterns = interruptible_task_patterns or ['step', 'execute', 'process']
		self.is_windows = platform.system() == 'Windows'

		self._initialize_loop_state()

		self.original_sigint_handler = None
		self.original_sigterm_handler = None

	def _initialize_loop_state(self) -> None:
		setattr(self.loop, 'ctrl_c_pressed', False)
		setattr(self.loop, 'waiting_for_input', False)

	def register(self) -> None:
		try:
			if self.is_windows:
				signal.signal(signal.SIGINT, self._handle_signal_wrapper)
			else:
				self.loop.add_signal_handler(signal.SIGINT, self._handle_signal_wrapper, signal.SIGINT)
				self.loop.add_signal_handler(signal.SIGTERM, self._handle_signal_wrapper, signal.SIGTERM)
		except Exception as e:
			logger.warning(f'Failed to register signal handlers: {e}')

	def _handle_signal_wrapper(self, signum=signal.SIGINT):
		asyncio.create_task(self._handle_signal(signum))

	async def _handle_signal(self, signum):
		global _exiting

		if _exiting:
			return

		if signum == signal.SIGINT:
			ctrl_c_pressed = getattr(self.loop, 'ctrl_c_pressed', False)
			
			if not ctrl_c_pressed:
				setattr(self.loop, 'ctrl_c_pressed', True)
				logger.info('🛑 Pausing... (Press Ctrl+C again to exit)')
				
				if self.pause_callback:
					self.pause_callback()
				
				await self._cancel_interruptible_tasks()
			else:
				if self.exit_on_second_int:
					await self._exit_gracefully()
		elif signum == signal.SIGTERM:
			await self._exit_gracefully()

	async def _cancel_interruptible_tasks(self):
		tasks = [task for task in asyncio.all_tasks(self.loop) if not task.done()]
		
		for task in tasks:
			task_name = getattr(task, 'get_name', lambda: '')()
			if any(pattern in task_name.lower() for pattern in self.interruptible_task_patterns):
				task.cancel()

	async def _exit_gracefully(self):
		global _exiting
		_exiting = True
		
		logger.info('🔄 Shutting down gracefully...')
		
		if self.custom_exit_callback:
			self.custom_exit_callback()
		
		tasks = [task for task in asyncio.all_tasks(self.loop) if not task.done() and task != asyncio.current_task()]
		if tasks:
			for task in tasks:
				task.cancel()
			await asyncio.gather(*tasks, return_exceptions=True)
		
		self.loop.stop()


def time_execution_sync(func: Callable[P, R]) -> Callable[P, R]:
	@wraps(func)
	def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
		start_time = time.time()
		result = func(*args, **kwargs)
		duration = time.time() - start_time
		
		func_name = getattr(func, '__name__', 'unknown')
		if duration > 1.0:
			logger.debug(f'⏱️  {func_name} took {duration:.2f}s')
		
		return result
	return wrapper


def time_execution_async(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
	@wraps(func)
	async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
		start_time = time.time()
		result = await func(*args, **kwargs)
		duration = time.time() - start_time
		
		func_name = getattr(func, '__name__', 'unknown')
		if duration > 1.0:
			logger.debug(f'⏱️  {func_name} took {duration:.2f}s')
		
		return result
	return wrapper


def _log_pretty_path(path: Path) -> str:
	try:
		return str(path.relative_to(Path.cwd()))
	except ValueError:
		return str(path)


def get_ai_agent_version() -> str:
	try:
		from importlib.metadata import version
		return version('ai-agent')
	except Exception:
		return 'unknown'


def get_git_info() -> dict[str, str]:
	info = {}
	try:
		import subprocess
		
		result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
							  capture_output=True, text=True, timeout=5)
		if result.returncode == 0:
			info['commit'] = result.stdout.strip()
		
		result = subprocess.run(['git', 'branch', '--show-current'], 
							  capture_output=True, text=True, timeout=5)
		if result.returncode == 0:
			info['branch'] = result.stdout.strip()
			
	except Exception:
		pass
	
	return info


def match_url_with_domain_pattern(url: str, domain_pattern: str, log_warnings: bool = False) -> bool:
	if not url or not domain_pattern:
		return False
	
	try:
		parsed = urlparse(url)
		hostname = parsed.hostname
		
		if not hostname:
			return False
		
		return fnmatch(hostname, domain_pattern)
		
	except Exception as e:
		if log_warnings:
			logger.warning(f'Failed to match URL {url} with pattern {domain_pattern}: {e}')
		return False


def is_complex_domain_pattern(pattern: str) -> bool:
	if not pattern:
		return False
	
	if '://' in pattern:
		pattern = urlparse(pattern).hostname or pattern
	
	bare_domain = pattern.replace('.*', '').replace('*.', '')
	
	return '*' in bare_domain