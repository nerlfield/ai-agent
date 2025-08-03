import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

F = TypeVar('F', bound=Callable[..., Any])


def _is_debug_mode() -> bool:
	observability_debug = os.getenv('AI_AGENT_OBSERVABILITY_DEBUG', '').lower()
	if observability_debug in ('true', '1', 'yes'):
		return True
	return False


_LMNR_AVAILABLE = False
_lmnr_observe = None

try:
	from lmnr import observe as _lmnr_observe  # type: ignore
	_LMNR_AVAILABLE = True
	if os.environ.get('AI_AGENT_VERBOSE_OBSERVABILITY', 'false').lower() == 'true':
		logger.info('Lmnr is available for observability')
except ImportError:
	if os.environ.get('AI_AGENT_VERBOSE_OBSERVABILITY', 'false').lower() == 'true':
		logger.info('Lmnr is not available, using no-op observability decorators')


def observe(
	name: str | None = None,
	trace_id: str | None = None,
	session_id: str | None = None,
	user_id: str | None = None,
	version: str | None = None,
	input_serializer: Callable[[Any], Any] | None = None,
	output_serializer: Callable[[Any], Any] | None = None,
) -> Callable[[F], F]:
	if _LMNR_AVAILABLE and _lmnr_observe:
		return _lmnr_observe(
			name=name,
			trace_id=trace_id,
			session_id=session_id,
			user_id=user_id,
			version=version,
			input_serializer=input_serializer,
			output_serializer=output_serializer,
		)
	else:
		def no_op_decorator(func: F) -> F:
			return func
		return no_op_decorator


def observe_debug(
	name: str | None = None,
	trace_id: str | None = None,
	session_id: str | None = None,
	user_id: str | None = None,
	version: str | None = None,
	input_serializer: Callable[[Any], Any] | None = None,
	output_serializer: Callable[[Any], Any] | None = None,
) -> Callable[[F], F]:
	if _is_debug_mode():
		return observe(
			name=name,
			trace_id=trace_id,
			session_id=session_id,
			user_id=user_id,
			version=version,
			input_serializer=input_serializer,
			output_serializer=output_serializer,
		)
	else:
		def no_op_decorator(func: F) -> F:
			return func
		return no_op_decorator