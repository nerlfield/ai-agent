"""
Advanced result handling system for actions with formatting, error propagation, and analysis.

This module extends the base ActionResult with comprehensive result management,
including result formatters, error handling, and result analysis tools.
"""

from __future__ import annotations

import json
import traceback
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.actions.base import ActionResult, ActionStatus

T = TypeVar('T')


class ResultSeverity(str, Enum):
	"""Severity levels for results and errors"""
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"
	CRITICAL = "critical"


class ResultCategory(str, Enum):
	"""Categories for organizing results"""
	EXECUTION = "execution"
	VALIDATION = "validation"
	NETWORK = "network"
	FILE_SYSTEM = "file_system"
	USER_INPUT = "user_input"
	SYSTEM = "system"
	BUSINESS_LOGIC = "business_logic"


class ErrorInfo(BaseModel):
	"""Detailed error information"""
	model_config = ConfigDict(extra='forbid')
	
	error_type: str
	error_message: str
	error_code: str | None = None
	severity: ResultSeverity = ResultSeverity.ERROR
	category: ResultCategory = ResultCategory.EXECUTION
	
	# Error context
	field_name: str | None = None
	field_value: Any = None
	stack_trace: str | None = None
	
	# Debugging information
	context: dict[str, Any] = Field(default_factory=dict)
	suggestions: list[str] = Field(default_factory=list)
	
	# Error propagation
	root_cause: ErrorInfo | None = None
	related_errors: list[ErrorInfo] = Field(default_factory=list)
	
	@classmethod
	def from_exception(
		cls,
		exception: Exception,
		severity: ResultSeverity = ResultSeverity.ERROR,
		category: ResultCategory = ResultCategory.EXECUTION,
		include_traceback: bool = True,
		context: dict[str, Any] | None = None
	) -> ErrorInfo:
		"""Create ErrorInfo from an exception"""
		return cls(
			error_type=type(exception).__name__,
			error_message=str(exception),
			severity=severity,
			category=category,
			stack_trace=traceback.format_exc() if include_traceback else None,
			context=context or {}
		)


class ResultMetrics(BaseModel):
	"""Metrics and performance data for action results"""
	model_config = ConfigDict(extra='forbid')
	
	# Timing metrics
	start_time: float | None = None
	end_time: float | None = None
	duration: float | None = None
	
	# Resource usage
	memory_peak: int | None = None  # bytes
	memory_average: int | None = None  # bytes
	cpu_time: float | None = None  # seconds
	
	# I/O metrics
	bytes_read: int | None = None
	bytes_written: int | None = None
	network_requests: int | None = None
	
	# Quality metrics
	accuracy: float | None = None  # 0.0 to 1.0
	confidence: float | None = None  # 0.0 to 1.0
	completeness: float | None = None  # 0.0 to 1.0
	
	# Custom metrics
	custom_metrics: dict[str, float] = Field(default_factory=dict)


class EnhancedActionResult(ActionResult):
	"""Enhanced action result with comprehensive error handling and analysis"""
	model_config = ConfigDict(extra='forbid')
	
	# Enhanced error handling
	errors: list[ErrorInfo] = Field(default_factory=list)
	warnings: list[ErrorInfo] = Field(default_factory=list)
	
	# Result metadata
	result_type: str | None = None
	result_category: ResultCategory = ResultCategory.EXECUTION
	result_tags: set[str] = Field(default_factory=set)
	
	# Performance and quality metrics
	metrics: ResultMetrics | None = None
	
	# Result formatting and display
	formatted_output: dict[str, str] = Field(default_factory=dict)
	summary: str | None = None
	
	# Debugging and analysis
	debug_info: dict[str, Any] = Field(default_factory=dict)
	analysis: dict[str, Any] = Field(default_factory=dict)
	
	# Override base properties to use enhanced error handling
	@property
	def error_message(self) -> str | None:
		"""Get the first error message if any"""
		if self.errors:
			return self.errors[0].error_message
		return super().error_message
	
	@property
	def error_code(self) -> str | None:
		"""Get the first error code if any"""
		if self.errors:
			return self.errors[0].error_code
		return super().error_code
	
	def add_error(
		self,
		error: str | Exception | ErrorInfo,
		severity: ResultSeverity = ResultSeverity.ERROR,
		category: ResultCategory = ResultCategory.EXECUTION,
		**kwargs
	) -> None:
		"""Add an error to the result"""
		if isinstance(error, ErrorInfo):
			error_info = error
		elif isinstance(error, Exception):
			error_info = ErrorInfo.from_exception(error, severity, category, **kwargs)
		else:
			error_info = ErrorInfo(
				error_type="GenericError",
				error_message=str(error),
				severity=severity,
				category=category,
				**kwargs
			)
		
		if severity in [ResultSeverity.ERROR, ResultSeverity.CRITICAL]:
			self.errors.append(error_info)
			self.success = False
			self.status = ActionStatus.FAILED
		else:
			self.warnings.append(error_info)
	
	def add_warning(
		self,
		warning: str | Exception | ErrorInfo,
		category: ResultCategory = ResultCategory.EXECUTION,
		**kwargs
	) -> None:
		"""Add a warning to the result"""
		self.add_error(warning, ResultSeverity.WARNING, category, **kwargs)
	
	def has_errors(self) -> bool:
		"""Check if result has any errors"""
		return len(self.errors) > 0
	
	def has_warnings(self) -> bool:
		"""Check if result has any warnings"""
		return len(self.warnings) > 0
	
	def get_error_summary(self) -> str:
		"""Get a summary of all errors"""
		if not self.errors:
			return "No errors"
		
		summary_parts = []
		for error in self.errors:
			summary_parts.append(f"[{error.severity.upper()}] {error.error_message}")
		
		return "; ".join(summary_parts)
	
	def get_all_issues(self) -> list[ErrorInfo]:
		"""Get all errors and warnings combined"""
		return self.errors + self.warnings
	
	@classmethod
	def success_with_data(
		cls,
		data: Any,
		result_type: str | None = None,
		summary: str | None = None,
		metrics: ResultMetrics | None = None,
		**kwargs
	) -> EnhancedActionResult:
		"""Create a successful result with data"""
		return cls(
			success=True,
			status=ActionStatus.SUCCESS,
			data=data,
			result_type=result_type,
			summary=summary,
			metrics=metrics,
			**kwargs
		)
	
	@classmethod
	def error_with_details(
		cls,
		error: str | Exception | ErrorInfo,
		category: ResultCategory = ResultCategory.EXECUTION,
		context: dict[str, Any] | None = None,
		**kwargs
	) -> EnhancedActionResult:
		"""Create an error result with detailed information"""
		result = cls(
			success=False,
			status=ActionStatus.FAILED,
			result_category=category,
			**kwargs
		)
		
		if context:
			result.debug_info.update(context)
		
		result.add_error(error, category=category)
		return result


class ResultFormatter:
	"""Formats action results for different output types"""
	
	@staticmethod
	def format_for_llm(result: EnhancedActionResult) -> str:
		"""Format result for LLM consumption"""
		parts = []
		
		# Status and summary
		status = "SUCCESS" if result.success else "FAILED"
		parts.append(f"Status: {status}")
		
		if result.summary:
			parts.append(f"Summary: {result.summary}")
		
		# Data (if not too large)
		if result.data is not None:
			data_str = str(result.data)
			if len(data_str) <= 500:
				parts.append(f"Data: {data_str}")
			else:
				parts.append(f"Data: <{len(data_str)} characters, {type(result.data).__name__}>")
		
		# Errors and warnings
		if result.errors:
			error_messages = [error.error_message for error in result.errors]
			parts.append(f"Errors: {'; '.join(error_messages)}")
		
		if result.warnings:
			warning_messages = [warning.error_message for warning in result.warnings]
			parts.append(f"Warnings: {'; '.join(warning_messages)}")
		
		# Metrics (if available)
		if result.metrics and result.metrics.duration:
			parts.append(f"Duration: {result.metrics.duration:.3f}s")
		
		return "\n".join(parts)
	
	@staticmethod
	def format_for_human(result: EnhancedActionResult) -> str:
		"""Format result for human consumption"""
		lines = []
		
		# Header
		if result.success:
			lines.append("✅ Action completed successfully")
		else:
			lines.append("❌ Action failed")
		
		if result.summary:
			lines.append(f"📋 {result.summary}")
		
		# Details
		if result.data is not None:
			lines.append(f"📊 Result: {result.data}")
		
		# Metrics
		if result.metrics:
			metrics_lines = []
			if result.metrics.duration:
				metrics_lines.append(f"⏱️  Duration: {result.metrics.duration:.3f}s")
			if result.metrics.memory_peak:
				metrics_lines.append(f"💾 Memory: {result.metrics.memory_peak / 1024 / 1024:.1f}MB")
			if result.metrics.accuracy:
				metrics_lines.append(f"🎯 Accuracy: {result.metrics.accuracy:.1%}")
			
			if metrics_lines:
				lines.extend(metrics_lines)
		
		# Issues
		if result.errors:
			lines.append("🚨 Errors:")
			for error in result.errors:
				lines.append(f"  • {error.error_message}")
		
		if result.warnings:
			lines.append("⚠️  Warnings:")
			for warning in result.warnings:
				lines.append(f"  • {warning.error_message}")
		
		return "\n".join(lines)
	
	@staticmethod
	def format_for_json(result: EnhancedActionResult) -> str:
		"""Format result as JSON"""
		data = result.model_dump()
		
		# Clean up for JSON serialization
		def clean_for_json(obj):
			if isinstance(obj, dict):
				return {k: clean_for_json(v) for k, v in obj.items()}
			elif isinstance(obj, list):
				return [clean_for_json(item) for item in obj]
			elif hasattr(obj, 'model_dump'):
				return obj.model_dump()
			else:
				return obj
		
		cleaned_data = clean_for_json(data)
		return json.dumps(cleaned_data, indent=2, default=str)


class ResultAnalyzer:
	"""Analyzes action results for patterns and insights"""
	
	@staticmethod
	def analyze_performance(results: list[EnhancedActionResult]) -> dict[str, Any]:
		"""Analyze performance metrics across multiple results"""
		if not results:
			return {"error": "No results to analyze"}
		
		durations = [r.metrics.duration for r in results if r.metrics and r.metrics.duration]
		success_rate = sum(1 for r in results if r.success) / len(results)
		
		analysis = {
			"total_results": len(results),
			"success_rate": success_rate,
			"failure_rate": 1 - success_rate,
		}
		
		if durations:
			analysis.update({
				"avg_duration": sum(durations) / len(durations),
				"min_duration": min(durations),
				"max_duration": max(durations),
			})
		
		# Error analysis
		all_errors = []
		for result in results:
			all_errors.extend(result.errors)
		
		if all_errors:
			error_types = {}
			error_categories = {}
			
			for error in all_errors:
				error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
				error_categories[error.category] = error_categories.get(error.category, 0) + 1
			
			analysis.update({
				"total_errors": len(all_errors),
				"common_error_types": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5],
				"error_categories": error_categories,
			})
		
		return analysis
	
	@staticmethod
	def get_failure_patterns(results: list[EnhancedActionResult]) -> dict[str, Any]:
		"""Identify patterns in failures"""
		failed_results = [r for r in results if not r.success]
		
		if not failed_results:
			return {"message": "No failures to analyze"}
		
		# Group by error types
		error_patterns = {}
		for result in failed_results:
			for error in result.errors:
				pattern_key = f"{error.error_type}:{error.category}"
				if pattern_key not in error_patterns:
					error_patterns[pattern_key] = {
						"count": 0,
						"messages": [],
						"contexts": [],
					}
				
				error_patterns[pattern_key]["count"] += 1
				error_patterns[pattern_key]["messages"].append(error.error_message)
				error_patterns[pattern_key]["contexts"].append(error.context)
		
		# Find most common patterns
		common_patterns = sorted(
			error_patterns.items(),
			key=lambda x: x[1]["count"],
			reverse=True
		)[:10]
		
		return {
			"total_failures": len(failed_results),
			"failure_rate": len(failed_results) / len(results) if results else 0,
			"common_patterns": common_patterns,
		}