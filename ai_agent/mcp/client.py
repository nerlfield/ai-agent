"""
MCP (Model Context Protocol) client implementation.

This module provides a client interface for communicating with MCP servers,
including tool execution, discovery, and connection management.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from ai_agent.mcp.models import (
	MCPResult,
	MCPResultStatus,
	MCPServerInfo,
	MCPTool,
)

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
	"""Base exception for MCP client errors"""
	pass


class MCPConnectionError(MCPClientError):
	"""MCP connection related errors"""
	pass


class MCPExecutionError(MCPClientError):
	"""MCP tool execution errors"""
	pass


class MCPTimeoutError(MCPClientError):
	"""MCP timeout errors"""
	pass


class MCPRequest(BaseModel):
	"""MCP request message"""
	model_config = ConfigDict(extra='forbid')
	
	id: str | int
	method: str
	params: dict[str, Any] = Field(default_factory=dict)
	jsonrpc: str = "2.0"


class MCPResponse(BaseModel):
	"""MCP response message"""
	model_config = ConfigDict(extra='allow')  # Allow extra fields for flexibility
	
	id: str | int
	jsonrpc: str = "2.0"
	result: dict[str, Any] | None = None
	error: dict[str, Any] | None = None
	
	@property
	def is_success(self) -> bool:
		"""Check if response indicates success"""
		return self.error is None
	
	@property
	def error_message(self) -> str | None:
		"""Get error message if present"""
		if self.error:
			return self.error.get('message')
		return None


class MCPClientConfig(BaseModel):
	"""Configuration for MCP client"""
	model_config = ConfigDict(extra='forbid')
	
	# Connection settings
	timeout_ms: int = 30000
	max_retries: int = 3
	retry_delay_ms: int = 1000
	
	# Protocol settings
	protocol_version: str = "1.0"
	
	# Feature support
	supports_streaming: bool = False
	supports_cancellation: bool = True
	
	# Logging
	log_requests: bool = False
	log_responses: bool = False


class MCPTransport(ABC):
	"""Abstract base class for MCP transport implementations"""
	
	@abstractmethod
	async def connect(self) -> None:
		"""Establish connection to MCP server"""
		pass
	
	@abstractmethod
	async def disconnect(self) -> None:
		"""Close connection to MCP server"""
		pass
	
	@abstractmethod
	async def send_request(self, request: MCPRequest) -> MCPResponse:
		"""Send request and wait for response"""
		pass
	
	@abstractmethod
	async def send_notification(self, method: str, params: dict[str, Any]) -> None:
		"""Send notification (no response expected)"""
		pass
	
	@property
	@abstractmethod
	def is_connected(self) -> bool:
		"""Check if transport is connected"""
		pass


class StdioTransport(MCPTransport):
	"""Standard I/O transport for MCP communication"""
	
	def __init__(self, command: list[str], cwd: str | None = None):
		self.command = command
		self.cwd = cwd
		self.process: asyncio.subprocess.Process | None = None
		self._request_counter = 0
		self._pending_requests: dict[str | int, asyncio.Future[MCPResponse]] = {}
		self._reader_task: asyncio.Task | None = None
	
	async def connect(self) -> None:
		"""Start subprocess and begin reading responses"""
		try:
			self.process = await asyncio.create_subprocess_exec(
				*self.command,
				stdin=asyncio.subprocess.PIPE,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=self.cwd
			)
			
			# Start background task to read responses
			self._reader_task = asyncio.create_task(self._read_responses())
			
			logger.info(f"Connected to MCP server via: {' '.join(self.command)}")
			
		except Exception as e:
			raise MCPConnectionError(f"Failed to start MCP server process: {e}") from e
	
	async def disconnect(self) -> None:
		"""Stop subprocess and cleanup"""
		if self._reader_task:
			self._reader_task.cancel()
			try:
				await self._reader_task
			except asyncio.CancelledError:
				pass
		
		if self.process:
			try:
				self.process.terminate()
				await asyncio.wait_for(self.process.wait(), timeout=5.0)
			except asyncio.TimeoutError:
				logger.warning("MCP server process did not terminate gracefully, killing")
				self.process.kill()
				await self.process.wait()
			except Exception as e:
				logger.error(f"Error terminating MCP server process: {e}")
			
			self.process = None
		
		# Cancel any pending requests
		for future in self._pending_requests.values():
			if not future.done():
				future.cancel()
		self._pending_requests.clear()
		
		logger.info("Disconnected from MCP server")
	
	async def send_request(self, request: MCPRequest) -> MCPResponse:
		"""Send request and wait for response"""
		if not self.is_connected:
			raise MCPConnectionError("Not connected to MCP server")
		
		# Assign request ID if not provided
		if not request.id:
			self._request_counter += 1
			request.id = self._request_counter
		
		# Create future for response
		response_future: asyncio.Future[MCPResponse] = asyncio.Future()
		self._pending_requests[request.id] = response_future
		
		try:
			# Send request
			request_data = request.model_dump_json() + '\n'
			self.process.stdin.write(request_data.encode())
			await self.process.stdin.drain()
			
			logger.debug(f"Sent MCP request: {request.method}")
			
			# Wait for response
			response = await response_future
			return response
			
		except Exception as e:
			# Clean up pending request
			self._pending_requests.pop(request.id, None)
			raise MCPExecutionError(f"Failed to send MCP request: {e}") from e
	
	async def send_notification(self, method: str, params: dict[str, Any]) -> None:
		"""Send notification (no response expected)"""
		if not self.is_connected:
			raise MCPConnectionError("Not connected to MCP server")
		
		notification = {
			"jsonrpc": "2.0",
			"method": method,
			"params": params
		}
		
		notification_data = json.dumps(notification) + '\n'
		self.process.stdin.write(notification_data.encode())
		await self.process.stdin.drain()
		
		logger.debug(f"Sent MCP notification: {method}")
	
	async def _read_responses(self) -> None:
		"""Background task to read responses from subprocess"""
		try:
			while self.process and self.process.stdout:
				line = await self.process.stdout.readline()
				if not line:
					break
				
				try:
					response_data = json.loads(line.decode().strip())
					response = MCPResponse(**response_data)
					
					# Find and resolve pending request
					future = self._pending_requests.pop(response.id, None)
					if future and not future.done():
						future.set_result(response)
					
					logger.debug(f"Received MCP response for ID: {response.id}")
					
				except Exception as e:
					logger.error(f"Failed to parse MCP response: {e}")
					logger.debug(f"Raw response: {line.decode().strip()}")
		
		except asyncio.CancelledError:
			logger.debug("MCP response reader task cancelled")
		except Exception as e:
			logger.error(f"Error in MCP response reader: {e}")
	
	@property
	def is_connected(self) -> bool:
		"""Check if transport is connected"""
		return self.process is not None and self.process.returncode is None


class MCPClient:
	"""Client for communicating with MCP servers"""
	
	def __init__(self, transport: MCPTransport, config: MCPClientConfig | None = None):
		self.transport = transport
		self.config = config or MCPClientConfig()
		self.server_info: MCPServerInfo | None = None
		self._connected = False
	
	@classmethod
	def create_stdio_client(
		cls,
		command: list[str],
		cwd: str | None = None,
		config: MCPClientConfig | None = None
	) -> MCPClient:
		"""Create client with stdio transport"""
		transport = StdioTransport(command, cwd)
		return cls(transport, config)
	
	async def connect(self) -> None:
		"""Connect to MCP server and initialize"""
		await self.transport.connect()
		
		try:
			# Initialize server
			response = await self._send_request("initialize", {
				"protocolVersion": self.config.protocol_version,
				"capabilities": {
					"tools": {"listChanged": True},
					"resources": {"subscribe": False, "listChanged": False},
					"prompts": {"listChanged": False}
				}
			})
			
			if not response.is_success:
				raise MCPConnectionError(f"Failed to initialize MCP server: {response.error_message}")
			
			# Parse server info
			server_capabilities = response.result.get("capabilities", {})
			self.server_info = MCPServerInfo(
				name=response.result.get("serverInfo", {}).get("name", "Unknown"),
				version=response.result.get("serverInfo", {}).get("version"),
				supports_tools=server_capabilities.get("tools") is not None,
				supports_resources=server_capabilities.get("resources") is not None,
				supports_prompts=server_capabilities.get("prompts") is not None,
			)
			
			# Send initialized notification
			await self.transport.send_notification("notifications/initialized", {})
			
			self._connected = True
			logger.info(f"Connected to MCP server: {self.server_info.name}")
			
		except Exception as e:
			await self.transport.disconnect()
			raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e
	
	async def disconnect(self) -> None:
		"""Disconnect from MCP server"""
		self._connected = False
		await self.transport.disconnect()
	
	@asynccontextmanager
	async def connection(self) -> AsyncIterator[MCPClient]:
		"""Context manager for MCP client connection"""
		await self.connect()
		try:
			yield self
		finally:
			await self.disconnect()
	
	async def list_tools(self) -> list[MCPTool]:
		"""List available tools from MCP server"""
		if not self._connected:
			raise MCPConnectionError("Not connected to MCP server")
		
		response = await self._send_request("tools/list", {})
		
		if not response.is_success:
			raise MCPExecutionError(f"Failed to list tools: {response.error_message}")
		
		tools = []
		tools_data = response.result.get("tools", [])
		
		for tool_data in tools_data:
			try:
				# Convert MCP tool schema to our format
				parameters = []
				input_schema = tool_data.get("inputSchema", {})
				properties = input_schema.get("properties", {})
				required_params = set(input_schema.get("required", []))
				
				for param_name, param_schema in properties.items():
					from ai_agent.mcp.models import MCPParameter, MCPDataType
					
					param = MCPParameter(
						name=param_name,
						type=MCPDataType(param_schema.get("type", "string")),
						description=param_schema.get("description"),
						required=param_name in required_params,
						default=param_schema.get("default")
					)
					parameters.append(param)
				
				tool = MCPTool(
					name=tool_data["name"],
					description=tool_data.get("description", ""),
					parameters=parameters,
					server_name=self.server_info.name if self.server_info else None
				)
				tools.append(tool)
				
			except Exception as e:
				logger.error(f"Failed to parse tool {tool_data.get('name', 'unknown')}: {e}")
				continue
		
		# Update server info with tools
		if self.server_info:
			self.server_info.tools = tools
		
		return tools
	
	async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPResult:
		"""Execute an MCP tool"""
		if not self._connected:
			raise MCPConnectionError("Not connected to MCP server")
		
		try:
			response = await self._send_request("tools/call", {
				"name": tool_name,
				"arguments": arguments
			})
			
			if not response.is_success:
				result = MCPResult(
					status=MCPResultStatus.ERROR,
					error_message=response.error_message,
					error_code=response.error.get("code") if response.error else None,
					server_name=self.server_info.name if self.server_info else None,
					tool_name=tool_name
				)
				return result
			
			# Convert response to MCPResult
			result_data = response.result
			result = MCPResult(
				status=MCPResultStatus.SUCCESS,
				server_name=self.server_info.name if self.server_info else None,
				tool_name=tool_name
			)
			
			# Process content
			content_list = result_data.get("content", [])
			if isinstance(content_list, list):
				from ai_agent.mcp.models import MCPContent
				for content_item in content_list:
					if isinstance(content_item, dict):
						content = MCPContent(
							type=content_item.get("type", "text"),
							data=content_item.get("text", content_item.get("data", "")),
							metadata=content_item.get("metadata", {})
						)
						result.content.append(content)
			
			# Extract value from result
			if "result" in result_data:
				result.value = result_data["result"]
			elif result.content and result.content[0].type == "text":
				result.value = result.content[0].data
			
			return result
			
		except asyncio.TimeoutError as e:
			return MCPResult(
				status=MCPResultStatus.TIMEOUT,
				error_message=f"Tool execution timed out after {self.config.timeout_ms}ms",
				server_name=self.server_info.name if self.server_info else None,
				tool_name=tool_name
			)
		except Exception as e:
			return MCPResult(
				status=MCPResultStatus.ERROR,
				error_message=str(e),
				server_name=self.server_info.name if self.server_info else None,
				tool_name=tool_name
			)
	
	async def _send_request(self, method: str, params: dict[str, Any]) -> MCPResponse:
		"""Send request with timeout and error handling"""
		request = MCPRequest(
			id=f"req_{asyncio.get_running_loop().time()}",
			method=method,
			params=params
		)
		
		if self.config.log_requests:
			logger.debug(f"MCP Request: {request.model_dump_json()}")
		
		try:
			response = await asyncio.wait_for(
				self.transport.send_request(request),
				timeout=self.config.timeout_ms / 1000.0
			)
			
			if self.config.log_responses:
				logger.debug(f"MCP Response: {response.model_dump_json()}")
			
			return response
			
		except asyncio.TimeoutError as e:
			raise MCPTimeoutError(f"MCP request timed out after {self.config.timeout_ms}ms") from e
	
	@property
	def is_connected(self) -> bool:
		"""Check if client is connected"""
		return self._connected and self.transport.is_connected


__all__ = [
	'MCPClient',
	'MCPClientConfig',
	'MCPTransport',
	'StdioTransport',
	'MCPClientError',
	'MCPConnectionError', 
	'MCPExecutionError',
	'MCPTimeoutError',
]