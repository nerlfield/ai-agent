"""Test MCP client integration with ai-agent system."""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path

from ai_agent.controller.service import Controller
from ai_agent.mcp.client import MCP_AVAILABLE, MCPClient


# Skip all tests if MCP is not available
pytestmark = pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP SDK not available")


# Simple test MCP server code to create a temporary server
TEST_SERVER_CODE = '''
import asyncio
import json
import sys
import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions


class TestMCPServer:
	"""A minimal MCP server for testing."""

	def __init__(self):
		self.server = Server('test-mcp-server')
		self._setup_handlers()

	def _setup_handlers(self):
		"""Setup MCP server handlers."""

		@self.server.list_tools()
		async def handle_list_tools() -> list[types.Tool]:
			"""List available test tools."""
			return [
				types.Tool(
					name='echo_text',
					description='Echo back the provided text',
					inputSchema={{
						'type': 'object',
						'properties': {{'text': {{'type': 'string', 'description': 'Text to echo'}}}},
						'required': ['text'],
					}},
				),
				types.Tool(
					name='get_info',
					description='Get server information',
					inputSchema={{'type': 'object', 'properties': {{}}}},
				),
			]

		@self.server.call_tool()
		async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
			"""Handle tool execution."""
			if name == 'echo_text':
				text = arguments.get('text', 'No text provided') if arguments else 'No text provided'
				result = f'Echoed: {{text}}'
			elif name == 'get_info':
				result = json.dumps({{'server': 'test-mcp-server', 'status': 'active'}})
			else:
				result = f'Unknown tool: {{name}}'

			return [types.TextContent(type='text', text=result)]

	async def run(self):
		"""Run the MCP server."""
		async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
			await self.server.run(
				read_stream,
				write_stream,
				InitializationOptions(
					server_name='test-mcp-server',
					server_version='0.1.0',
					capabilities=self.server.get_capabilities(
						notification_options=NotificationOptions(),
						experimental_capabilities={{}},
					),
				),
			)


async def main():
	server = TestMCPServer()
	await server.run()


if __name__ == "__main__":
	asyncio.run(main())
'''


@pytest.fixture
async def test_mcp_server_script():
	"""Create a temporary script file for the test MCP server."""
	import sys
	with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
		f.write(TEST_SERVER_CODE)
		script_path = f.name

	yield script_path

	Path(script_path).unlink(missing_ok=True)


async def test_mcp_client_basic_connection(test_mcp_server_script):
	"""Test basic MCP client connection and tool discovery."""
	import sys
	
	controller = Controller()
	mcp_client = MCPClient(
		server_name='test-server',
		command=sys.executable,
		args=[test_mcp_server_script]
	)

	try:
		# Connect to server
		await mcp_client.connect()

		# Verify tools were discovered
		assert len(mcp_client._tools) == 2
		assert 'echo_text' in mcp_client._tools
		assert 'get_info' in mcp_client._tools

		# Register tools to controller
		await mcp_client.register_to_controller(controller)

		# Verify tools are registered as actions
		actions = controller.registry.registry.actions
		assert 'echo_text' in actions
		assert 'get_info' in actions

		# Test executing a tool with parameters
		echo_action = actions['echo_text']
		params = echo_action.param_model(text='Hello MCP!')
		result = await echo_action.function(params=params)

		assert result.success is True
		assert 'Echoed: Hello MCP!' in result.extracted_content
		assert 'echo_text' in result.long_term_memory
		assert 'test-server' in result.long_term_memory

		# Test executing a tool without parameters
		info_action = actions['get_info']
		result = await info_action.function()

		assert result.success is True
		assert 'test-mcp-server' in result.extracted_content
		assert 'get_info' in result.long_term_memory

	finally:
		await mcp_client.disconnect()


async def test_mcp_client_context_manager(test_mcp_server_script):
	"""Test using MCP client as context manager."""
	import sys
	
	controller = Controller()

	# Use as context manager
	async with MCPClient(
		server_name='test-server',
		command=sys.executable,
		args=[test_mcp_server_script]
	) as mcp_client:
		# Should auto-connect
		assert mcp_client.session is not None

		await mcp_client.register_to_controller(controller)

		# Test tool works
		action = controller.registry.registry.actions['get_info']
		result = await action.function()
		assert 'test-mcp-server' in result.extracted_content

	# Should auto-disconnect
	assert mcp_client.session is None


async def test_mcp_client_error_handling(test_mcp_server_script):
	"""Test error handling in MCP client."""
	import sys
	
	controller = Controller()
	mcp_client = MCPClient(
		server_name='test-server',
		command=sys.executable,
		args=[test_mcp_server_script]
	)

	try:
		await mcp_client.connect()
		await mcp_client.register_to_controller(controller)

		# Disconnect to simulate connection loss
		await mcp_client.disconnect()

		# Try to use tool after disconnect
		echo_action = controller.registry.registry.actions['echo_text']
		params = echo_action.param_model(text='Test')
		result = await echo_action.function(params=params)

		# Should handle error gracefully
		assert result.success is False
		assert len(result.errors) > 0
		assert 'not connected' in result.errors[0]

	finally:
		# Already disconnected
		pass


async def test_mcp_client_with_prefix_and_filter(test_mcp_server_script):
	"""Test tool filtering and prefixing."""
	import sys
	
	controller = Controller()
	mcp_client = MCPClient(
		server_name='test-server',
		command=sys.executable,
		args=[test_mcp_server_script]
	)

	try:
		await mcp_client.connect()

		# Register only specific tools with prefix
		await mcp_client.register_to_controller(
			controller, 
			tool_filter=['echo_text'], 
			prefix='mcp_'
		)

		# Verify registration
		actions = controller.registry.registry.actions
		assert 'mcp_echo_text' in actions
		assert 'mcp_get_info' not in actions  # Filtered out
		assert 'get_info' not in actions  # Not registered without prefix

		# Test prefixed action works
		action = actions['mcp_echo_text']
		params = action.param_model(text='Prefixed test')
		result = await action.function(params=params)
		assert 'Echoed: Prefixed test' in result.extracted_content

	finally:
		await mcp_client.disconnect()


if __name__ == "__main__":
	pytest.main([__file__, "-v"])