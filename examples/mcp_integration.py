"""Example demonstrating MCP client integration with ai-agent.

This example shows how to:
1. Set up an MCP client to connect to external MCP servers
2. Register MCP tools as ai-agent actions
3. Use those tools in an AI agent workflow
"""

import asyncio
import logging

from ai_agent.agent.service import Agent
from ai_agent.controller.service import Controller
from ai_agent.llm.openai.chat import OpenAIChat
from ai_agent.mcp.client import MCPClient, MCP_AVAILABLE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleContext:
	"""Simple context for demonstration purposes."""
	
	def __init__(self):
		self.data = {}
	
	def update(self, key: str, value: str):
		self.data[key] = value
	
	def get(self, key: str) -> str | None:
		return self.data.get(key)


async def main():
	"""Demonstrate MCP client integration with ai-agent."""
	
	if not MCP_AVAILABLE:
		logger.error("MCP SDK not available. Install with: pip install mcp")
		return
	
	# Create a simple context and controller
	context = SimpleContext()
	controller = Controller[SimpleContext]()
	
	# Example 1: Connect to a filesystem MCP server
	# You would need to have an MCP server running, e.g.:
	# npx @modelcontextprotocol/server-filesystem /path/to/directory
	
	logger.info("Setting up MCP client for filesystem server...")
	
	# For demonstration, we'll show the setup without actually connecting
	# since we don't have a real MCP server running
	filesystem_client = MCPClient(
		server_name="filesystem",
		command="npx",
		args=["@modelcontextprotocol/server-filesystem", "/tmp"]
	)
	
	# Example 2: Connect to a web search MCP server
	web_search_client = MCPClient(
		server_name="web-search",
		command="npx", 
		args=["@modelcontextprotocol/server-brave-search"]
	)
	
	try:
		# In a real scenario, you would connect and register:
		# await filesystem_client.connect()
		# await filesystem_client.register_to_controller(controller, prefix="fs_")
		
		# await web_search_client.connect()
		# await web_search_client.register_to_controller(controller, prefix="web_")
		
		logger.info("MCP clients would be connected and registered here")
		
		# Show what the available actions would look like
		logger.info("Available actions after MCP registration:")
		for action_name in controller.registry.registry.actions:
			action = controller.registry.registry.actions[action_name]
			logger.info(f"  - {action_name}: {action.description}")
		
		# Example 3: Using MCP tools in an agent workflow
		
		# Set up LLM (you would need your API key)
		# llm = OpenAIChat(model="gpt-4", api_key="your-api-key-here")
		
		# Create agent with MCP-enhanced controller
		# agent = Agent(
		#     llm=llm,
		#     controller=controller,
		#     context=context,
		#     task="Search for information about Python asyncio and save it to a file"
		# )
		
		# Run the agent - it would have access to both filesystem and web search tools
		# result = await agent.run(max_steps=10)
		
		logger.info("Agent would execute using both filesystem and web search MCP tools")
		
	except Exception as e:
		logger.error(f"Error in MCP integration example: {e}")
	
	finally:
		# Clean up connections
		# await filesystem_client.disconnect()
		# await web_search_client.disconnect()
		logger.info("MCP clients would be disconnected here")


async def demonstrate_mcp_tool_discovery():
	"""Show how MCP tool discovery works."""
	
	if not MCP_AVAILABLE:
		logger.error("MCP SDK not available. Install with: pip install mcp")
		return
	
	logger.info("=== MCP Tool Discovery Demo ===")
	
	# Example of connecting to different types of MCP servers
	mcp_servers = [
		{
			"name": "filesystem",
			"command": "npx",
			"args": ["@modelcontextprotocol/server-filesystem", "/tmp"],
			"description": "Provides file system operations"
		},
		{
			"name": "sqlite",
			"command": "npx", 
			"args": ["@modelcontextprotocol/server-sqlite", "database.db"],
			"description": "Provides SQLite database operations"
		},
		{
			"name": "github",
			"command": "npx",
			"args": ["@modelcontextprotocol/server-github"],
			"description": "Provides GitHub API access"
		}
	]
	
	controller = Controller()
	
	for server_config in mcp_servers:
		logger.info(f"\n--- Setting up {server_config['name']} MCP server ---")
		logger.info(f"Description: {server_config['description']}")
		logger.info(f"Command: {server_config['command']} {' '.join(server_config['args'])}")
		
		client = MCPClient(
			server_name=server_config['name'],
			command=server_config['command'],
			args=server_config['args']
		)
		
		# In a real scenario, you would:
		# 1. Connect to discover tools
		# 2. Register tools with optional prefix/filtering  
		# 3. Use the tools in your agent
		
		logger.info(f"Would connect to {server_config['name']} and discover available tools")
		logger.info(f"Tools would be registered with prefix '{server_config['name']}_'")
		
		# Example of how tools might be registered:
		# await client.register_to_controller(
		#     controller,
		#     prefix=f"{server_config['name']}_",
		#     tool_filter=None  # Register all tools
		# )


async def demonstrate_multiple_mcp_servers():
	"""Show how to use multiple MCP servers together."""
	
	logger.info("=== Multiple MCP Servers Demo ===")
	
	if not MCP_AVAILABLE:
		logger.error("MCP SDK not available. Install with: pip install mcp")
		return
	
	controller = Controller()
	
	# Create multiple MCP clients for different purposes
	clients = []
	
	# Data processing server
	data_client = MCPClient(
		server_name="data-processor",
		command="python",
		args=["-m", "my_data_mcp_server"]
	)
	clients.append(("data", data_client))
	
	# Web services server  
	web_client = MCPClient(
		server_name="web-services",
		command="node",
		args=["web_services_server.js"]
	)
	clients.append(("web", web_client))
	
	# File operations server
	file_client = MCPClient(
		server_name="file-ops", 
		command="npx",
		args=["@modelcontextprotocol/server-filesystem", "."]
	)
	clients.append(("file", file_client))
	
	try:
		# Connect and register all servers
		for prefix, client in clients:
			logger.info(f"Setting up {client.server_name} with prefix '{prefix}_'")
			
			# In real usage:
			# await client.connect()
			# await client.register_to_controller(controller, prefix=f"{prefix}_")
			
			logger.info(f"  - Tools from {client.server_name} would be available as {prefix}_*")
		
		# Now the agent would have access to tools like:
		# - data_process_csv, data_analyze_trends
		# - web_fetch_url, web_send_request  
		# - file_read_text, file_write_text, file_list_directory
		
		logger.info("\nAgent would have access to tools from all MCP servers:")
		logger.info("  - data_* tools for data processing")
		logger.info("  - web_* tools for web operations") 
		logger.info("  - file_* tools for file operations")
		
		# Example task that uses tools from multiple servers:
		example_task = """
		1. Use file_list_directory to find CSV files
		2. Use data_process_csv to analyze the data
		3. Use web_send_request to post results to an API
		4. Use file_write_text to save a summary report
		"""
		
		logger.info(f"\nExample multi-server task: {example_task}")
		
	finally:
		# Clean up all connections
		for _, client in clients:
			# await client.disconnect()
			logger.info(f"Would disconnect from {client.server_name}")


if __name__ == "__main__":
	print("MCP Integration Examples")
	print("========================")
	
	asyncio.run(main())
	print()
	asyncio.run(demonstrate_mcp_tool_discovery()) 
	print()
	asyncio.run(demonstrate_multiple_mcp_servers())