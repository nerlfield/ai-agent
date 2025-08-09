# AI Agent with MCP Integration

An intelligent conversational agent powered by OpenAI's GPT-5 model with Model Context Protocol (MCP) tool integration. The agent provides an interactive CLI interface with rich formatting and supports extensible tool capabilities through MCP servers.

## Architecture Overview

```mermaid
graph TB
    User([User Input]) --> App[app.py<br/>CLI Interface]
    App --> Agent[Agent<br/>src/service.py]
    Agent --> OpenAI[OpenAI API<br/>GPT-5 Model]
    Agent --> MCP[MCP Manager<br/>src/mcp_bridge.py]
    MCP --> Rules[Rules Server<br/>src/tools/rules_server.py]
    Rules --> RulesFile[.agent_rules.json]
    Config[src/config.py] --> Agent
    
    subgraph "Rich Console UI"
        UserPanel[User Messages]
        AssistantPanel[Assistant Responses]
        ToolPanel[Tool Call Display]
        ReasoningPanel[Reasoning Display]
    end
    
    App --> UserPanel
    App --> AssistantPanel
    App --> ToolPanel
    App --> ReasoningPanel
```

## Module Breakdown

### Core Components

- **`app.py`** - Main CLI application entry point with rich console interface
- **`src/service.py`** - Core Agent class handling OpenAI API communication and message processing
- **`src/mcp_bridge.py`** - MCP server management and tool integration bridge
- **`src/config.py`** - Configuration management and environment variables
- **`src/tools/rules_server.py`** - MCP server for persistent agent rule management

### Key Features

- **Interactive CLI**: Rich-formatted console interface with color-coded message panels
- **MCP Integration**: Extensible tool system using Model Context Protocol
- **Persistent Rules**: Dynamic agent behavior configuration via JSON storage
- **Streaming Responses**: Real-time response processing with tool call visualization
- **Signal Handling**: Graceful shutdown on interruption

## Installation & Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. **Create and activate conda environment**
   ```bash
   conda create -n aimcpagent python=3.11
   conda activate aimcpagent
   ```

2. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-agent
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY='sk-your-api-key-here'
   ```

### Dependencies

```
openai==1.99.6      # OpenAI API client
rich==14.1.0        # Rich console formatting
python-dotenv==1.1.1 # Environment variable management
mcp==1.12.4         # Model Context Protocol
```

## Usage

### Starting the Agent

```bash
python app.py
```

### Basic Interaction

The agent starts with an interactive prompt:

```
Agent (GPT-5 + MCP) started. Type 'exit' to quit.

> Hello, how can you help me?
```

### Available Commands

- **Chat**: Type any message to interact with the agent
- **Exit**: Type `exit` or `quit` to terminate the session
- **Tool Usage**: The agent automatically uses available MCP tools when appropriate

### Rule Management

The agent includes a built-in rules system for persistent behavior configuration:

- **View rules**: Agent can display current rules via MCP tools
- **Set rules**: Modify agent behavior (tone, formatting, explanations)
- **Delete rules**: Remove specific behavioral rules

Current default rules:
- **Tone**: Concise, friendly
- **Formatting**: Prefer bullets, minimal markup  
- **Explanations**: Default brief; add detail on request

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_MODEL` - Model to use (default: "gpt-5")

### Rule Storage

Agent rules are stored in `.agent_rules.json` in the project root and persist across sessions.

## Development

### Project Structure

```
ai-agent/
├── app.py                    # CLI entry point
├── src/
│   ├── config.py            # Configuration management
│   ├── service.py           # Core agent logic
│   ├── mcp_bridge.py        # MCP integration
│   └── tools/
│       └── rules_server.py  # Rules management server
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
└── .agent_rules.json       # Persistent agent rules
```

### Adding New MCP Servers

1. Create new server file in `src/tools/`
2. Implement using FastMCP framework
3. Add server path to initialization in `app.py:28`

### Extending the Agent

The agent architecture supports:
- Multiple MCP servers
- Custom tool implementations  
- Configurable response formatting
- Persistent state management