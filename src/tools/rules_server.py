import json, os, sys, logging
from pathlib import Path
from typing import Optional, Dict

from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("rules")

RULES_FILE = Path(os.path.expanduser(".agent_rules.json"))

def _load_rules() -> Dict[str, str]:
    if RULES_FILE.exists():
        try:
            with RULES_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
        except Exception as e:
            logging.error("Failed to read rules file: %s", e)
    return {}

def _save_rules(rules: Dict[str, str]) -> None:
    RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RULES_FILE.open("w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

@mcp.tool()
async def list_rules() -> Dict[str, str]:
    """Return all rules as a {name: value} JSON object."""
    return _load_rules()

@mcp.tool()
async def get_rule(name: str) -> Optional[str]:
    """Return a single rule value by name, or null if missing."""
    return _load_rules().get(name)

@mcp.tool()
async def set_rule(name: str, value: str) -> str:
    """Create or update a rule: set_rule(name, value) -> 'ok'."""
    rules = _load_rules()
    rules[name] = value
    _save_rules(rules)
    return "ok"

@mcp.tool()
async def delete_rule(name: str) -> str:
    """Delete a rule by name. Returns 'deleted' or 'not_found'."""
    rules = _load_rules()
    if name in rules:
        del rules[name]
        _save_rules(rules)
        return "deleted"
    return "not_found"

if __name__ == "__main__":
    mcp.run(transport="stdio")