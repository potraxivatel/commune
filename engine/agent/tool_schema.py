"""
Tool Schema definitions for Commune AI.
Converts Component manifests into standard JSON-schema tool definitions
compatible with modern LLM tool-calling conventions (OpenAI / MCP).
"""

from __future__ import annotations

from typing import Any

from engine.core.component import Component


def component_to_tool_spec(comp: Component) -> dict[str, Any]:
    """
    Converts a Commune Component into a standard function declaration.
    """
    manifest = comp.manifest()
    name = manifest.name
    desc = manifest.description or f"Tool: {name}"

    # Default schemas for our standard deterministic tools
    properties: dict[str, Any] = {}
    required: list[str] = []

    if name == "read_file":
        properties = {
            "path": {"type": "string", "description": "Path to the file to read"},
            "offset": {"type": "integer", "description": "Starting line index (0-based)", "default": 0},
            "limit": {"type": "integer", "description": "Maximum number of lines to read"},
        }
        required = ["path"]
    elif name == "write_file":
        properties = {
            "path": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content string to write into the file"},
            "append": {"type": "boolean", "description": "If true, append to existing file", "default": False},
        }
        required = ["path", "content"]
    elif name == "run_command":
        properties = {
            "command": {"type": "string", "description": "Shell command to execute in bash"},
            "timeout": {"type": "number", "description": "Execution timeout in seconds", "default": 30.0},
        }
        required = ["command"]
    elif name == "list_dir":
        properties = {
            "path": {"type": "string", "description": "Directory path to list", "default": "."},
            "pattern": {"type": "string", "description": "Glob filter (e.g. *.py)", "default": "*"},
            "recursive": {"type": "boolean", "description": "Traverse subdirectories", "default": False},
        }
    else:
        properties = {
            "input": {"type": "string", "description": "Input payload for this component"}
        }

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
