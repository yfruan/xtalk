from __future__ import annotations

"""
Helper utilities unrelated to agent decision logic.

This module defines a constructor for a pseudo tool named "tool_call_result".
It wraps each finished tool invocation into a consistent payload so upstream
components (e.g., Pipeline/TTSManager) can consume the result and downstream
modules (e.g., RetrievalManager) can react accordingly.

Notes:
- The tool is NOT registered with the LLM (models never call it directly).
- The payload simply tags a tool result with:
  name: original tool name (e.g., "web_search")
  args: original arguments (verbatim)
  content: textual output from the tool
"""

from typing import Any, Dict


def build_tool_call_result_payload(
    *, name: str, args: Dict[str, Any] | None, content: str
) -> Dict[str, Any]:
    """Build a normalized payload for tool_call_result events.

    Args:
        name: Original tool name (e.g., "web_search")
        args: Original tool arguments (dict or None)
        content: Tool textual result

    Returns:
        Dict formatted as {"name": "tool_call_result", "args": {...}}
    """
    return {
        "name": "tool_call_result",
        "args": {
            "name": name or "",
            "args": dict(args or {}),
            "content": content or "",
        },
    }
