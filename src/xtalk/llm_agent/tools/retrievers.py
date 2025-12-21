from __future__ import annotations

from typing import Any, Dict, Optional
import os
from langchain.tools import tool, BaseTool
from langchain_chroma import Chroma  # type: ignore


def _resolve_env(*names: str) -> Optional[str]:
    """Return the first non-empty environment variable value among the names."""
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return None


WEB_SEARCH_TOOL = "web_search"


def build_web_search_tool() -> BaseTool:
    """Build a Serper-based web search tool with graceful degradation."""

    # JSON schema for tool arguments
    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string."},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
                "description": "Max number of results (1-10).",
            },
            "region": {
                "type": "string",
                "description": "Region code, e.g. 'us', 'cn', 'gb'. Optional.",
            },
            "lang": {
                "type": "string",
                "description": "Language code, e.g. 'en', 'zh-CN'. Optional.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    resolved_key = _resolve_env("SERPER_API_KEY", "GOOGLE_SERPER_API_KEY") or ""

    @tool(WEB_SEARCH_TOOL, args_schema=args_schema)
    def web_search(
        query: str,
        max_results: int = 5,
        region: str | None = None,
        lang: str | None = None,
    ) -> str:
        """Search the web and return a concise textual summary of top results."""
        # Lazy import requests so the module loads even if dependency is missing
        try:
            import requests  # type: ignore
        except Exception:
            return "Search service is unavailable: missing requests package."

        if not resolved_key:
            return "Search service is unavailable: missing SERPER_API_KEY."

        try:
            headers = {
                "X-API-KEY": resolved_key,
                "Content-Type": "application/json",
            }
            payload: Dict[str, Any] = {"q": query}
            if region:
                payload["location"] = region
            if lang:
                payload["gl"] = region or "us"
                payload["hl"] = lang

            resp = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload,
                timeout=12,
            )
            data = resp.json() if resp.ok else {}
            items = (data.get("organic") or [])[
                : max(1, min(10, int(max_results or 5)))
            ]
            if not items:
                return "No relevant results found."
            lines = []
            for i, it in enumerate(items, 1):
                title = (it.get("title") or "").strip()
                snippet = (it.get("snippet") or "").strip()
                link = (it.get("link") or "").strip()
                lines.append(f"{i}. {title} — {snippet}\n{link}")
            return "\n".join(lines)
        except Exception as e:  # fail-safe
            return f"Search error: {e}"

    return web_search


TIME_TOOL = "get_time"


def build_time_tool() -> BaseTool:
    """Build a current-time tool that supports optional timezone/format."""

    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone, e.g. 'UTC', 'America/Los_Angeles'. Optional.",
            },
            "fmt": {
                "type": "string",
                "description": "strftime format, default '%Y-%m-%d %H:%M:%S'.",
                "default": "%Y-%m-%d %H:%M:%S",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    @tool(TIME_TOOL, args_schema=args_schema)
    def get_time(timezone: str | None = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Return current datetime as a string with optional timezone and format."""
        from datetime import datetime

        tzinfo = None
        if timezone:
            try:
                from zoneinfo import ZoneInfo  # Python 3.9+

                tzinfo = ZoneInfo(timezone)
            except Exception:
                tzinfo = None

        now = datetime.now(tzinfo)
        try:
            return now.strftime(fmt)
        except Exception:
            return now.strftime("%Y-%m-%d %H:%M:%S")

    return get_time


LOCAL_SEARCH_TOOL = "local_search"


def build_local_search_tool(
    db: Chroma,
) -> BaseTool:
    """Build a Chroma-based local vector search tool (read-only).

    Args:
        db: Initialized Chroma instance reused for retrieval.
    """

    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Query string."},
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 4,
                "description": "Number of results to return.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    @tool(LOCAL_SEARCH_TOOL, args_schema=args_schema)
    def local_search(query: str, k: int = 4) -> str:
        """
        Search in user uploaded documents. Always try to use this tool if user's query is related to their uploaded docs.
        """
        try:
            if db is None:
                return (
                    "Search service is unavailable: missing Chroma database instance."
                )

            search_kwargs = {"k": int(max(1, min(50, int(k))))}
            retriever = db.as_retriever(
                search_type="similarity", search_kwargs=search_kwargs
            )

            contexts = []
            try:
                if hasattr(retriever, "get_relevant_documents"):
                    contexts = retriever.get_relevant_documents(query)
                elif hasattr(retriever, "invoke"):
                    contexts = retriever.invoke(query)
                elif callable(retriever):
                    contexts = retriever(query)
                else:
                    return "Retriever does not expose a supported retrieval method."
            except Exception:
                try:
                    contexts = retriever.invoke(query)
                except Exception as e:
                    return f"Error during database retrieval: {e}"

            texts = [getattr(r, "page_content", str(r)) for r in contexts]
            return "\n\n".join(texts) if texts else "No relevant information found."
        except Exception as e:
            return f"Error during database retrieval: {e}"

    return local_search
