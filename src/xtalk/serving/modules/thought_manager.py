# -*- coding: utf-8 -*-
"""
ThoughtManager periodically rewrites the conversation history plus the previous
thought into a concise internal summary. It updates Pipeline context and emits
ThoughtUpdated events at a fixed cadence.

Notes:
- Requires both an agent (for chat history) and a thought rewriter.
- Operates passively; does not mutate agents beyond writing to context.
- Simply skips work when dependencies are missing.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Any

from ...log_utils import logger
from ..event_bus import EventBus
from ..interfaces import Manager
from ..events import ThoughtUpdated
from ...pipelines import Pipeline
from ...rewriter.interfaces import Rewriter


class ThoughtManager(Manager):
    """Manager that periodically refreshes conversation thoughts."""

    REFRESH_INTERVAL_SEC: float = 1.0  # Refresh cadence

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        pipeline: Pipeline,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.pipeline = pipeline
        self.config: dict[str, Any] = config or {}

        # Dependencies: agent & rewriter
        self.agent = pipeline.get_agent()
        self.rewriter: Optional[Rewriter] = pipeline.get_thought_rewriter_model()

        # Latest thought & background task
        self._latest_thought: Optional[str] = None
        self._refresh_task: Optional[asyncio.Task] = None

        # Start looping when dependencies exist
        if self.agent is not None and self.rewriter is not None:
            self._refresh_task = asyncio.create_task(self._refresh_loop())
        else:
            logger.info(
                "ThoughtManager disabled: agent=%s rewriter=%s",
                bool(self.agent),
                bool(self.rewriter),
            )

    async def _refresh_loop(self) -> None:
        try:
            while True:
                await self._try_refresh()
                await asyncio.sleep(self.REFRESH_INTERVAL_SEC)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("ThoughtManager refresh loop error: %s", e)

    async def _try_refresh(self) -> None:
        """Attempt to refresh the current thought if dependencies are ready."""
        if self.agent is None or self.rewriter is None:
            return

        # Pull plain-text conversation history
        history = None
        try:
            history = self.agent.get_chat_history()
        except Exception as e:
            logger.warning("Failed to get chat history: %s", e)
            return
        if not history:
            return

        prev = self._latest_thought or ""
        # Feed previous + current history to the rewriter
        compose = f"<thought>{prev}</thought>\n\n" f"<chat>\n{history}\n</chat>\n"

        try:
            new_thought = await self.rewriter.async_rewrite(compose)
        except Exception as e:
            logger.warning("Thought rewrite failed: %s", e)
            return

        new_thought = (new_thought or "").strip()
        if not new_thought:
            return

        # Update pipeline context and publish
        self._latest_thought = new_thought
        try:
            # Write the thought into pipeline context for downstream consumers
            ctx = self.pipeline.context
            ctx["thought"] = new_thought
            self.pipeline.context = ctx
        except Exception as e:
            logger.warning("Failed to update pipeline context with thought: %s", e)
        await self.event_bus.publish(
            ThoughtUpdated(
                session_id=self.session_id,
                text=new_thought,
                is_final=False
            )
        )

    async def shutdown(self) -> None:
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        self._refresh_task = None
        self._latest_thought = None
