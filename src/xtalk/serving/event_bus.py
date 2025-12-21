# -*- coding: utf-8 -*-
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional, Set, Type, Union
import weakref

from ..log_utils import logger

from .events import BaseEvent, ErrorOccurred


@dataclass
class EventHandler:
    """Container for handler metadata."""

    handler: Callable
    priority: int = 0
    error_count: int = 0
    last_error: Optional[Exception] = None


class EventBus:
    """Event bus abstraction."""

    def __init__(self, enable_history: bool = False, max_history: int = 1000):
        """
        Initialize the event bus.

        Args:
            enable_history: whether to track event history
            max_history: max number of history entries
        """
        # Map of event type string to handlers
        # {event_type(str): [EventHandler]}
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)

        # Event history tracking
        self._enable_history = enable_history
        self._max_history = max_history
        self._event_history: List[BaseEvent] = []

        # Metrics
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "errors_occurred": 0,
            "handlers_count": 0,
        }

        # Active async tasks
        self._active_tasks: Set[asyncio.Task] = set()

        # Weak references for cleanup
        self._weak_refs: List[weakref.ref] = []

    def _get_event_key(self, event_identifier: Union[Type[BaseEvent], str]) -> str:
        """Normalize event identifier (class or string) into type string."""
        # Already a string
        if isinstance(event_identifier, str):
            return event_identifier
        # Prefer TYPE attribute on event class
        try:
            evt_type = getattr(event_identifier, "TYPE", None)
            if isinstance(evt_type, str) and evt_type:
                return evt_type
        except Exception:
            pass
        # Fall back to class name
        try:
            return event_identifier.__name__
        except Exception:
            return str(event_identifier)

    def subscribe(
        self,
        event_class: Union[Type[BaseEvent], str],
        handler: Callable[[BaseEvent], Any],
        priority: int = 0,
    ) -> None:
        """
        Subscribe to an event.

        Args:
            event_class: event class or type string (e.g., "tts.started")
            handler: callable invoked for the event
            priority: higher numbers run earlier
        """
        event_handler = EventHandler(handler=handler, priority=priority)

        key = self._get_event_key(event_class)
        self._handlers[key].append(event_handler)

        # Sort by priority descending
        self._handlers[key].sort(key=lambda h: h.priority, reverse=True)

        self._stats["handlers_count"] += 1

    def unsubscribe(
        self, event_class: Union[Type[BaseEvent], str], handler: Callable
    ) -> bool:
        """
        Unsubscribe a handler from an event.

        Args:
            event_class: event class or type string
            handler: callable to remove

        Returns:
            True if removed, False otherwise
        """
        key = self._get_event_key(event_class)
        handlers = self._handlers.get(key, [])
        for i, event_handler in enumerate(handlers):
            if event_handler.handler == handler:
                del handlers[i]
                self._stats["handlers_count"] -= 1

                return True
        return False

    async def publish(
        self, event: BaseEvent, wait_for_completion: bool = False
    ) -> bool:
        """
        Publish an event.

        Args:
            event: event object
            wait_for_completion: whether to await all handlers

        Returns:
            True on success, False otherwise
        """
        try:
            self._stats["events_published"] += 1

            # Append to history if enabled
            if self._enable_history:
                self._add_to_history(event)

            # Retrieve handlers for this event type
            event_type = event.event_type
            handlers = self._handlers.get(event_type, [])
            if not handlers:

                return True

            # spawn handler tasks
            tasks = []
            for handler in handlers:
                task = asyncio.create_task(self._handle_event_safe(handler, event))
                tasks.append(task)
                self._active_tasks.add(task)

                # cleanup after completion
                task.add_done_callback(self._active_tasks.discard)

            if wait_for_completion and tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            return True

        except Exception as e:
            logger.error("Failed to publish event: %s", e)
            self._stats["errors_occurred"] += 1

            # Publish error event
            if event.event_type != "error.occurred":
                error_event = ErrorOccurred(
                    session_id=event.session_id,
                    error_type="event_bus_publish_error",
                    error_message=str(e),
                    component="EventBus",
                )
                # Fire-and-forget to avoid recursion
                asyncio.create_task(self.publish(error_event))

            return False

    async def _handle_event_safe(self, handler: EventHandler, event: BaseEvent) -> None:
        """
        Safely handle an event, capturing exceptions.

        Args:
            handler: event handler metadata
            event: event instance
        """
        try:
            self._stats["events_processed"] += 1

            # Invoke handler
            if asyncio.iscoroutinefunction(handler.handler):
                await handler.handler(event)
            else:
                handler.handler(event)

        except Exception as e:
            handler.error_count += 1
            handler.last_error = e
            self._stats["errors_occurred"] += 1

            logger.error("Event handler raised: %s, event_type: %s", e, event.event_type)

            # Publish error event
            if event.event_type != "error.occurred":
                error_event = ErrorOccurred(
                    session_id=event.session_id,
                    error_type="event_handler_error",
                    error_message=str(e),
                    component=f"Handler:{handler.handler.__name__}",
                )
                # Schedule async publication to avoid blocking
                error_task = asyncio.create_task(self.publish(error_event))
                self._active_tasks.add(error_task)
                error_task.add_done_callback(self._active_tasks.discard)

    def _add_to_history(self, event: BaseEvent) -> None:
        """
        Append event to history storage.

        Args:
            event: event instance
        """
        self._event_history.append(event)

        # Trim history size
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

    def get_history(
        self, event_type: Optional[str] = None, session_id: Optional[str] = None
    ) -> List[BaseEvent]:
        """
        Retrieve event history with optional filters.

        Args:
            event_type: filter by type
            session_id: filter by session id

        Returns:
            List of events
        """
        if not self._enable_history:
            return []

        history = self._event_history

        if event_type:
            history = [e for e in history if e.event_type == event_type]

        if session_id:
            history = [e for e in history if e.session_id == session_id]

        return history

    def get_stats(self) -> Dict[str, Any]:
        """
        Return current event bus statistics.

        Returns:
            Dict of stats
        """
        handler_stats = {}
        for event_type, handlers in self._handlers.items():
            handler_stats[event_type] = len(handlers)

        return {
            **self._stats,
            "handler_stats": handler_stats,
            "active_tasks": len(self._active_tasks),
            "history_enabled": self._enable_history,
            "history_count": len(self._event_history) if self._enable_history else 0,
        }

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    async def _wait_for_all_handlers(self, timeout: float = 3.0):
        """
        Wait for all active handler tasks to finish.

        Args:
            timeout: timeout in seconds

        Returns:
            None
        """
        if not self._active_tasks:
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._active_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for handlers; cancelling %d pending tasks",
                len(self._active_tasks),
            )
            for task in self._active_tasks:
                if not task.done():
                    task.cancel()

    async def shutdown(self) -> None:
        """Shut down the event bus and release resources."""

        # Wait for active tasks to finish
        await self._wait_for_all_handlers(timeout=3.0)

        # Cleanup structures
        self._handlers.clear()
        self._event_history.clear()
        self._active_tasks.clear()

    def __del__(self):
        """Destructor: best-effort cleanup."""
        # Release resources
        if hasattr(self, "_active_tasks"):
            for task in self._active_tasks:
                if not task.done():
                    task.cancel()
