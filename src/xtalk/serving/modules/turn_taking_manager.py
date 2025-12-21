# -*- coding: utf-8 -*-
"""
TurnTakingManager

Mediator that subscribes to core events (VAD/ASR/TTS/etc.) and dispatches turn
control events to keep modules decoupled while preserving the public event API.
"""

from typing import Any

from ..event_bus import EventBus
from ..interfaces import Manager
from ..events import (
    # Original events
    ASRResultFinal,
    VADSpeechStart,
    VADSpeechEnd,
    VerificationResult,
    TTSStarted,
    TTSPlaybackFinished,
    # Mediator turn-control events
    TurnLLMAgentResumeRequested,
    TurnLLMAgentStopRequested,
    TurnLLMAgentPauseRequested,
    TurnLLMAgentStartRequested,
    TurnASRResetRequested,
    TurnASRStartRequested,
    TurnASREndRequested,
    TurnASRFlushRequested,
)
from ..events import ASRStableSegmentReady


class TurnTakingManager(Manager):
    """Mediator coordinating ASR/TTS/LLM turn transitions."""

    def __init__(
        self, event_bus: EventBus, session_id: str, config: dict[str, Any] | None = None
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        # Per-session config
        self.config: dict[str, Any] = config or {}

        # Simultaneous-generation mode
        self._sim_gen: bool = bool(self.config.get("sim_gen", False))
        # Track whether LLM/TTS started for sim-gen
        self._sim_tts_started: bool = False
        self._sim_partial_text: str = ""

    @Manager.event_handler(
        VADSpeechStart,
        priority=95,
        enabled_if=lambda mgr: not mgr._sim_gen,
    )
    async def _on_vad_start(self, _event: VADSpeechStart) -> None:
        """VAD start: pause LLM and notify ASR to start."""
        await self.event_bus.publish(
            TurnLLMAgentPauseRequested(
                session_id=self.session_id,
            ),
            wait_for_completion=True,
        )
        await self.event_bus.publish(TurnASRStartRequested(session_id=self.session_id))

    @Manager.event_handler(VADSpeechEnd, priority=95)
    async def _on_vad_end(self, _event: VADSpeechEnd) -> None:
        """VAD end: either flush ASR segments (sim-gen) or finalize the turn."""
        if getattr(self, "_sim_gen", False):

            await self.event_bus.publish(
                TurnASRFlushRequested(session_id=self.session_id, reason="vad_end")
            )
        else:
            await self.event_bus.publish(
                TurnASREndRequested(session_id=self.session_id)
            )

    @Manager.event_handler(
        VADSpeechStart,
        priority=95,
        enabled_if=lambda mgr: mgr._sim_gen,
    )
    async def _on_vad_start_sim(self, _event: VADSpeechStart) -> None:
        """Sim-gen: VAD start only triggers ASR (TTS keeps running)."""

        self._sim_tts_started = False
        self._sim_partial_text = ""
        await self.event_bus.publish(TurnASRStartRequested(session_id=self.session_id))

    @Manager.event_handler(
        ASRStableSegmentReady,
        priority=92,
        enabled_if=lambda mgr: mgr._sim_gen,
    )
    async def _on_stable_segment(self, event: ASRStableSegmentReady) -> None:
        """Sim-gen: feed stable ASR segment into TTS/LLM."""
        seg = getattr(event, "text", "") or ""
        if not seg:
            return
        self._sim_partial_text += seg
        if not self._sim_tts_started:
            self._sim_tts_started = True

            await self.event_bus.publish(
                TurnLLMAgentStartRequested(
                    session_id=self.session_id,
                    text=self._sim_partial_text,
                )
            )
        else:

            await self.event_bus.publish(
                TurnLLMAgentResumeRequested(
                    session_id=self.session_id,
                    text=self._sim_partial_text,
                )
            )

    @Manager.event_handler(ASRResultFinal, priority=98)
    async def _on_asr_final(self, event: ASRResultFinal) -> None:
        """ASR final result triggers LLM generation."""
        text = getattr(event, "text", "") or ""
        if not text:
            return
        await self.event_bus.publish(
            TurnLLMAgentStartRequested(
                session_id=self.session_id,
                text=text,
            )
        )

    @Manager.event_handler(
        VerificationResult,
        priority=90,
        enabled_if=lambda mgr: not mgr._sim_gen,
    )
    async def _on_verification_result(self, event: VerificationResult) -> None:
        """Stop LLM on valid verification; resume if invalid."""
        if event.is_valid:
            await self.event_bus.publish(
                TurnLLMAgentStopRequested(
                    session_id=self.session_id,
                    reason="verification_valid",
                ),
                wait_for_completion=True,
            )
        else:
            await self.event_bus.publish(
                TurnLLMAgentResumeRequested(
                    session_id=self.session_id,
                ),
                wait_for_completion=True,
            )

    @Manager.event_handler(
        TTSStarted,
        priority=85,
        enabled_if=lambda mgr: not mgr._sim_gen,
    )
    async def _on_llm_agent_response_update(self, _event: TTSStarted) -> None:
        """When TTS starts outputting, request ASR reset."""
        await self.event_bus.publish(TurnASRResetRequested(session_id=self.session_id))

    @Manager.event_handler(
        TTSPlaybackFinished,
        priority=85,
        enabled_if=lambda mgr: not mgr._sim_gen,
    )
    async def _on_tts_playback_finished(self, _event: TTSPlaybackFinished) -> None:
        """Frontend playback finished – stop LLM agent to clean up."""
        await self.event_bus.publish(
            TurnLLMAgentStopRequested(
                session_id=self.session_id,
                reason="playback_finished",
            )
        )

    async def shutdown(self):
        pass
