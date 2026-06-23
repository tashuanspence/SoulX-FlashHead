"""
ConcurrentSessionManager
------------------------
Central registry for all active StreamingSessions.

Key responsibilities:
- Enforce the hard cap of MAX_CONCURRENT_STREAMS (default 3).
- Provide a shared asyncio.Semaphore that all sessions compete for before
  each GPU inference call, serialising GPU access across sessions.
- Expose helpers for creating, retrieving, and closing sessions.
- Surface live capacity info for the /api/status endpoint.
"""

import asyncio
import time
from typing import Dict, Optional

from loguru import logger

from config import MAX_CONCURRENT_STREAMS
from metrics import gauge, put_cloudwatch_metric


class CapacityError(Exception):
    """Raised when the server is already at maximum concurrent stream capacity."""


class ConcurrentSessionManager:
    def __init__(self, max_streams: int = MAX_CONCURRENT_STREAMS):
        self.max_streams = max_streams
        # Semaphore shared across ALL sessions — serialises GPU inference calls.
        # MUST be 1 (not max_streams) because run_pipeline_for_session()
        # restores per-session state onto the shared pipeline object before
        # calling generate().  If two sessions restore concurrently, one
        # overwrites the other's driving-image state, causing avatar
        # cross-contamination between streams.
        self.gpu_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)

        # session_id → StreamingSession
        self._sessions: Dict[str, object] = {}
        # session_id → creation timestamp
        self._created_at: Dict[str, float] = {}

        logger.info(
            f"ConcurrentSessionManager initialised: max_streams={max_streams}"
        )
        try:
            from app import log_event
            log_event("session_manager_initialized", max_streams=max_streams)
        except Exception:
            pass
        self._emit_capacity_gauges()

    def _emit_capacity_gauges(self) -> None:
        active = self.active_count()
        available = self.max_streams - active
        utilization = (active / self.max_streams * 100) if self.max_streams > 0 else 0

        # DogStatsD metrics
        gauge("soulx.sessions.active", active)
        gauge("soulx.sessions.capacity", self.max_streams)
        gauge("soulx.sessions.available", available)

        # CloudWatch metrics for ECS auto-scaling
        put_cloudwatch_metric("ActiveStreams", active)
        put_cloudwatch_metric("CapacityUtilization", utilization, unit="Percent")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def active_count(self) -> int:
        return len(self._sessions)

    def is_at_capacity(self) -> bool:
        return self.active_count() >= self.max_streams

    def create_session(
        self,
        session_id: str,
        pipeline,
        ctx,            # inference.SessionContext
        infer_params: dict,
    ):
        """
        Create and register a new StreamingSession.

        Raises CapacityError if the server is already at max_streams.
        """
        from streaming_session import StreamingSession

        if self.is_at_capacity():
            try:
                from app import log_event
                log_event("capacity_rejected", session_id=session_id, active_streams=self.active_count(), capacity=self.max_streams)
            except Exception:
                pass
            self._emit_capacity_gauges()
            raise CapacityError(
                f"Server is at capacity ({self.max_streams} concurrent streams). "
                "Please try again later."
            )

        session = StreamingSession(
            session_id=session_id,
            pipeline=pipeline,
            ctx=ctx,
            infer_params=infer_params,
            gpu_semaphore=self.gpu_semaphore,
        )
        self._sessions[session_id] = session
        self._created_at[session_id] = time.time()

        logger.info(
            f"Session {session_id} created. "
            f"Active: {self.active_count()}/{self.max_streams}"
        )
        try:
            from app import log_event
            log_event("session_created", session_id=session_id, active_streams=self.active_count(), capacity=self.max_streams)
        except Exception:
            pass
        self._emit_capacity_gauges()
        return session

    def get_session(self, session_id: str) -> Optional[object]:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        self._created_at.pop(session_id, None)
        if session is not None:
            session.stop()
            logger.info(
                f"Session {session_id} closed. "
                f"Active: {self.active_count()}/{self.max_streams}"
            )
            try:
                from app import log_event
                log_event("session_closed", session_id=session_id, active_streams=self.active_count(), capacity=self.max_streams)
            except Exception:
                pass
            self._emit_capacity_gauges()

    # ------------------------------------------------------------------
    # Status / metrics
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a dict suitable for the /api/status endpoint."""
        sessions_info = []
        now = time.time()
        for sid, session in self._sessions.items():
            stats = session.get_stats()
            sessions_info.append(
                {
                    "session_id": sid,
                    "age_seconds": round(now - self._created_at.get(sid, now), 1),
                    **stats,
                }
            )
        return {
            "active_streams": self.active_count(),
            "capacity": self.max_streams,
            "available": self.max_streams - self.active_count(),
            "at_capacity": self.is_at_capacity(),
            "sessions": sessions_info,
        }
