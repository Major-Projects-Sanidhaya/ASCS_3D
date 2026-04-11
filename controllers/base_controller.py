"""
Abstract base for all four tier controllers
(GeneralController, NodeController, ScoutController, WorkerController).
Concrete methods are fully implemented shared utilities.
Abstract methods define the interface each tier controller must satisfy.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import time


class BaseController(ABC):
    """
    Abstract base for all four tier controllers. Provides fully implemented
    shared utilities (logging, uptime tracking, silence detection, packet
    validation, vector helpers) and declares the abstract interface that
    every concrete tier controller must satisfy.
    """

    SILENCE_WARN    = 2.0
    SILENCE_DEAD    = 10.0
    MAX_LOG_SIZE    = 500
    STATUS_HEALTHY  = 'HEALTHY'
    STATUS_DEGRADED = 'DEGRADED'
    STATUS_SILENT   = 'SILENT'

    # ── Initialisation ──

    def __init__(self, tier_name: str, agent_id: str) -> None:
        self.tier_name: str                       = tier_name
        self.agent_id: str                        = agent_id
        self._event_log: List[dict]               = []
        self._uptime: float                       = 0.0
        self._step_count: int                     = 0
        self._silence_timers: Dict[str, float]    = {}
        self._last_downlink_time: float           = time.time()
        self._last_uplink_time: float             = time.time()
        self._downlink_packet: Optional[dict]     = None
        self._uplink_packet: Optional[dict]       = None
        self._status: str                         = self.STATUS_HEALTHY

    # ── Abstract Interface ──

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance the controller by one simulation tick of length dt seconds."""

    @abstractmethod
    def receive_downlink(self, packet: dict) -> None:
        """Accept an incoming packet from the tier above."""

    @abstractmethod
    def send_uplink(self, packet: dict) -> None:
        """Dispatch a packet to the tier above."""

    @abstractmethod
    def receive_uplink(self, packet: dict) -> None:
        """Accept an incoming packet from the tier below."""

    @abstractmethod
    def send_downlink(self, packet: dict) -> None:
        """Dispatch a packet to the tier below."""

    @abstractmethod
    def get_status(self) -> dict:
        """Return a tier-specific status snapshot."""

    @abstractmethod
    def handle_silence(self, source: str, dt: float) -> None:
        """React to a communication gap from the named source."""

    @abstractmethod
    def reset(self) -> None:
        """Return the controller to its initial state."""

    # ── Logging ──

    def log_event(self, event_type: str, payload: dict) -> None:
        """
        Append a structured event entry to the internal event log.

        Each entry carries simulation uptime, wall-clock time, tier name,
        agent ID, event type, and an arbitrary payload dict. When the log
        exceeds MAX_LOG_SIZE the oldest entry is evicted.
        """
        entry = {
            'timestamp':  self._uptime,
            'wall_time':  time.time(),
            'tier':       self.tier_name,
            'agent_id':   self.agent_id,
            'event_type': event_type,
            'payload':    payload,
        }
        self._event_log.append(entry)
        if len(self._event_log) > self.MAX_LOG_SIZE:
            self._event_log.pop(0)

    def get_event_log(self, last_n: Optional[int] = None) -> List[dict]:
        """
        Return a copy of the event log, optionally limited to the last n entries.
        """
        if last_n is None:
            return list(self._event_log)
        return self._event_log[-last_n:]

    def clear_event_log(self) -> None:
        """
        Discard all log entries and record a LOG_CLEARED sentinel event.
        """
        self._event_log = []
        self.log_event('LOG_CLEARED', {'step': self._step_count})

    # ── Uptime & Step Counter ──

    def tick_uptime(self, dt: float) -> None:
        """
        Advance the simulation uptime and increment the step counter.
        Call once per simulation tick before any other per-tick logic.
        """
        self._uptime     += dt
        self._step_count += 1

    def get_uptime(self) -> float:
        """Return total simulated uptime in seconds."""
        return self._uptime

    def get_step_count(self) -> int:
        """Return the total number of simulation ticks processed."""
        return self._step_count

    # ── Silence Detection ──

    def update_silence_timer(
        self, source: str, dt: float, received: bool
    ) -> str:
        """
        Advance or reset the silence timer for a named communication source.

        If a packet was received this tick the timer resets to zero and
        STATUS_HEALTHY is returned. Otherwise the timer accumulates and the
        appropriate status string is returned:
          >= SILENCE_DEAD  → STATUS_SILENT   (event logged)
          >= SILENCE_WARN  → STATUS_DEGRADED (event logged)
          otherwise        → STATUS_HEALTHY
        """
        if received:
            self._silence_timers[source] = 0.0
            return self.STATUS_HEALTHY

        self._silence_timers[source] = (
            self._silence_timers.get(source, 0.0) + dt
        )
        t = self._silence_timers[source]

        if t >= self.SILENCE_DEAD:
            self.log_event('SILENCE_DEAD', {'source': source, 't': t})
            return self.STATUS_SILENT
        elif t >= self.SILENCE_WARN:
            self.log_event('SILENCE_WARN', {'source': source, 't': t})
            return self.STATUS_DEGRADED
        return self.STATUS_HEALTHY

    def get_silence_status(self, source: str) -> str:
        """
        Return the current silence status for a named source without
        modifying any timers.
        """
        t = self._silence_timers.get(source, 0.0)
        if t >= self.SILENCE_DEAD:
            return self.STATUS_SILENT
        elif t >= self.SILENCE_WARN:
            return self.STATUS_DEGRADED
        return self.STATUS_HEALTHY

    # ── Packet Utilities ──

    def get_last_downlink(self) -> Optional[dict]:
        """Return the most recently stored downlink packet, or None."""
        return self._downlink_packet

    def get_last_uplink(self) -> Optional[dict]:
        """Return the most recently stored uplink packet, or None."""
        return self._uplink_packet

    def is_packet_expired(self, packet: dict, current_time: float) -> bool:
        """
        Return True if the packet's age exceeds its declared TTL.

        Returns False if either 'ttl' or 'timestamp' is absent — missing
        fields are treated as non-expiring rather than immediately expired.
        """
        if 'ttl' not in packet or 'timestamp' not in packet:
            return False
        return (current_time - packet['timestamp']) > packet['ttl']

    def validate_packet(
        self, packet: dict, required_keys: List[str]
    ) -> Tuple[bool, str]:
        """
        Check that packet is a dict containing all required keys.

        Returns (True, '') on success or (False, reason_string) on failure.
        """
        if not isinstance(packet, dict):
            return False, f'Expected dict, got {type(packet).__name__}'
        for key in required_keys:
            if key not in packet:
                return False, f'Missing key: {key}'
        return True, ''

    # ── Status Helper ──

    def build_base_status(self) -> dict:
        """
        Return a status dict populated with fields common to all tiers.

        Tier-specific controllers should call this and merge their own fields
        on top rather than duplicating the common fields.
        """
        return {
            'tier':            self.tier_name,
            'agent_id':        self.agent_id,
            'uptime':          self._uptime,
            'step_count':      self._step_count,
            'status':          self._status,
            'silence_timers':  dict(self._silence_timers),
            'log_length':      len(self._event_log),
        }

    # ── Math Utilities ──

    def clamp(self, value: float, low: float, high: float) -> float:
        """Return value clamped to [low, high]."""
        return float(np.clip(value, low, high))

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        """
        Return the unit vector of vec, or a zero vector if the norm is
        below 1e-8 (avoids division by zero for stationary agents).
        """
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return np.zeros_like(vec)
        return vec / norm

    # ── Repr ──

    def __repr__(self) -> str:
        return (
            f'<{self.tier_name} id={self.agent_id} '
            f'uptime={self._uptime:.1f}s>'
        )
