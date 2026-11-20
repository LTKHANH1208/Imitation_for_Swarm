"""Timestamp utilities for observation-action pairing.

The recorder uses the latest sensor sample whose timestamp is <= the action
stamp. This avoids attaching a future observation to a past action.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Optional, TypeVar

T = TypeVar("T")


def stamp_to_ns(stamp) -> int:
    """Convert a builtin_interfaces/Time-like object to integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def message_stamp_ns(msg) -> int:
    header = getattr(msg, "header", None)
    if header is None or not hasattr(header, "stamp"):
        return 0
    return stamp_to_ns(header.stamp)


@dataclass(frozen=True)
class TimedSample(Generic[T]):
    stamp_ns: int
    message: T


@dataclass(frozen=True)
class SelectedSample(Generic[T]):
    message: T
    stamp_ns: int
    age_sec: float


class StampedCache(Generic[T]):
    """Small timestamp-ordered cache; bounded memory and deterministic lookup."""

    def __init__(self, maxlen: int = 30):
        self._items: Deque[TimedSample[T]] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def append(self, msg: T, stamp_ns: Optional[int] = None) -> None:
        if stamp_ns is None:
            stamp_ns = message_stamp_ns(msg)
        stamp_ns = int(stamp_ns)
        if stamp_ns <= 0:
            return
        # Sensor callbacks should be monotonic, but simulation reset or delayed
        # transport can produce an older stamp. Keep the cache sorted and bounded.
        if not self._items or stamp_ns >= self._items[-1].stamp_ns:
            self._items.append(TimedSample(stamp_ns, msg))
            return
        items = list(self._items)
        items.append(TimedSample(stamp_ns, msg))
        items.sort(key=lambda x: x.stamp_ns)
        self._items = deque(items[-self._items.maxlen :], maxlen=self._items.maxlen)

    def latest_at_or_before(
        self, target_ns: int, max_age_sec: float
    ) -> Optional[SelectedSample[T]]:
        if target_ns <= 0:
            return None
        max_age_ns = int(max_age_sec * 1_000_000_000)
        for item in reversed(self._items):
            if item.stamp_ns <= target_ns:
                age_ns = target_ns - item.stamp_ns
                if age_ns <= max_age_ns:
                    return SelectedSample(
                        message=item.message,
                        stamp_ns=item.stamp_ns,
                        age_sec=age_ns / 1_000_000_000.0,
                    )
                return None
        return None

    def latest(self) -> Optional[TimedSample[T]]:
        return self._items[-1] if self._items else None
