"""Receiver-relative temporal frontiers for asynchronous systems.

This module keeps one objective `now_tick` while exposing source->receiver path
delays explicitly.

The key distinction is:

    world age = now - event.world_tick              (global/objective)
    path frontier = now - path_delay[source,recv]   (receiver-relative)

For a fixed-delay path, `path_frontier` is the latest source-world tick whose
events would have had enough time to reach that receiver by `now`.

This is ordinary latency bookkeeping.  It is useful in WidePresent because a
single global temporal projection can otherwise hide that different receivers
have access to different slices of the same event stream.

No claim of novelty is made; this is a small runtime primitive for demos and
agent experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


PathKey = tuple[str, str]


@dataclass(frozen=True)
class TransitItem:
    """One source event travelling toward one receiver."""

    event_id: int
    source: str
    receiver: str
    value: Any
    world_tick: int
    launch_tick: int
    arrival_tick: int

    def world_age(self, now_tick: int) -> int:
        return int(now_tick) - self.world_tick

    def arrival_age(self, now_tick: int) -> Optional[int]:
        """Ticks since receiver arrival, or None while still in flight."""
        if int(now_tick) < self.arrival_tick:
            return None
        return int(now_tick) - self.arrival_tick

    def path_progress(self, now_tick: int) -> float:
        """0..1 progress through the fixed-delay path."""
        total = self.arrival_tick - self.world_tick
        if total <= 0:
            return 1.0
        elapsed = int(now_tick) - self.world_tick
        return min(1.0, max(0.0, elapsed / total))


@dataclass(frozen=True)
class ReceiverSnapshot:
    receiver: str
    now_tick: int
    arrived: tuple[TransitItem, ...]
    in_flight: tuple[TransitItem, ...]
    path_frontiers: Mapping[str, int]

    def latest_arrived(self, source: str) -> Optional[TransitItem]:
        matches = [item for item in self.arrived if item.source == str(source)]
        if not matches:
            return None
        return max(matches, key=lambda item: (item.world_tick, item.event_id))

    def render(self) -> str:
        lines = [
            f"RECEIVER PRESENT receiver={self.receiver} now={self.now_tick}",
            "- source path frontiers (latest source tick that had time to arrive):",
        ]
        if self.path_frontiers:
            for source, tick in sorted(self.path_frontiers.items()):
                lines.append(f"  - {source}: {tick}")
        else:
            lines.append("  - none")

        lines.append("- arrived:")
        if self.arrived:
            for item in sorted(self.arrived, key=lambda x: (x.arrival_tick, x.event_id)):
                lines.append(
                    f"  - event={item.event_id} source={item.source} "
                    f"world={item.world_tick} arrival={item.arrival_tick} "
                    f"world_age={item.world_age(self.now_tick)} "
                    f"arrival_age={item.arrival_age(self.now_tick)} value={item.value!r}"
                )
        else:
            lines.append("  - none")

        lines.append("- in flight:")
        if self.in_flight:
            for item in sorted(self.in_flight, key=lambda x: (x.arrival_tick, x.event_id)):
                lines.append(
                    f"  - event={item.event_id} source={item.source} "
                    f"world={item.world_tick} eta={item.arrival_tick - self.now_tick} "
                    f"path_progress={item.path_progress(self.now_tick):.3f} "
                    f"value={item.value!r}"
                )
        else:
            lines.append("  - none")

        return "\n".join(lines)


class ReceiverPresent:
    """One objective clock plus fixed source->receiver transport delays.

    `path_delays[(source, receiver)] = delay_ticks`

    An emitted source event is expanded into one `TransitItem` per destination.
    The values are not transformed; this class only models availability.
    """

    def __init__(self, path_delays: Mapping[PathKey, int]):
        if not path_delays:
            raise ValueError("path_delays must not be empty")

        normalized: dict[PathKey, int] = {}
        for (source, receiver), delay in path_delays.items():
            key = (str(source), str(receiver))
            delay = int(delay)
            if delay < 0:
                raise ValueError("path delays must be non-negative")
            normalized[key] = delay

        self.path_delays = normalized
        self.now_tick = 0
        self.items: list[TransitItem] = []
        self._next_event_id = 0

    @property
    def receivers(self) -> tuple[str, ...]:
        return tuple(sorted({receiver for _, receiver in self.path_delays}))

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({source for source, _ in self.path_delays}))

    def advance(self, ticks: int = 1) -> None:
        ticks = int(ticks)
        if ticks < 0:
            raise ValueError("ticks must be non-negative")
        self.now_tick += ticks

    def path_delay(self, source: str, receiver: str) -> int:
        key = (str(source), str(receiver))
        if key not in self.path_delays:
            raise KeyError(f"no path delay registered for {key}")
        return self.path_delays[key]

    def path_frontier(self, source: str, receiver: str) -> int:
        """Latest source tick that had enough transit time to reach receiver."""
        return self.now_tick - self.path_delay(source, receiver)

    def emit(
        self,
        source: str,
        value: Any,
        *,
        world_tick: Optional[int] = None,
        receivers: Optional[Iterable[str]] = None,
    ) -> int:
        """Emit one source event toward selected receivers.

        `world_tick` may be older than now (late dispatch / replay), but not future.
        Arrival is defined from world time plus the registered fixed path delay.
        """
        source = str(source)
        world_tick = self.now_tick if world_tick is None else int(world_tick)
        if world_tick > self.now_tick:
            raise ValueError("cannot emit an event from future world time")

        if receivers is None:
            destinations = [
                receiver
                for (registered_source, receiver) in self.path_delays
                if registered_source == source
            ]
        else:
            destinations = [str(receiver) for receiver in receivers]

        if not destinations:
            raise ValueError(f"no receiver paths registered for source {source!r}")

        event_id = self._next_event_id
        self._next_event_id += 1

        for receiver in destinations:
            delay = self.path_delay(source, receiver)
            self.items.append(
                TransitItem(
                    event_id=event_id,
                    source=source,
                    receiver=receiver,
                    value=value,
                    world_tick=world_tick,
                    launch_tick=self.now_tick,
                    arrival_tick=world_tick + delay,
                )
            )

        return event_id

    def snapshot(self, receiver: str) -> ReceiverSnapshot:
        receiver = str(receiver)
        if receiver not in self.receivers:
            raise KeyError(f"unknown receiver {receiver!r}")

        arrived = tuple(
            item
            for item in self.items
            if item.receiver == receiver and item.arrival_tick <= self.now_tick
        )
        in_flight = tuple(
            item
            for item in self.items
            if item.receiver == receiver
            and item.world_tick <= self.now_tick < item.arrival_tick
        )
        frontiers = {
            source: self.path_frontier(source, receiver)
            for source in self.sources
            if (source, receiver) in self.path_delays
        }
        return ReceiverSnapshot(
            receiver=receiver,
            now_tick=self.now_tick,
            arrived=arrived,
            in_flight=in_flight,
            path_frontiers=frontiers,
        )

    def frontier_width(self, source: str) -> int:
        """Spread of source-time frontiers across all connected receivers."""
        values = [
            self.path_frontier(source, receiver)
            for receiver in self.receivers
            if (str(source), receiver) in self.path_delays
        ]
        if not values:
            raise KeyError(f"source {source!r} has no registered paths")
        return max(values) - min(values)
