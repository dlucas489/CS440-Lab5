from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence, TypeVar

from aima.agents import Agent, Environment

T = TypeVar("T")
Location = tuple[int, int]

START_TOKEN = "start"
MIN_DIRT, MAX_DIRT = 0, 10
NO_OP, STOP = "NoOp", "Stop"
MOVES: dict[str, Location] = {
    "Up": (0, -1),
    "Down": (0, 1),
    "Left": (-1, 0),
    "Right": (1, 0),
}

HILL_CLIMBING_LAYOUT = [
    [0, 3, 2, "start"],
    [0, 0, 0, 0],
    [0, 4, 6, 8],
    [7, 9, 0, 10],
]

ANNEALING_LAYOUT = [
    [0, 3, 2, "start"],
    [3, 6, 0, 0],
    [5, 0, 6, 8],
    [4, 9, 0, 10],
]

MDP_LAYOUT = [
    [0, 0, 0, "start"],
    [0, "X", 0, 0],
    [0, 0, 0, 0],
    [0, -1, 0, 10],
]

@dataclass(frozen=True)
class DirtPercept:
    location: Location
    dirt: int
    readings: dict[str, int]


@dataclass(frozen=True)
class PathEntry:
    step: int
    action: str
    location: Location
    dirt: int


@dataclass(frozen=True)
class AnnealingEvent:
    step: int
    temperature: float
    location: Location
    dirt: int
    action: str
    delta: int
    accepted: bool


class DirtGridEnvironment(Environment):

    def __init__(self, layout: Sequence[Sequence[int | str]]):
        super().__init__()
        self.dirt, self.start = _parse_layout(layout)
        self.path_log: list[PathEntry] = []

    def thing_classes(self):
        return [Agent]

    def default_location(self, thing) -> Location:
        return self.start

    def add_thing(self, thing, location=None):
        super().add_thing(thing, location)
        if isinstance(thing, Agent):
            self.path_log.append(PathEntry(0, "start", thing.location, self.dirt_at(thing.location)))

    def percept(self, agent) -> DirtPercept:
        location = agent.location
        return DirtPercept(location, self.dirt_at(location), self.sense_neighbors(location))

    def sense_neighbors(self, location: Location) -> dict[str, int]:
        return {
            action: self.dirt_at(target)
            for action in MOVES
            if self._in_bounds(target := _shift(location, action))
        }

    def execute_action(self, agent, action: str) -> None:
        if action == STOP:
            agent.alive = False
        elif action in MOVES:
            self._move(agent, action)

    def dirt_at(self, location: Location) -> int:
        x, y = location
        return self.dirt[y][x]

    def dirtiest(self) -> Location:
        cells = ((x, y) for y, row in enumerate(self.dirt) for x in range(len(row)))
        return max(cells, key=self.dirt_at)

    def _move(self, agent, action: str) -> None:
        target = _shift(agent.location, action)
        agent.bump = not self._in_bounds(target)
        if agent.bump:
            return
        agent.location = target
        dirt = self.dirt_at(target)
        agent.performance = max(agent.performance, dirt)
        self.path_log.append(PathEntry(len(self.path_log), action, target, dirt))

    def _in_bounds(self, location: Location) -> bool:
        x, y = location
        return 0 <= y < len(self.dirt) and 0 <= x < len(self.dirt[y])

def HillClimbingVacuumProgram():

    def program(percept: DirtPercept) -> str:
        action, dirt = max(percept.readings.items(), key=lambda reading: reading[1])
        return action if dirt > percept.dirt else STOP

    return program


class RandomSource(Protocol):
    def random(self) -> float: ...
    def choice(self, seq: Sequence[T]) -> T: ...


@dataclass(frozen=True)
class AnnealingSchedule:
    initial_temperature: float = 10.0
    cooling_rate: float = 0.95
    minimum_temperature: float = 0.01

    def temperatures(self) -> Iterator[float]:
        temperature = self.initial_temperature
        while temperature > self.minimum_temperature:
            yield temperature
            temperature *= self.cooling_rate


class SimulatedAnnealingVacuumProgram:

    def __init__(self, schedule: AnnealingSchedule = AnnealingSchedule(),
                 rng: RandomSource | None = None):
        self._temperatures = schedule.temperatures()
        self._rng = rng or random.Random()
        self.log: list[AnnealingEvent] = []

    def __call__(self, percept: DirtPercept) -> str:
        temperature = next(self._temperatures, None)
        if temperature is None:
            return STOP
        action = self._rng.choice(list(percept.readings))
        delta = percept.readings[action] - percept.dirt
        accepted = self._should_accept(delta, temperature)
        self.log.append(AnnealingEvent(
            len(self.log) + 1, temperature, percept.location,
            percept.dirt, action, delta, accepted,
        ))
        return action if accepted else NO_OP

    def _should_accept(self, delta: int, temperature: float) -> bool:
        return delta > 0 or self._rng.random() < math.exp(delta / temperature)

def format_path(path_log: Sequence[PathEntry]) -> str:
    return "\n".join(
        f"Step {e.step}: {e.action:<5} -> {e.location} dirt={e.dirt}" for e in path_log
    )


def format_annealing_log(log: Sequence[AnnealingEvent]) -> str:
    return "\n".join(
        f"Step {e.step:03d} | T={e.temperature:7.3f} | at {e.location}[{e.dirt}] "
        f"try {e.action:<5} | delta={e.delta:+d} | {'ACCEPT' if e.accepted else 'REJECT'}"
        for e in log
    )


def format_summary(env: DirtGridEnvironment, agent) -> str:
    peak = env.dirtiest()
    return (f"Dirtiest spot: dirt={agent.performance}. "
            f"max: {peak} dirt={env.dirt_at(peak)}.")

def _shift(location: Location, action: str) -> Location:
    dx, dy = MOVES[action]
    return location[0] + dx, location[1] + dy

def _parse_layout(layout):
    start, dirt = None, []
    for y, row in enumerate(layout):
        parsed = []
        for x, token in enumerate(row):
            if token == START_TOKEN:
                start = (x, y)
                parsed.append(MIN_DIRT)
            elif isinstance(token, int) and MIN_DIRT <= token <= MAX_DIRT:
                parsed.append(token)
            else:
                raise ValueError(f"Invalid cell. {(x, y)}: {token!r}")
        dirt.append(tuple(parsed))
    if start is None:
        raise ValueError(f"Layout must contain the '{START_TOKEN}' cell")
    return tuple(dirt), start