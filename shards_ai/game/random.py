import random
from hashlib import sha256
from typing import TypeVar


T = TypeVar("T")


class GameRandom:
    """Small injectable wrapper around Python's local pseudo-random generator."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._root_seed = seed if seed is not None else random.SystemRandom().getrandbits(128)
        self._random = random.Random(seed)

    def shuffle(self, values: list[object]) -> None:
        self._random.shuffle(values)

    def choice(self, values: list[T]) -> T:
        return self._random.choice(values)

    def random(self) -> float:
        return self._random.random()

    def derive(self, label: str) -> "GameRandom":
        """Return a deterministic independent stream derived from this stream's seed."""
        material = f"{self._root_seed}:{label}".encode("utf-8")
        derived_seed = int.from_bytes(sha256(material).digest(), "big")
        return GameRandom(derived_seed)
