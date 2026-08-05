"""Deterministic pycma adapter for the evolutionary-search package."""

from __future__ import annotations

import base64
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass
from typing import Sequence

# Matplotlib is an optional pycma dependency and inspects WINDIR during import.
# The restricted Windows runner omits it although the standard font location is stable.
os.environ.setdefault("WINDIR", r"C:\\Windows")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "esn-matplotlib"))
import cma
import numpy as np


@dataclass(frozen=True, slots=True)
class CMAESConfig:
    dimension: int
    population_size: int = 16
    initial_sigma: float = 0.35
    seed: int = 1

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.population_size < 2 or self.initial_sigma <= 0:
            raise ValueError("invalid CMA-ES configuration")


class CMAES:
    """Maximisation-oriented ``ask``/``tell`` façade backed by pycma."""

    def __init__(self, config: CMAESConfig, mean: Sequence[float] | None = None) -> None:
        self.config = config
        center = list(mean) if mean is not None else [0.0] * config.dimension
        self._strategy = cma.CMAEvolutionStrategy(center, config.initial_sigma, {"popsize": config.population_size, "seed": config.seed, "verbose": -9})
        self._rng_state = np.random.get_state()

    @property
    def generation(self) -> int:
        return int(self._strategy.countiter)

    @property
    def sigma(self) -> float:
        return float(self._strategy.sigma)

    @property
    def mean(self) -> tuple[float, ...]:
        """Current distribution centre, for task-level parent-centred births."""
        return tuple(float(value) for value in self._strategy.mean)

    def ask(self) -> tuple[tuple[float, ...], ...]:
        caller_state = np.random.get_state()
        np.random.set_state(self._rng_state)
        try:
            candidates = self._strategy.ask()
            self._rng_state = np.random.get_state()
        finally:
            np.random.set_state(caller_state)
        return tuple(tuple(float(value) for value in candidate) for candidate in candidates)

    def tell(self, population: Sequence[Sequence[float]], fitnesses: Sequence[float]) -> None:
        if len(population) != self.config.population_size or len(fitnesses) != self.config.population_size:
            raise ValueError("tell requires exactly one full CMA-ES population")
        # pycma minimises; Phase 1A fitness is maximised.
        caller_state = np.random.get_state(); np.random.set_state(self._rng_state)
        try:
            self._strategy.tell([list(candidate) for candidate in population], [-float(value) for value in fitnesses])
            self._rng_state = np.random.get_state()
        finally:
            np.random.set_state(caller_state)

    def state_dict(self) -> dict[str, object]:
        payload = base64.b64encode(pickle.dumps(self._strategy, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
        return {"config": asdict(self.config), "pycma_pickle": payload, "rng_pickle": base64.b64encode(pickle.dumps(self._rng_state)).decode("ascii")}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "CMAES":
        instance = cls(CMAESConfig(**state["config"]))
        instance._strategy = pickle.loads(base64.b64decode(str(state["pycma_pickle"])))
        instance._rng_state = pickle.loads(base64.b64decode(str(state["rng_pickle"])))
        return instance
