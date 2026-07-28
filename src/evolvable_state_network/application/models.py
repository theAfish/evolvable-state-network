"""Validated request models at the application boundary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base request model that rejects misspelled or unsupported fields."""

    model_config = ConfigDict(extra="forbid")


class ExperimentPayload(StrictModel):
    seed: int = Field(7, ge=0, lt=2**32)
    nodes: int = Field(24, ge=2, le=200)
    mean_degree: float = Field(5.0, ge=0)
    steps: int = Field(300, ge=1, le=5000)
    batch_size: int = Field(4, ge=1, le=64)
    dt: float = Field(.05, gt=0, le=1)
    baseline: Literal["both", "fixed_rnn", "homeostatic"] = "both"

    @model_validator(mode="after")
    def validate_degree(self) -> "ExperimentPayload":
        if self.mean_degree > self.nodes - 1:
            raise ValueError("mean_degree cannot exceed nodes - 1")
        return self


class EvolutionPayload(StrictModel):
    seed: int | None = Field(None, ge=0, lt=2**32)
    samples: int = Field(16, ge=4, le=64)
    generations: int = Field(4, ge=1, le=20)
    population: int = Field(8, ge=2, le=32)


class AsyncDiagnosticPayload(StrictModel):
    seed: int | None = Field(None, ge=0, lt=2**32)


class AsyncTrainingPayload(StrictModel):
    seed: int | None = Field(None, ge=0, lt=2**32)
    candidate_budget: int = Field(200, ge=1, le=5000, description="Evidence checkpoint; does not stop stage evolution")
    max_ticks: int | None = Field(
        None,
        ge=20,
        le=100_000,
        description="Optional explicit interruption cap; normal training ends only with a stable final-stage population.",
    )
    slots: int = Field(8, ge=1, le=32)
    replicas: int = Field(3, ge=1, le=8)
    stable_population_size: int = Field(3, ge=1, le=32)
    optimizer_batch: int = Field(8, ge=2, le=32)
    state_width: int = Field(2, ge=1, le=8)
    stage_1_lifetime: int = Field(40, ge=4, le=2000)
    stage_2_lifetime: int = Field(100, ge=8, le=5000)
    stage_1_nodes: int = Field(8, ge=3, le=100)
    stage_2_nodes: int = Field(12, ge=3, le=200)
    mean_degree: float = Field(3.0, ge=0.5, le=20)
    input_scale: float = Field(.12, gt=0, le=2)
    disturbance_interval: int = Field(10, ge=2, le=1000)
    disturbance_strength: float = Field(.12, ge=0, le=2)
    fatal_threshold: float = Field(8.0, gt=0, le=100)
    node_growth_alert: float = Field(3.2, gt=0, lt=4)
    one_direction_steps: int = Field(12, ge=2, le=1000)
    probe_interval: int = Field(8, ge=2, le=1000)

    @model_validator(mode="after")
    def validate_training_shape(self) -> "AsyncTrainingPayload":
        if self.stage_2_lifetime <= self.stage_1_lifetime:
            raise ValueError("stage 2 lifetime must exceed stage 1 lifetime")
        if self.mean_degree > min(self.stage_1_nodes, self.stage_2_nodes) - 1:
            raise ValueError("mean_degree must fit both curriculum graph sizes")
        if self.stable_population_size > self.slots:
            raise ValueError("stable_population_size cannot exceed slots")
        return self


class LiveSessionPayload(StrictModel):
    model_id: str
    seed: int = Field(7, ge=0, lt=2**32)
    nodes: int = Field(24, ge=2, le=200)
    mean_degree: float = Field(5.0, ge=0)
    batch_size: int = Field(1, ge=1, le=64)
    dt: float = Field(.05, gt=0, le=1)
    topology: Literal["erdos_renyi", "ring"] = "erdos_renyi"
    input_seed: int = Field(108, ge=0, lt=2**32)
    input_standard_deviation: float = Field(.28, ge=0)

    @model_validator(mode="after")
    def validate_degree(self) -> "LiveSessionPayload":
        if self.mean_degree > self.nodes - 1:
            raise ValueError("mean_degree cannot exceed nodes - 1")
        return self


class LiveStepPayload(StrictModel):
    steps: int = Field(1, ge=1, le=8)
