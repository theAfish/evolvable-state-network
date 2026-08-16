"""Validated request models at the application boundary."""

from __future__ import annotations

from math import ceil, pi
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base request model that rejects misspelled or unsupported fields."""

    model_config = ConfigDict(extra="forbid")


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
    initial_state_scale: float = Field(.12, gt=0, le=1)
    stage_1_lifetime: int = Field(40, ge=4, le=2000)
    stage_2_lifetime: int = Field(100, ge=8, le=5000)
    stage_1_nodes: int = Field(8, ge=3, le=100)
    stage_2_nodes: int = Field(12, ge=3, le=200)
    mean_degree: float = Field(3.0, ge=0.5, le=20)
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


class EmbodiedFoodWebTrainingPayload(StrictModel):
    """One species-specific evolutionary run in the predator–prey–plant task."""

    model_id: str | None = None
    continue_run_id: str | None = None
    seed: int | None = Field(None, ge=0, lt=2**32)
    training_mode: Literal["batch", "continuous"] = "batch"
    batch_population_mode: Literal["shared_rule_cohort", "mixed_individual_population"] = "shared_rule_cohort"
    algorithm: Literal["cma_es", "genetic"] = "cma_es"
    execution_backend: Literal["python", "torch"] = "torch"
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    workers: int = Field(0, ge=0, le=32)
    body_inputs: tuple[Literal["hunger", "energy_change", "ate", "time_since_meal"], ...] = Field(("hunger",), min_length=1)
    population_size: int = Field(24, ge=2, le=64)
    prey_count: int = Field(5, ge=1, le=64)
    predator_count: int = Field(2, ge=0, le=64)
    hidden_nodes: int = Field(31, ge=1, le=367)
    state_width: int = Field(2, ge=2, le=8)
    node_hidden_layers: tuple[int, ...] = Field((8,), min_length=1, max_length=6)
    node_activation: Literal["tanh", "relu", "gelu", "silu"] = "tanh"
    edge_hidden_layers: tuple[int, ...] = Field((12,), min_length=1, max_length=6)
    edge_activation: Literal["tanh", "relu", "gelu", "silu"] = "tanh"
    edge_latent_width: int = Field(3, ge=1, le=16)
    mean_degree: float = Field(6.0, ge=0, le=40)
    allow_input_output_connections: bool = False
    initial_state_scale: float = Field(.12, ge=0, le=1)
    network_dt: float = Field(.05, gt=0, le=.25)
    max_delta: float = Field(.12, gt=0, le=.5)
    edge_step_scale: float = Field(.06, gt=0, le=.5)
    rule_output_scale: float = Field(1.0, gt=0, le=1.0)
    mutation_sigma: float | None = Field(None, gt=0, le=2.0)
    elite_fraction: float = Field(.25, gt=0, lt=1)
    immigrant_fraction: float = Field(.25, ge=0, lt=1)
    immigrant_sigma: float | None = Field(None, gt=0, le=4.0)
    immigrant_mode: Literal["zero", "population"] = "zero"
    max_genome_norm: float | None = Field(None, gt=0)
    max_parameter_magnitude: float | None = Field(None, gt=0)
    initial_energy_scale: float = Field(1.0, gt=0, le=20)
    max_food: int = Field(80, ge=0, le=10_000)
    food_growth_rate: float = Field(24.0, ge=0, le=10_000)
    max_speed: float = Field(20.0, gt=0, le=100)
    max_turn: float = Field(2 * pi, gt=0, le=8 * pi)
    plant_cluster_count: int = Field(4, ge=0, le=64)
    plant_cluster_radius: float = Field(5.0, ge=0, le=100)
    ticks: int = Field(600, ge=1, le=100_000)
    batch_generations: int = Field(48, ge=1)
    batch_episode_steps: int = Field(256, ge=8, le=5000)
    batch_trials: int = Field(4, ge=1, le=32)
    batch_validation_trials: int = Field(8, ge=1, le=32)
    batch_test_trials: int = Field(16, ge=1, le=64)
    batch_opponents: int = Field(2, ge=1, le=16)
    enforce_survival_pressure: bool = True

    @model_validator(mode="after")
    def validate_degree(self) -> "EmbodiedFoodWebTrainingPayload":
        if any(width < 1 or width > 512 for width in self.node_hidden_layers + self.edge_hidden_layers):
            raise ValueError("every hidden layer width must be between 1 and 512")
        # One port per selected body input, 27 ray-image ports, and two actions.
        if self.mean_degree > self.hidden_nodes + len(self.body_inputs) + 29 - 1:
            raise ValueError("mean_degree cannot exceed total network nodes - 1")
        if len(set(self.body_inputs)) != len(self.body_inputs):
            raise ValueError("body_inputs cannot contain duplicates")
        if self.model_id and self.continue_run_id:
            raise ValueError("choose either a basic model or a prior embodied run, not both")
        if self.execution_backend == "python" and self.device == "cuda":
            raise ValueError("the reference Python backend cannot run on CUDA")
        if (
            self.training_mode == "batch"
            and self.batch_population_mode == "mixed_individual_population"
            and self.population_size * max(self.prey_count, self.predator_count, 1) > 512
        ):
            raise ValueError(
                "mixed-individual population is limited to 512 genomes per species; "
                "reduce population_size or the corresponding species count"
            )
        natural_lifetime_steps = self.initial_energy_scale * 9.0 / (3.6 * .125)
        minimum_survival_horizon = ceil(3.0 * natural_lifetime_steps)
        if (
            self.training_mode == "batch" and self.enforce_survival_pressure
            and self.batch_episode_steps < minimum_survival_horizon
        ):
            raise ValueError(
                f"set batch_episode_steps to at least {minimum_survival_horizon} "
                f"for initial_energy_scale={self.initial_energy_scale:g}; "
                "or lower initial_energy_scale or explicitly disable survival-pressure enforcement"
            )
        prey_supply = self.food_growth_rate * 2.0 if self.max_food > 0 else 0.0
        prey_demand = self.prey_count * 3.6
        if self.enforce_survival_pressure and prey_supply < prey_demand:
            raise ValueError(
                "plant regrowth cannot sustain the configured prey population even under perfect collection; "
                "increase food_growth_rate, lower prey_count, or explicitly disable survival-pressure enforcement"
            )
        return self


class EmbodiedDemoPayload(StrictModel):
    run_id: str
    seed: int = Field(7, ge=0, lt=2**32)
    network_hidden_nodes: int | None = Field(
        None,
        ge=1,
        le=367,
        description="Anonymous recurrent nodes in each newly sampled demonstration network",
    )
    network_mean_degree: float | None = Field(
        None,
        ge=0,
        le=400,
        description="Requested mean degree for each newly sampled demonstration network",
    )
    prey_count: int | None = Field(None, ge=1, le=64)
    predator_count: int | None = Field(None, ge=0, le=64)
    initial_food: int | None = Field(None, ge=0, le=10_000)
    max_food: int | None = Field(None, ge=0, le=10_000)
    food_growth_rate: float | None = Field(None, ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_food_amount(self) -> "EmbodiedDemoPayload":
        if self.initial_food is not None and self.max_food is not None and self.initial_food > self.max_food:
            raise ValueError("initial_food cannot exceed max_food")
        return self


class EmbodiedDemoStepPayload(StrictModel):
    ticks: int = Field(1, ge=1, le=128)


class EmbodiedRandomGraphDiagnosticPayload(StrictModel):
    """Post-run evaluation of fixed saved rules on newly sampled embodied graphs."""

    sample_count: int = Field(64, ge=1, le=256)
    seed: int | None = Field(None, ge=0, lt=2**32)


class EmbodiedRunComparisonPayload(StrictModel):
    """Common synthetic probes for comparing two saved embodied rule sets."""

    left_run_id: str
    right_run_id: str
    probe_count: int = Field(1024, ge=32, le=16_384)
    evaluation_samples: int = Field(4, ge=1, le=32)
    parameter_scales: tuple[float, ...] = Field((1.0, .5, .1), min_length=1, max_length=8)
    seed: int | None = Field(None, ge=0, lt=2**32)

    @model_validator(mode="after")
    def validate_scales(self) -> "EmbodiedRunComparisonPayload":
        if any(scale <= 0 or scale > 10 for scale in self.parameter_scales):
            raise ValueError("parameter scales must be in (0, 10]")
        return self


class LiveSessionPayload(StrictModel):
    model_id: str
    seed: int = Field(7, ge=0, lt=2**32)
    initial_state_scale: float = Field(
        .12,
        ge=0,
        le=1,
        description="Standard deviation of independent zero-mean initial node coordinates",
    )
    nodes: int = Field(24, ge=2, le=200)
    mean_degree: float = Field(5.0, ge=0)
    batch_size: int = Field(1, ge=1, le=64)
    dt: float = Field(.05, gt=0, le=1)
    topology: Literal["erdos_renyi", "ring"] = "erdos_renyi"

    @model_validator(mode="after")
    def validate_degree(self) -> "LiveSessionPayload":
        if self.mean_degree > self.nodes - 1:
            raise ValueError("mean_degree cannot exceed nodes - 1")
        return self


class LiveStepPayload(StrictModel):
    steps: int = Field(1, ge=1, le=8)
