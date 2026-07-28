"""Translate validated API requests into domain configuration objects."""

from __future__ import annotations

from .models import AsyncTrainingPayload
from ..evolution.asynchronous import (
    AsyncEvolutionConfig,
    CurriculumLevel,
    PathologyConfig,
    ProbeConfig,
)
from ..evolution.candidate import EdgeArchitecture, RuleArchitecture


def build_async_training_config(
    payload: AsyncTrainingPayload, seed: int
) -> AsyncEvolutionConfig:
    """Build the survival-evolution configuration represented by a request."""

    architecture = RuleArchitecture(state_width=payload.state_width, hidden_width=3)
    edge_architecture = EdgeArchitecture(
        node_state_width=payload.state_width,
        latent_width=payload.state_width,
        hidden_width=3,
    )
    return AsyncEvolutionConfig(
        slots=payload.slots,
        replicas=payload.replicas,
        result_batch_size=payload.optimizer_batch,
        max_ticks=payload.max_ticks,
        candidate_budget=payload.candidate_budget,
        seed=seed,
        architecture=architecture,
        edge_architecture=edge_architecture,
        target="joint",
        levels=(
            CurriculumLevel(
                payload.stage_1_lifetime,
                input_scale=payload.input_scale,
                graph_nodes=payload.stage_1_nodes,
                mean_degree=payload.mean_degree,
            ),
            CurriculumLevel(
                payload.stage_2_lifetime,
                payload.disturbance_interval,
                payload.disturbance_strength,
                True,
                payload.input_scale * 1.5,
                payload.stage_2_nodes,
                payload.mean_degree,
            ),
        ),
        pathology=PathologyConfig(
            fatal_threshold=payload.fatal_threshold,
            node_growth_alert=payload.node_growth_alert,
            one_direction_steps=payload.one_direction_steps,
        ),
        probes=ProbeConfig(interval=payload.probe_interval, duration=2, amplitude=.10),
        curriculum_window=max(8, payload.optimizer_batch),
        censor_interval=max(2, payload.stage_1_lifetime // 5),
    )
