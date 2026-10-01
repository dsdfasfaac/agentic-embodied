# Copyright (c) 2026 Zetta Contributors
"""Fresh Cosmos inference; only this adapter receives proprioception."""

from __future__ import annotations

import numpy as np


class ZevaPlanner:
    def __init__(
        self, predict, policy_observation, *, execution_steps, action_horizon=32
    ):
        if not 1 <= execution_steps <= action_horizon:
            raise ValueError("invalid execution prefix")
        self.predict = predict
        self.policy_observation = policy_observation
        self.execution_steps = execution_steps
        self.action_horizon = action_horizon

    def prepare(self, args, context):
        return ZevaPlan(self, args.max_chunks)


class ZevaPlan:
    def __init__(self, planner, chunks):
        self.planner, self.remaining = planner, chunks
        self.limit = chunks * planner.execution_steps
        self.planned_steps = None
        self.reached = None

    def next_targets(self, context):
        if not self.remaining:
            return None
        p = self.planner
        actions = np.asarray(p.predict(p.policy_observation()), dtype=np.float32)
        if actions.shape != (p.action_horizon, 14) or not np.isfinite(actions).all():
            raise ValueError("invalid Zeva action chunk")
        self.remaining -= 1
        return actions[: p.execution_steps].copy()

    def on_commit(self, context):
        pass


class CosmosPredictor:
    def __init__(self, *, host, port, contract, task, timeout_s, inference_seed=None):
        self.host, self.port, self.contract, self.task = host, port, contract, task
        self.timeout_s, self.inference_seed = timeout_s, inference_seed

    def __call__(self, observation):
        from robots.arx.control import prepare_model_state
        from robots.arx.cosmos_edge_client import CosmosEdgeClient

        with CosmosEdgeClient(
            self.host, self.port, self.contract, timeout_sec=self.timeout_s
        ) as client:
            return client.predict(
                observation.images,
                prepare_model_state(observation.state, self.task),
                self.task.instruction,
                seed=self.inference_seed,
            ).actions
