"""Opt-in Direct motion limits, checked only at acknowledged command boundaries."""
import math
import time

from uni_robot_agent.services.environment.contracts import RequestRejected


class MotionBudget:
    def __init__(self, description, requested_steps, *, clock=None):
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.limits = dict(description.metadata.get("direct_control", {}))
        self.wall_s = self.limits.get("wall_seconds")
        sim_s = self.limits.get("max_sim_seconds")
        for value in (self.wall_s, sim_s):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise RequestRejected("Direct motion limits must be finite and positive")
        if sim_s is not None and requested_steps > math.floor(sim_s * description.control_frequency_hz + 1e-9):
            raise RequestRejected("Requested operation exceeds the simulation-time budget")

    def expired(self):
        return self.wall_s is not None and self.clock() - self.started >= self.wall_s

    def feedback(self, reason):
        if not self.limits:
            return {}
        return {"wall_elapsed_s": self.clock() - self.started,
                "wall_budget_scope": "motion_loop_between_acknowledged_primitives",
                "requested_wall_limit_s": self.wall_s,
                "wall_limit_is_cooperative": True,
                "truncated_by_budget": reason == "wall_time_budget",
                "continuation_note": ("Execution stopped at an acknowledged command boundary because the wall-time budget was reached. "
                    "Use the returned measured state and timestamped RGB history to decide the next operation. "
                    "Do not assume the requested target was reached or replay the operation automatically."
                    if reason == "wall_time_budget" else None)}
