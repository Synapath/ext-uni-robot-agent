"""Bound real RGB samples by simulation time; never synthesize missing frames."""
from collections import deque
import math

from .snapshots import CapturedHistoryFrame


class RGBHistory:
    def __init__(self, *, episode_id, window_s=2.0, sample_hz=5.0, control_hz=25.0):
        if any(not math.isfinite(x) or x <= 0 for x in (window_s, sample_hz, control_hz)):
            raise ValueError("RGB history settings must be finite and positive")
        if sample_hz > control_hz or window_s * sample_hz < 1:
            raise ValueError("RGB history requires at least one sample and cannot exceed control Hz")
        self.episode_id, self.window_s, self.sample_hz = episode_id, window_s, sample_hz
        self.max_groups = math.floor(window_s * sample_hz + 1e-9)
        self.capacity = math.ceil(window_s * control_hz) + 2
        self.frames = deque(maxlen=self.capacity)

    def record(self, *, simulation_time_s, state_version, captured_at, cameras):
        if self.frames and simulation_time_s == self.frames[-1][0]:
            if state_version != self.frames[-1][1]:
                raise ValueError("State changed without advancing simulation time")
            return
        if self.frames and (simulation_time_s < self.frames[-1][0] or state_version < self.frames[-1][1]):
            raise ValueError("RGB history cannot move backwards")
        group = {name: CapturedHistoryFrame(
            f"{self.episode_id}:{name}:{state_version}", captured_at, simulation_time_s,
            state_version, frame).freeze() for name, frame in cameras.items()}
        self.frames.append((simulation_time_s, state_version, group))
        while self.frames and self.frames[0][0] <= simulation_time_s - self.window_s + 1e-9:
            self.frames.popleft()

    def sample(self, cameras):
        if not self.frames:
            return {}
        newest = self.frames[-1][0]
        selected, previous = [], None
        for timestamp, _, group in reversed(self.frames):
            if timestamp <= newest - self.window_s + 1e-9:
                break
            if not set(cameras) <= set(group):
                raise ValueError("RGB history must contain synchronized camera groups")
            if previous is None or previous - timestamp >= 1 / self.sample_hz - 1e-9:
                selected.append(group)
                previous = timestamp
                if len(selected) == self.max_groups:
                    break
        return {name: tuple(group[name] for group in reversed(selected)) for name in cameras}
