"""Explicit receipt-projection contract; concrete integrations supply the data."""
from dataclasses import dataclass
import math
import json
from typing import Any


@dataclass(frozen=True)
class RobotContextProfile:
    name: str
    integration: str
    control_hz: float
    arm_names: tuple[str, ...]
    cameras: tuple[str, ...]
    pose_site: str
    frame: str = 'world'

    def __post_init__(self) -> None:
        if (not all(isinstance(v, str) and v for v in (self.name, self.integration, self.pose_site))
                or self.frame != 'world' or isinstance(self.control_hz, bool)
                or not isinstance(self.control_hz, (int, float))
                or not math.isfinite(self.control_hz) or self.control_hz <= 0):
            raise ValueError('invalid_robot_context_profile')
        for names in (self.arm_names, self.cameras):
            if (not isinstance(names, tuple) or not names
                    or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names)):
                raise ValueError('invalid_robot_context_profile_names')

    def metadata(self) -> dict[str, Any]:
        return {'name': self.name, 'integration': self.integration,
                'control_hz': self.control_hz, 'arms': list(self.arm_names),
                'cameras': list(self.cameras), 'pose_site': self.pose_site,
                'frame': self.frame, 'position_unit': 'm', 'angle_unit': 'rad',
                'opening_unit': 'm', 'rotation': 'Rz(yaw)@Ry(pitch)@Rx(roll)'}

    def validate_episode(self, description: Any) -> None:
        """Check declared capabilities without reading task outcomes or native data."""
        info = description.model_dump() if hasattr(description, 'model_dump') else description
        if not isinstance(info, dict) or info.get('integration') != self.integration:
            raise ValueError('robot_context_integration_mismatch')
        env = info.get('environment', {})
        hz = env.get('control_frequency_hz')
        if isinstance(hz, bool) or hz != self.control_hz:
            raise ValueError('robot_context_frequency_mismatch')
        described_arms = env.get('arms', [])
        arms = {v['name']: v for v in described_arms}
        if len(arms) != len(described_arms) or set(arms) != set(self.arm_names) or any(
                arms[n].get('control_frame') != self.pose_site for n in self.arm_names):
            raise ValueError('robot_context_arm_frame_mismatch')
        cameras = {v['name']: v for v in env.get('cameras', [])}
        if not set(self.cameras).issubset(cameras) or any(
                cameras[n].get('has_calibration') is not True for n in self.cameras):
            raise ValueError('robot_context_camera_mismatch')

    def instruction(self) -> str:
        return 'ROBOT CONTEXT PROFILE: '+json.dumps(self.metadata(), sort_keys=True)

    def validate_projected_state(self, state: dict[str, Any]) -> None:
        if (state.get('profile') != self.name or state.get('frame') != self.frame
                or state.get('pose_site') != self.pose_site
                or state.get('format') != '[x,y,z,roll,pitch,yaw,opening_m]'
                or set(state.get('arms', {})) != set(self.arm_names)):
            raise ValueError('robot_context_projected_state_mismatch')
        for values in state['arms'].values():
            if (not isinstance(values, list) or len(values) != 7
                    or any(isinstance(v, bool) or not isinstance(v, (float, int))
                           or not math.isfinite(v) for v in values[:6])
                    or (values[6] is not None and (isinstance(values[6], bool)
                        or not isinstance(values[6], (float, int)) or not math.isfinite(values[6])))):
                raise ValueError('robot_context_projected_pose_invalid')
        clock = state.get('clock', {})
        step, sim = clock.get('native_step'), clock.get('simulation_time_s')
        if (type(step) is not int or step < 0 or clock.get('control_hz') != self.control_hz
                or isinstance(sim, bool) or not isinstance(sim, (float, int)) or not math.isfinite(sim)
                or abs(sim-step/self.control_hz) > 1e-6):
            raise ValueError('robot_context_projected_clock_mismatch')


def require_profile(profile: Any) -> RobotContextProfile:
    if not isinstance(profile, RobotContextProfile):
        raise ValueError('explicit_robot_context_profile_required')
    return profile
