"""Explicit simulation, UTC capture and process-local steady-clock semantics."""
from datetime import datetime, timezone
import math

from uni_robot_agent.provider.budget.core import Stop


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def epoch(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            raise Stop('timestamp_timezone_missing')
        return dt.timestamp()
    raise Stop('invalid_wall_timestamp')


def state_clock(state, *, control_hz):
    if isinstance(control_hz, bool) or not math.isfinite(control_hz) or control_hz <= 0:
        raise Stop('invalid_control_frequency')
    step = state.get('native_steps')
    if type(step) is not int or step < 0:
        raise Stop('native_step_missing')
    sim = state.get('metadata', {}).get('simulation_time_s', step/control_hz)
    if not math.isfinite(sim) or abs(sim-step/control_hz) > 1e-6:
        raise Stop('simulation_clock_mismatch')
    return {'schema_version': 1, 'control_hz': control_hz, 'native_step': step,
            'state_version': state.get('state_version', step), 'simulation_time_s': sim}


def frame_clock(frame, *, control_hz, current_step, latest_capture, returned_at=None, sample_index=0):
    if isinstance(control_hz, bool) or not math.isfinite(control_hz) or control_hz <= 0:
        raise Stop('invalid_control_frequency')
    sim = frame['simulation_time_s']
    step = frame['state_version']
    if type(step) is not int or step < 0 or step > current_step or abs(sim-step/control_hz) > 1e-6:
        raise Stop('future_or_inconsistent_frame')
    capture = epoch(frame['captured_at_utc'])
    relative = capture-epoch(latest_capture)
    # UTC may jump; preserve such differences as wall-clock facts, never as durations.
    return {'camera': frame['camera'], 'frame_id': frame['frame_id'],
            'native_step': step, 'state_version': step, 'simulation_time_s': sim,
            'captured_at_utc': frame['captured_at_utc'], 'sample_index': sample_index,
            'sensor_frame_index': frame.get('sensor_frame_index'),
            'relative_sim_s': round(sim-current_step/control_hz, 6),
            'relative_capture_wall_s': round(relative, 6),
            'age_at_tool_return_s': (None if returned_at is None else
                                     round(epoch(returned_at)-capture, 6))}
