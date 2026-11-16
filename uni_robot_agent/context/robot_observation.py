"""Reusable structured robot receipts and timestamped RGB projection."""
from copy import deepcopy
import json
import math
import warnings
from .wire_history import FRAME, payload, records
from .timing import frame_clock, state_clock, utc
from .robot_profile import require_profile
from uni_robot_agent.provider.budget.core import Stop

DEMO = '[RUA DEMONSTRATIONS]'
CURRENT = '[RUA CURRENT OBSERVATION]'

def robot_state(state, *, profile):
    profile = require_profile(profile)
    from scipy.spatial.transform import Rotation
    if set(state.get('arms', {})) != set(profile.arm_names):
        raise Stop('measured_arm_profile_mismatch')
    out = {'clock': state_clock(state, control_hz=profile.control_hz),
           'profile': profile.name, 'frame': profile.frame, 'pose_site': profile.pose_site,
           'format': '[x,y,z,roll,pitch,yaw,opening_m]', 'arms': {}, 'drive_intent': {}}
    for name in profile.arm_names:
        arm = state.get('arms', {}).get(name)
        if arm is None:
            raise Stop('measured_arm_missing')
        pose = arm['control_pose']; q = pose['quaternion']
        if len(q) != 4 or not all(math.isfinite(v) for v in [*pose['position'], *q]):
            raise Stop('measured_pose_invalid')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Gimbal lock detected')
            rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz').tolist()
        meta = arm.get('metadata', {})
        gap = meta.get('gripper_opening_m')
        if gap is not None and not math.isfinite(gap):
            raise Stop('measured_opening_invalid')
        out['arms'][name] = [round(float(v), 6) for v in [*pose['position'], *rpy]] + [
            None if gap is None else round(float(gap), 6)]
        out['drive_intent'][name] = {k: meta[k] for k in
            ('gripper_target_opening_m', 'gripper_applied_target_opening_m') if k in meta}
    return out


def compact_result(value, *, profile):
    profile = require_profile(profile)
    if value is None:
        return None
    out = deepcopy(value)
    for key in ('temporal_rgb', 'image_cameras', 'observation_path'):
        out.pop(key, None)
    if 'state' in out:
        state = out.pop('state')
        out['robot_state'] = robot_state(state, profile=profile)
        out['sim_feedback'] = {'ended': state.get('ended'),
                              'remaining_steps': state.get('metadata', {}).get('remaining_steps')}
    if 'feedback' in out:
        feedback = out.pop('feedback')
        if feedback.get('execution_uncertain') or feedback.get('execution_outcome') == 'unknown':
            raise Stop('execution_unknown')
        out['execution'] = {k: v for k, v in feedback.items() if k not in ('arms', 'ik_statistics')}
        duplicate = ('actual_position', 'actual_quaternion', 'actual_pose', 'target_position',
                     'target_quaternion', 'target_pose') if 'robot_state' in out else ()
        out['tool_result'] = {name: {k: v for k, v in arm.items() if k not in duplicate}
            for name, arm in feedback.get('arms', {}).items()}
        if feedback.get('ik_statistics') and feedback.get('stop_reason') in (
                'planning_rejected', 'unsafe_motion', 'tracking_interrupted', 'reachable_prefix'):
            out['execution']['ik_statistics'] = feedback['ik_statistics']
    return out


def frames(parts, value, *, profile):
    profile = require_profile(profile)
    found, overlays, label, transport = [], [], None, ''
    for part in parts if isinstance(parts, list) else []:
        match = FRAME.search(part.get('text', ''))
        if match:
            label = {'camera': match[1], 'frame_id': match[2],
                     'simulation_time_s': float(match[3]), 'captured_at_utc': match[4],
                     'state_version': int(match[5]), 'label': deepcopy(part)}
        elif part.get('type') in ('text', 'input_text') and 'RUA transport sha256=' in part.get('text', ''):
            if label: label['label']['text'] += '\n'+part['text']
            else: transport = part['text']
        if part.get('type') == 'input_image':
            if label:
                found.append(dict(label, image=deepcopy(part))); label = None
            else:
                overlays.append({'image': deepcopy(part), 'transport': transport})
            transport = ''
    expected = (value or {}).get('temporal_rgb', {}).get('frame_count', 0)
    if expected != len(found):
        raise Stop('temporal_metadata_image_mismatch')
    if found:
        if value.get('image_state_version') not in (None, value.get('state_version', value.get('state', {}).get('state_version'))):
            raise Stop('calibration_state_mismatch')
        declared = value.get('temporal_rgb', {}).get('frames')
        if declared:
            if len(declared) != len(found): raise Stop('temporal_metadata_image_mismatch')
            for a, b in zip(declared, found):
                if any(a.get(k) != b[k] for k in ('camera', 'frame_id', 'state_version')):
                    raise Stop('frame_label_metadata_mismatch')
    # The initial capture has no temporal samples yet. Its images are still current.
    cameras = (value or {}).get('image_cameras', [])
    state = (value or {}).get('state', {})
    if not found and overlays and len(cameras) == len(overlays) and state and value.get('captured_at') is not None:
        current = state_clock(state, control_hz=profile.control_hz)
        for camera, image in zip(cameras, overlays):
            f = {'camera': camera, 'frame_id': f"{value['snapshot_id']}:{camera}:current",
                 'simulation_time_s': current['simulation_time_s'],
                 'captured_at_utc': utc(value['captured_at']), 'state_version': current['state_version']}
            label = f"[RUA RGB frame {len(found):03d}] camera={camera}; frame_id={f['frame_id']}; simulation_time_s={f['simulation_time_s']:.3f}; captured_at_utc={f['captured_at_utc']}; state_version={f['state_version']}."
            if image['transport']: label += '\n'+image['transport']
            found.append(dict(f, label={'type': 'input_text', 'text': label}, image=image['image']))
        overlays = []
    return found, overlays


def select_frames(source, *, profile, current_step, max_groups=3, window_s=2):
    profile = require_profile(profile)
    if type(max_groups) is not int or not 1 <= max_groups <= 3 or not 0 < window_s <= 2:
        raise Stop('invalid_simple_window')
    if not source:
        return []
    versions = sorted({f['state_version'] for f in source})
    if versions[-1] != current_step:
        raise Stop('observation_not_current')
    by_group = {v: [f for f in source if f['state_version'] == v] for v in versions}
    expected = {f['camera'] for f in by_group[current_step]}
    seen = set()
    for f in source:
        key = f['camera'], f['frame_id']
        if key in seen:
            raise Stop('duplicate_frame_id')
        seen.add(key)
        frame_clock(f, control_hz=profile.control_hz, current_step=current_step,
                    latest_capture=source[-1]['captured_at_utc'])
    if any({f['camera'] for f in group} != expected or len(group) != len(expected)
           for group in by_group.values()):
        raise Stop('unsynchronized_frame_groups')
    versions = [v for v in versions if 0 <= (current_step-v)/profile.control_hz <= window_s+1e-9]
    if len(versions) > max_groups:
        indices = ([len(versions)-1] if max_groups == 1 else [0, len(versions)-1]
                   if max_groups == 2 else [0, len(versions)//2, len(versions)-1])
        versions = [versions[i] for i in indices]
    return [f for v in versions for f in by_group[v]]
