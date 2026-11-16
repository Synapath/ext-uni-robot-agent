"""Immutable robot text history with a replaceable current RGB suffix."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import threading

from uni_robot_agent.context.history import ContextDecision, register_context_policy
from uni_robot_agent.context.image_transport import restore_selected
from uni_robot_agent.context.observation_history import estimate_context
from uni_robot_agent.context.robot_observation import CURRENT, DEMO, compact_result, frames, select_frames
from uni_robot_agent.context.timing import frame_clock, state_clock, epoch
from uni_robot_agent.context.robot_profile import require_profile
from uni_robot_agent.context.wire_history import FRAME, records


CACHE_HISTORY_PROMPT = """PERSISTENT HISTORY AND CURRENT OBSERVATION
The fixed task/episode context and chronological text history are immutable.
Each new turn appends the original model instructions and acknowledged tool receipts.
The final CURRENT OBSERVATION block contains ONE current synchronized RGB group,
with camera/frame IDs, native step, simulation/UTC times and matching calibration.
Older image IDs in text are provenance references, not pixels still in this request.
Fixed DEMONSTRATIONS are earlier episodes, never current observations.
Use the latest acknowledged measured robot state and current images for decisions.
Historical localization results and failures keep their sources; they are not live
object poses after motion. Tool success/reached refers to the requested operation,
not task completion. Home means the initial full poses and opening conditions.
Simulation pauses during inference. Inspect actual execution and truncation before
continuing; never replay an action with an unknown acknowledgement.
All visible robot instructions and action explanations must be in English.
"""


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def without_cache_markers(value):
    if isinstance(value, dict):
        return {k: without_cache_markers(v) for k, v in value.items() if k != 'prompt_cache_breakpoint'}
    if isinstance(value, list):
        return [without_cache_markers(v) for v in value]
    return value


def mark(item):
    key = 'output' if item.get('type') == 'function_call_output' else 'content'
    parts = item.get(key)
    if not isinstance(parts, list) or not parts or parts[-1].get('type') != 'input_text':
        raise ValueError('cache_boundary_requires_text_block')
    parts[-1]['prompt_cache_breakpoint'] = {'mode': 'explicit'}


@register_context_policy('append_only_observation')
class AppendOnlyObservationPolicy:
    """One instance per ContextManager/session; no hardware or model queries."""
    name, version = 'append_only_observation', '2'

    def __init__(self, *, profile=None):
        self.profile = require_profile(profile)
        self.cameras = self.profile.cameras
        self._raw = []
        self._text = []
        self._header = None
        self._head = None
        self._lock = threading.RLock()

    @property
    def options(self):
        return {'current_groups': 1, 'cameras': list(self.cameras), 'text_history': 'append_only',
                'cache_mode': 'explicit', 'ttl': '30m', 'robot_profile': self.profile.metadata()}

    def select(self, request):
        with self._lock:
            return self._select(request)

    def _select(self, request):
        body = request.body
        if request.wire_format != 'responses' or request.preserve_history:
            raise ValueError('append_only_requires_native_responses_without_compaction')
        if any(body.get(k) for k in ('previous_response_id', 'conversation', 'context_management')):
            raise ValueError('append_only_requires_explicit_complete_history')
        items = body.get('input')
        if not isinstance(items, list) or not items:
            raise ValueError('append_only_requires_message_history')
        if ' chars truncated' in json.dumps(body):
            raise ValueError('observation_json_truncated')
        header = fingerprint({k: body.get(k) for k in
            ('model', 'instructions', 'tools', 'tool_choice', 'parallel_tool_calls',
             'reasoning', 'text', 'prompt_cache_key')})
        digests = [fingerprint(i) for i in items]
        if self._header is not None and (header != self._header or digests[:len(self._raw)] != self._raw):
            raise ValueError('append_only_history_or_header_changed')
        rows = records(body)
        row_by_index = {r['result_index']: r for r in rows}
        latest = None
        current = -1
        overlays = []
        for row in rows:
            value = row['payload'] or {}
            if value.get('state'):
                step = state_clock(value['state'], control_hz=self.profile.control_hz)['native_step']
                if step < current:
                    raise ValueError('measured_state_regressed')
                current = step
            fs, extra = frames(row['result'].get('output'), value, profile=self.profile)
            if fs:
                latest = (row, fs)
            if extra:
                overlays.append((row, extra))

        text = deepcopy(self._text)
        for index in range(len(text), len(items)):
            item = deepcopy(items[index])
            row = row_by_index.get(index)
            if row and row['payload'] is not None:
                value = compact_result(row['payload'], profile=self.profile)
                value.pop('camera_calibration', None)
                value['action_id'] = row['call_id']
                item['output'] = [{'type': 'input_text', 'text': json.dumps(value, separators=(',', ':'))}]
            else:
                for key in ('content', 'output'):
                    parts = item.get(key)
                    if not isinstance(parts, list):
                        continue
                    demo = any(DEMO in p.get('text', '') for p in parts)
                    images = [p for p in parts if p.get('type') == 'input_image']
                    if demo and len(images) > 6:
                        raise ValueError('demonstration_image_limit')
                    if images and not demo:
                        raise ValueError('unclassified_history_image')
                    item[key] = [p for p in parts if not FRAME.search(p.get('text', ''))]
            text.append(item)
        head = self._head
        if head is None:
            first_call = next((i for i, v in enumerate(text) if v.get('type') == 'function_call'), len(text))
            head = next((i for i in range(first_call-1, -1, -1)
                         if text[i].get('type') == 'message' and text[i].get('role') in ('user', 'developer')), None)
            if head is None:
                raise ValueError('fixed_context_message_missing')
            if isinstance(text[head].get('content'), str):
                text[head]['content'] = [{'type': 'input_text', 'text': text[head]['content']}]
            text[head]['content'].append({'type': 'input_text', 'text': '[RUA FIXED CONTEXT END]'})

        effective = deepcopy(body)
        effective['input'] = deepcopy(text)
        effective['instructions'] = (body.get('instructions', '') if CACHE_HISTORY_PROMPT in body.get('instructions', '')
                                     else CACHE_HISTORY_PROMPT+'\n'+body.get('instructions', ''))
        profile_note = self.profile.instruction()
        if profile_note not in effective['instructions']:
            effective['instructions'] += '\n'+profile_note
        effective['prompt_cache_options'] = {'mode': 'explicit', 'ttl': '30m'}
        boundaries = [head] + [r['result_index'] for r in rows][-2:]
        for index in dict.fromkeys(boundaries):
            mark(effective['input'][index])
        live = []
        if latest:
            row, fs = latest
            if current < 0:
                current = max(f['state_version'] for f in fs)
            live = select_frames(fs, profile=self.profile, current_step=current, max_groups=1)
            if {f['camera'] for f in live} != set(self.cameras) or len(live) != len(self.cameras):
                raise ValueError('current_camera_set_mismatch')
            value = row['payload']
            calibration = value.get('camera_calibration', {})
            if not set(self.cameras).issubset(calibration):
                raise ValueError('current_calibration_missing')
            meta = {'schema_version': 1, 'snapshot_id': value['snapshot_id'],
                    'robot_profile': self.profile.metadata(),
                    'native_step': current, 'state_version': current, 'simulation_time_s': current/self.profile.control_hz,
                    'live_frame_count': len(live), 'camera_calibration': calibration,
                    'frames': [frame_clock(f, control_hz=self.profile.control_hz, current_step=current,
                        latest_capture=max((x['captured_at_utc'] for x in live), key=epoch),
                        returned_at=value.get('tool_timing', {}).get('completed_at_utc')) for f in live]}
            content = [{'type': 'input_text', 'text': CURRENT+'\n'+json.dumps(meta, separators=(',', ':'))}]
            for f in live:
                content.extend([f['label'], f['image']])
            selected = []
            for source, images in overlays:
                v = source['payload'] or {}
                if source['name'] == 'locate' and (v.get('snapshot_id') == meta['snapshot_id'] or v.get('state_version') == current):
                    selected.extend(images)
            for image in selected[-3:]:
                content.extend([{'type': 'input_text', 'text': '[RUA CURRENT LOCALIZATION OVERLAY]'+
                                 ('\n'+image['transport'] if image['transport'] else '')}, image['image']])
            effective['input'].append({'type': 'message', 'role': 'user', 'content': content})
        effective = restore_selected(effective)
        validate_effective_request(effective, profile=self.profile)
        before, after = estimate_context(body), estimate_context(effective)
        # Commit only after all decoding, image restoration and validation succeed.
        self._raw, self._text, self._header, self._head = digests, text, header, head
        return ContextDecision(request=replace(request, body=effective), policy=self.name, version=self.version,
            truncated=True, estimated_tokens_before=before['text']+before['images'],
            estimated_tokens_after=after['text']+after['images'], kept_item_ids=(),
            reason='append_only_text_current_rgb', options={**self.options,
                'retained_pairs': len(rows), 'live_images': len(live), 'cache_boundaries': boundaries,
                'stable_prefix_sha256': fingerprint(text)})


def validate_effective_request(body, *, profile):
    """Validate the actual selected wire view, including trusted-hop checks."""
    from uni_robot_agent.context.wire_history import payload
    from uni_robot_agent.context.image_transport import has_references
    profile = require_profile(profile)
    cameras = profile.cameras
    if profile.instruction() not in body.get('instructions', ''):
        raise ValueError('robot_context_profile_header_missing')
    if has_references(body):
        raise ValueError('selected_images_not_restored')
    rows = records(body)
    latest = -1
    for row in rows:
        v = row['payload'] or {}
        if 'state' in v:
            raise ValueError('unprojected_robot_state')
        if 'camera_calibration' in v:
            raise ValueError('calibration_in_persistent_receipt')
        if 'robot_state' in v:
            profile.validate_projected_state(v['robot_state'])
        clock = v.get('robot_state', {}).get('clock')
        if clock:
            if clock['native_step'] < latest:
                raise ValueError('measured_state_regressed')
            latest = clock['native_step']
    tails = []
    demos = 0
    for index, item in enumerate(body['input']):
        content = item.get('content', item.get('output', []))
        if not isinstance(content, list):
            continue
        texts = [p.get('text', '') for p in content]
        count = sum(p.get('type') == 'input_image' for p in content)
        if any(t.startswith(CURRENT+'\n') for t in texts):
            meta = json.loads(next(t for t in texts if t.startswith(CURRENT+'\n')).split('\n', 1)[1])
            overlays = sum(t.startswith('[RUA CURRENT LOCALIZATION OVERLAY]') for t in texts)
            if (meta.get('robot_profile') != profile.metadata()
                    or type(meta.get('native_step')) is not int or meta['native_step'] < 0
                    or meta.get('state_version') != meta['native_step']
                    or not isinstance(meta.get('simulation_time_s'), (int, float))
                    or isinstance(meta['simulation_time_s'], bool) or not math.isfinite(meta['simulation_time_s'])
                    or abs(meta['simulation_time_s']-meta['native_step']/profile.control_hz) > 1e-6
                    or index != len(body['input'])-1 or count != len(cameras)+overlays or overlays > 3
                    or meta['live_frame_count'] != len(cameras) or len(meta['frames']) != len(cameras)
                    or {f['camera'] for f in meta['frames']} != set(cameras)
                    or not set(cameras).issubset(meta.get('camera_calibration', {}))):
                raise ValueError('invalid_current_observation_tail')
            if latest >= 0 and meta['native_step'] != latest:
                raise ValueError('current_observation_not_latest_state')
            for f in meta['frames']:
                if f['state_version'] != meta['native_step']:
                    raise ValueError('historical_rgb_in_current_only_view')
                frame_clock(f, control_hz=profile.control_hz, current_step=meta['native_step'],
                            latest_capture=f['captured_at_utc'])
            tails.append(meta)
        elif count:
            if not any(DEMO in t for t in texts):
                raise ValueError('unclassified_history_image')
            demos += count
    if len(tails) > 1 or demos > 6 or (latest >= 0 and not tails):
        raise ValueError('current_observation_missing_or_image_limit')
