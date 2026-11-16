"""Append-only context and restricted explicit caching; no model or simulator."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from uni_robot_agent.context.history import ContextManager, ContextRequest
from uni_robot_agent.context.policies.append_only_observation import (
    AppendOnlyObservationPolicy, CACHE_HISTORY_PROMPT, without_cache_markers,
    validate_effective_request)
from uni_robot_agent.provider.budget.core import prepare_request, Stop, Ledger
from uni_robot_agent.provider.budget.responses import DirectSolLimits, CacheStableDirectSolBudgetGate

from uni_robot_agent.integrations.robodojo.context_profile import ROBODOJO_X5_25HZ

CAMERAS = ROBODOJO_X5_25HZ.cameras


def state(step):
    arm = {'control_pose': {'position': [.1, -.2, .9], 'quaternion': [1., 0., 0., 0.]},
           'metadata': {'gripper_opening_m': .02}}
    return {'native_steps': step, 'state_version': step, 'ended': False,
            'metadata': {'simulation_time_s': step/25, 'remaining_steps': 800-step},
            'arms': {'left': deepcopy(arm), 'right': deepcopy(arm)}}


def observation(step, *, cameras=CAMERAS):
    value = {'state': state(step), 'snapshot_id': f's{step}', 'image_state_version': step,
             'captured_at': 1794787250.+step, 'camera_calibration': {}, 'image_cameras': [],
             'feedback': {'executed_native_steps': 1, 'stop_reason': 'requested_step_budget'}}
    parts, frames = [], []
    for version in (step-1, step):
        for camera in cameras:
            frame = {'camera': camera, 'frame_id': f'{camera}-{version}', 'state_version': version,
                'simulation_time_s': version/25, 'captured_at_utc': '2026-11-16T00:00:00+00:00'}
            label = (f'[RUA RGB frame {len(frames):03d}] camera={camera}; frame_id={frame["frame_id"]}; '
                f'simulation_time_s={version/25:.3f}; captured_at_utc={frame["captured_at_utc"]}; state_version={version}.')
            parts.extend([{'type': 'input_text', 'text': label},
                          {'type': 'input_image', 'image_url': f'fixture:{camera}:{version}'}])
            frames.append(frame)
    for camera in cameras:
        value['camera_calibration'][camera] = {'intrinsic_K': [[1,0,0],[0,1,0],[0,0,1]],
                                              'T_world_from_optical': [[step]], 'image_path': f'{camera}-{step}'}
    value['temporal_rgb'] = {'frame_count': len(frames), 'frames': frames}
    return value, [{'type': 'input_text', 'text': json.dumps(value)}, *parts]


def pair(i, *, cameras=CAMERAS):
    value, parts = observation(i, cameras=cameras)
    return [{'type': 'reasoning', 'id': f'opaque{i}', 'encrypted_content': f'unchanged{i}'},
            {'type': 'function_call', 'call_id': f'c{i}', 'name': 'move', 'arguments': '{"max_steps":1}'},
            {'type': 'function_call_output', 'call_id': f'c{i}', 'output': parts}]


def body(items=None):
    initial = [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Task fixed.'}]}]
    return {'model': 'gpt-6.1-sol', 'instructions': 'Fixed robot instructions.',
            'reasoning': {'effort': 'high'}, 'input': initial if items is None else items,
            'tools': [{'type': 'namespace', 'name': 'mcp__uni_robot_agent',
                       'tools': [{'type': 'function', 'name': 'move'}]}]}


class CacheHistoryChecks(unittest.TestCase):
    def test_forty_turns_append_only_current_rgb_and_previous_boundary(self):
        policy = AppendOnlyObservationPolicy(profile=ROBODOJO_X5_25HZ); raw = body(); previous = []
        for i in range(41):
            if i: raw['input'].extend(pair(i))
            effective = policy.select(ContextRequest(raw)).request.body
            again = policy.select(ContextRequest(raw)).request.body
            self.assertEqual(effective, again)
            self.assertIn(CACHE_HISTORY_PROMPT, effective['instructions'])
            prefix = effective['input'][:-1] if i else effective['input']
            clean = without_cache_markers(prefix)
            self.assertEqual(clean[:len(previous)], previous)
            if i:
                tail = effective['input'][-1]['content'];meta = json.loads(tail[0]['text'].split('\n',1)[1])
                self.assertEqual(meta['native_step'], i)
                self.assertEqual(meta['live_frame_count'], 3)
                self.assertEqual({f['state_version'] for f in meta['frames']}, {i})
                self.assertEqual(sum(p.get('type')=='input_image' for p in tail), 3)
                self.assertNotIn('camera_calibration', json.loads(prefix[-1]['output'][0]['text']))
                if i>1:
                    self.assertIn('prompt_cache_breakpoint', prefix[-4]['output'][0])
            self.assertLessEqual(json.dumps(effective).count('"prompt_cache_breakpoint"'),3)
            previous = clean

    def test_history_header_mutation_and_compaction_fail_closed(self):
        p = AppendOnlyObservationPolicy(profile=ROBODOJO_X5_25HZ); raw = body();p.select(ContextRequest(raw))
        for key,value in [('instructions','changed'),('reasoning',{'effort':'medium'})]:
            bad=deepcopy(raw);bad[key]=value
            with self.assertRaisesRegex(ValueError,'changed'):p.select(ContextRequest(bad))
        raw['input'][0]['content'][0]['text']='changed'
        with self.assertRaisesRegex(ValueError,'changed'):p.select(ContextRequest(raw))
        with self.assertRaisesRegex(ValueError,'compaction'):p.select(ContextRequest(body(),preserve_history=True))

    def test_failed_prepare_does_not_commit_and_missing_camera_is_not_invented(self):
        p=AppendOnlyObservationPolicy(profile=ROBODOJO_X5_25HZ);raw=body();p.select(ContextRequest(raw))
        bad=deepcopy(raw);bad['input'].extend(pair(1,cameras=CAMERAS[:2]))
        with self.assertRaisesRegex(ValueError,'camera_set'):p.select(ContextRequest(bad))
        raw['input'].extend(pair(1))
        result=p.select(ContextRequest(raw)).request.body
        self.assertEqual(result['input'][-4]['encrypted_content'],'unchanged1')

    def test_bad_pair_unknown_execution_and_wrong_frame_fail(self):
        for kind in ('orphan','unknown','frame'):
            p=AppendOnlyObservationPolicy(profile=ROBODOJO_X5_25HZ);raw=body();parts=pair(1)
            if kind=='orphan':parts=parts[2:]
            else:
                v=json.loads(parts[-1]['output'][0]['text'])
                if kind=='unknown':v['feedback']['execution_uncertain']=True
                else:v['temporal_rgb']['frames'][0]['frame_id']='mismatch'
                parts[-1]['output'][0]['text']=json.dumps(v)
            raw['input'].extend(parts)
            with self.assertRaises(Exception):p.select(ContextRequest(raw))

    def test_count_has_same_semantics_without_cache_controls(self):
        p=AppendOnlyObservationPolicy(profile=ROBODOJO_X5_25HZ);raw=body();raw['input'].extend(pair(1))
        effective=p.select(ContextRequest(raw)).request.body
        generated,count=prepare_request(effective,DirectSolLimits())
        self.assertEqual(generated['input'],effective['input'])
        self.assertEqual(count['input'],without_cache_markers(generated['input']))
        self.assertNotIn('prompt_cache_options',count)
        for bad in ({'mode':'explicit','ttl':'12h'},{'mode':'implicit','ttl':'30m'}, {'prewarm':True}):
            v=deepcopy(effective);v['prompt_cache_options']=bad
            with self.assertRaises(Stop):prepare_request(v,DirectSolLimits())

    def test_trusted_parent_rejects_history_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'budget.db',DirectSolLimits());ledger.claim('mock')
            gate=CacheStableDirectSolBudgetGate(ledger,'mock',None,profile=ROBODOJO_X5_25HZ)
            p=AppendOnlyObservationPolicy(profile=ROBODOJO_X5_25HZ);raw=body()
            effective=p.select(ContextRequest(raw)).request.body
            gate.prepare(effective);gate.prepare(effective)
            raw['input'].extend(pair(1));effective=p.select(ContextRequest(raw)).request.body
            gate.prepare(effective)
            effective['input'][0]['content'][0]['text']='retroactive rewrite'
            with self.assertRaisesRegex(Stop,'history_changed'):gate.prepare(effective)
            ledger.db.close()

    def test_current_frame_times_are_relative_to_latest_capture(self):
        p=AppendOnlyObservationPolicy(profile=ROBODOJO_X5_25HZ);raw=body();parts=pair(1)
        value=json.loads(parts[-1]['output'][0]['text'])
        # Current camera readout times can differ; retain that measured difference.
        for i,f in enumerate(value['temporal_rgb']['frames']):
            if f['state_version']==1:
                stamp='2026-11-16T00:00:01+00:00' if f['camera']=='right_wrist' else '2026-11-16T00:00:00+00:00'
                old=f['captured_at_utc'];f['captured_at_utc']=stamp
                parts[-1]['output'][1+2*i]['text']=parts[-1]['output'][1+2*i]['text'].replace(old,stamp)
        parts[-1]['output'][0]['text']=json.dumps(value);raw['input'].extend(parts)
        result=p.select(ContextRequest(raw)).request.body
        meta=json.loads(result['input'][-1]['content'][0]['text'].split('\n',1)[1])
        self.assertEqual([f['relative_capture_wall_s'] for f in meta['frames']],[-1.,-1.,0.])


if __name__ == '__main__':unittest.main()
