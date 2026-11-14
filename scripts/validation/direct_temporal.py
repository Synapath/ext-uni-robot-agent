"""CPU-only acceptance for temporal RGB, bounded actions and model-visible receipts."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import tempfile
import unittest

import numpy as np
from uni_robot_agent.artifacts.store import ArtifactStore
from uni_robot_agent.context.history import ContextRequest
from uni_robot_agent.context.policies.observation_window import ObservationWindowPolicy
from uni_robot_agent.execution import ExecutionResult
from uni_robot_agent.services.environment.history import RGBHistory
from uni_robot_agent.services.environment.snapshots import CameraFrame, CapturedObservation, SnapshotStore
from uni_robot_agent.services.environment.contracts import ArmDescription, ArmState, EnvironmentDescription, EnvironmentState, Pose, StepReceipt
from uni_robot_agent.services.mcp.results import encode_call_result
from uni_robot_agent.tools.gripper.contracts import HoldRequest, MoveRequest
from uni_robot_agent.tools.gripper.motion import MotionExecutor
from uni_robot_agent.tools.scene.perception import snapshot_images, snapshot_metadata
from uni_robot_agent.tools.toolkit import ToolOutput

CAMS = ['head', 'left_wrist', 'right_wrist']


def history(ticks):
    result = RGBHistory(episode_id='cpu')
    for index in range(ticks + 1):
        result.record(simulation_time_s=index / 25, state_version=index,
            captured_at=1700000000 + index / 25,
            cameras={name: CameraFrame(np.full((4, 6, 3), index % 255, np.uint8)) for name in CAMS})
    return result


class Backend:
    def __init__(self, clock, *, delay=1.2, limits=True):
        self.clock, self.delay, self.calls = clock, delay, 0
        self.description = EnvironmentDescription(arms=[ArmDescription(name='left', control_frame='grasp',
            grasp_site_description='Test grasp site', rotation_radius_m=0.1)],
            control_frequency_hz=25., max_episode_steps=800,
            metadata={'direct_control': {'max_sim_seconds':1., 'wall_seconds':1.}} if limits else {})
        self.state = EnvironmentState(arms={'left':ArmState(control_pose=Pose(position=(0.,0.,0.),
            quaternion=(1.,0.,0.,0.)), joints=[0.]*6, gripper=1., gripper_command=1.)},
            state_version=0, native_steps=0, ended=False)

    def read_state(self):
        return self.state.model_copy(deep=True)

    def step_pose(self, command):
        self.calls += 1
        self.clock[0] += self.delay
        self.state.arms['left'].control_pose = command.arms['left'].pose
        self.state.state_version += 1
        self.state.native_steps += 1
        return StepReceipt(accepted=True, state=self.read_state())


class Checks(unittest.TestCase):
    def test_codex_failed_terminal_is_not_clean_completion(self):
        from uni_robot_agent.agents.codex.agent import _consume_main_turn
        turn=SimpleNamespace(stream=lambda:iter([{'method':'turn/completed',
            'payload':{'turn':{'error':{'message':'mock SDK failure'}}}}]))
        state={'pending_reasoning':False}
        _consume_main_turn(turn,state,[],0,None,None)
        self.assertEqual(state['error'],'mock SDK failure')
        self.assertEqual(state['stop_reason'],'sdk_error')
    def test_sampling_short_and_full_history(self):
        self.assertEqual(len(history(0).sample(CAMS)['head']), 1)
        h = history(100).sample(CAMS)
        self.assertEqual([x.state_version for x in h['head']], list(range(55,101,5)))
        self.assertEqual(sum(map(len,h.values())),30)
        self.assertTrue(all(x.simulation_time_s > 2 and x.simulation_time_s <= 4 for x in h['head']))
        self.assertEqual([x.simulation_time_s for x in h['head']], [x.simulation_time_s for x in h['right_wrist']])

    def test_sampling_rate_uses_real_ticks(self):
        for hz in (2., 5., 10.):
            h=history(75)
            h.sample_hz=hz
            h.max_groups=int(2*hz)
            frames=h.sample(CAMS)['head']
            self.assertEqual(frames[-1].state_version,75)
            self.assertTrue(all(b.simulation_time_s-a.simulation_time_s>=1/hz-1e-9
                                for a,b in zip(frames,frames[1:])))
            self.assertLessEqual(len(frames),2*hz)

    def test_invalid_history_and_duplicate_time(self):
        with self.assertRaises(ValueError):RGBHistory(episode_id='x', sample_hz=26)
        h=history(1)
        h.record(simulation_time_s=.04,state_version=1,captured_at=1700000001,cameras={})
        self.assertEqual(len(h.frames),2)
        with self.assertRaises(ValueError):h.record(simulation_time_s=0,state_version=0,captured_at=0,cameras={})

    def test_labels_and_old_clip_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            store=ArtifactStore(Path(directory))
            h=history(55).sample(CAMS)
            snapshot=SnapshotStore(store).publish(CapturedObservation('cpu',55,1700000003,
                {'state_version':55}, {name:h[name][-1].frame for name in CAMS},history=h))
            mirror=SimpleNamespace(store=store, mirror_ref=lambda ref:ref,
                                   local_path=lambda ref:store.local_path(ref['artifact_id']))
            images=snapshot_images(None,snapshot,mirror=mirror)
            self.assertEqual(len(images),30)
            self.assertIn('captured_at_utc=',images[0].label)
            self.assertIn('simulation_time_s=',images[0].label)
            payload={**snapshot_metadata(snapshot,[x.name for x in images],mirror=mirror),
                     'image_cameras':[x.name for x in images]}
            encoded=encode_call_result(ExecutionResult.completed(ToolOutput(extra=payload,images=images)))
            self.assertEqual(encoded['content'][0]['type'],'text')
            parts=[{'type':'input_text','text':x['text']} if x['type']=='text' else
                   {'type':'input_image','image_url':'data:image/png;base64,'+x['data']} for x in encoded['content']]
            items=[]
            for index in range(2):
                items += [{'type':'function_call','call_id':str(index),'name':'move'},
                          {'type':'function_call_output','call_id':str(index),'output':copy.deepcopy(parts)}]
            decision=ObservationWindowPolicy(observation_horizon=1).select(ContextRequest(body={'input':items}))
            outputs=decision.request.body['input']
            self.assertEqual(sum(p['type']=='input_image' for x in outputs for p in x.get('output',[])),30)
            old=next(json.loads(p['text']) for p in outputs[1]['output'] if p.get('text','').startswith('{'))
            self.assertEqual(old['temporal_rgb']['frames'],[])
            self.assertFalse(any(p.get('text','').startswith('[RUA RGB frame ') for p in outputs[1]['output']))
            store.close()

    def test_wall_cut_waits_for_acknowledgement(self):
        clock=[0.]
        backend=Backend(clock)
        with patch('uni_robot_agent.tools.gripper.budget.time.monotonic',lambda:clock[0]):
            result=MotionExecutor(backend).move(MoveRequest(targets=[{'arm':'left','pose':[.5,0.,0.,0.,0.,0.]}]))
        self.assertEqual(backend.calls,1)
        self.assertEqual(result['steps_executed'],1)
        self.assertEqual(result['stop_reason'],'wall_time_budget')
        self.assertTrue(result['truncated_by_budget'])
        self.assertGreaterEqual(result['wall_elapsed_s'],1.2)

    def test_hold_limit_and_no_automatic_replay(self):
        backend=Backend([0.])
        with self.assertRaises(ValueError):MotionExecutor(backend).hold(HoldRequest(steps=26))
        self.assertEqual(backend.calls,0)
        def unknown(command):
            backend.calls+=1
            raise RuntimeError('Unknown acknowledgement')
        backend.step_pose=unknown
        with self.assertRaisesRegex(RuntimeError,'Unknown'):MotionExecutor(backend).hold(HoldRequest(steps=1))
        self.assertEqual(backend.calls,1)

    def test_default_motion_compatibility(self):
        backend=Backend([0.],limits=False)
        result=MotionExecutor(backend).hold(HoldRequest(steps=3))
        self.assertEqual(backend.calls,3)
        self.assertNotIn('truncated_by_budget',result)


if __name__=='__main__':unittest.main()
