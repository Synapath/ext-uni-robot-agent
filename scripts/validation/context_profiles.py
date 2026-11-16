"""Prepared robot-profile selection, contract rejection and clock independence."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uni_robot_agent.config import AgentConfig, SessionConfig, resolve_context_config, validate_context_episode
from uni_robot_agent.context.history import ContextRequest
from uni_robot_agent.context.policies.append_only_observation import AppendOnlyObservationPolicy, validate_effective_request
from uni_robot_agent.context.robot_observation import robot_state, select_frames
from uni_robot_agent.context.robot_profile import RobotContextProfile
from uni_robot_agent.context.timing import state_clock, frame_clock
from uni_robot_agent.integrations.robodojo.context_profile import ROBODOJO_X5_25HZ as X5
from uni_robot_agent.provider.budget.core import Ledger
from uni_robot_agent.provider.budget.responses import DirectSolLimits, CacheStableDirectSolBudgetGate
from cache_history import body, pair, state


def description():
    return {'integration': 'robodojo', 'environment': {
        'control_frequency_hz': 25,
        'arms': [{'name': n, 'control_frame': 'x5_grasp_site'} for n in X5.arm_names],
        'cameras': [{'name': n, 'has_calibration': True} for n in X5.cameras]}}


def args(**values):
    return SimpleNamespace(integration='robodojo', context_policy='append_only_observation',
        context_robot_profile=X5.name, **values)


class ProfileChecks(unittest.TestCase):
    def test_explicit_python_selection_and_default_compatibility(self):
        config = resolve_context_config(args())
        self.assertEqual(config['context_options']['profile'], X5)
        self.assertEqual(resolve_context_config(SimpleNamespace(context_policy='identity'))['context_options'], {})
        self.assertIsNone(AgentConfig(model='gpt-6.1-sol').context_robot_profile)
        for field, value in [('context_robot_profile', None), ('context_robot_profile', 'unknown')]:
            a = args(); setattr(a, field, value)
            with self.assertRaises(ValueError): resolve_context_config(a)
        for policy in ['identity', 'observation_window', 'sliding_window']:
            a=args();a.context_policy=policy
            with self.assertRaisesRegex(ValueError,'requires_append_only'):resolve_context_config(a)
        with self.assertRaisesRegex(ValueError, 'explicit_robot'): AppendOnlyObservationPolicy()

    def test_python_session_validation_before_connection(self):
        values=dict(integration='robodojo',env_url='http://127.0.0.1:1',no_sam3=True,
                    tool_profile='direct',agent=AgentConfig(model='gpt-6.1-sol',
                    provider_url='http://127.0.0.1:2',context_policy='append_only_observation',
                    context_robot_profile=X5.name))
        resolved=SessionConfig(**values).resolve()
        self.assertEqual(resolve_context_config(resolved)['context_options']['profile'],X5)
        values['agent']=values['agent'].model_copy(update={'context_robot_profile':None})
        with self.assertRaisesRegex(ValueError,'explicit_robot'):SessionConfig(**values).resolve()

    def test_cli_selection_and_session_start_hook(self):
        from uni_robot_agent.cli import build_parser
        from uni_robot_agent.integrations import load_integration
        from uni_robot_agent.config import resolve_config
        from uni_robot_agent.session import assemble_session
        spec=load_integration('robodojo').get_integration_spec()
        parser=build_parser(spec)
        argv=['run','--integration','robodojo','--agent','codex','--model','gpt-6.1-sol',
              '--env-url','http://127.0.0.1:1','--provider-url','http://127.0.0.1:2',
              '--no-sam3','--tool-profile','direct','--context-policy','append_only_observation',
              '--context-robot-profile',X5.name]
        a=resolve_config(parser.parse_args(argv),parser,spec,argv)
        self.assertEqual(resolve_context_config(a)['context_options']['profile'],X5)
        with tempfile.TemporaryDirectory() as d:
            a.output_dir=d
            bad=description();bad['environment']['control_frequency_hz']=50
            from unittest.mock import MagicMock
            fake=MagicMock();fake.episode.description=bad
            fake_spec=SimpleNamespace(create_session=lambda context:fake,tool_profiles={'direct':()})
            module=SimpleNamespace(get_integration_spec=lambda:fake_spec)
            with patch('uni_robot_agent.session.load_integration',return_value=module), \
                 patch('uni_robot_agent.provider.gateway.ProviderGateway.start') as model:
                with self.assertRaisesRegex(ValueError,'frequency_mismatch'):assemble_session(a)
                model.assert_not_called();fake.start.assert_called_once();fake.close.assert_called_once()

    def test_integration_cannot_borrow_another_robot_profile(self):
        a=args();a.integration='piper_x'
        with self.assertRaisesRegex(ValueError,'unsupported_robot'):resolve_context_config(a)

    def test_profile_enables_native_claude_but_not_silent_compaction(self):
        a=args(context_max_input_tokens=100)
        with self.assertRaisesRegex(ValueError,'silently trim'):resolve_context_config(a)
        config=SessionConfig(integration='robodojo',env_url='http://127.0.0.1:1',no_sam3=True,
            tool_profile='direct',agent=AgentConfig(model='claude-fable-5-2',agent_backend='claude_code',
            provider_url='http://127.0.0.1:2',context_policy='append_only_observation',
            context_robot_profile=X5.name))
        config.resolve()

    def test_prepared_description_rejected_before_model_start(self):
        validate_context_episode(args(),SimpleNamespace(description=description()))
        changes=[('integration','piper_x'),('control_frequency_hz',50),
                 ('control_frame','piper_site'),('cameras',[]),('arms',[])]
        for key,value in changes:
            d=description()
            if key=='integration':d[key]=value
            elif key=='control_frame':d['environment']['arms'][0][key]=value
            else:d['environment'][key]=value
            with self.subTest(key=key), patch('uni_robot_agent.provider.gateway.ProviderGateway.start') as start:
                with self.assertRaises(ValueError):validate_context_episode(args(),SimpleNamespace(description=d))
                start.assert_not_called()

    def test_helpers_have_no_implicit_robot_or_rate(self):
        with self.assertRaises(TypeError):state_clock(state(1))
        with self.assertRaises(TypeError):robot_state(state(1))
        alt=RobotContextProfile(name='fixture_50hz',integration='fixture',control_hz=50,
            arm_names=('arm',),cameras=('front',),pose_site='fixture_tip')
        s=state(50);s['arms']={'arm':s['arms']['left']};s['metadata']['simulation_time_s']=1.
        projected=robot_state(s,profile=alt)
        self.assertEqual(projected['clock']['simulation_time_s'],1.)
        self.assertEqual(projected['pose_site'],'fixture_tip')
        self.assertEqual(set(projected['arms']),{'arm'})
        alt.validate_projected_state(projected)
        f={'camera':'front','frame_id':'f','state_version':50,'simulation_time_s':1.,
           'captured_at_utc':'2026-11-16T00:00:00+00:00'}
        self.assertEqual(frame_clock(f,control_hz=50,current_step=100,latest_capture=f['captured_at_utc'])['relative_sim_s'],-1.)
        self.assertEqual(select_frames([f],profile=alt,current_step=50),[f])
        with self.assertRaises(Exception):state_clock(s,control_hz=25)

    def test_trusted_profile_binding_and_invalid_pose(self):
        p=AppendOnlyObservationPolicy(profile=X5);raw=body();raw['input'].extend(pair(1))
        effective=p.select(ContextRequest(raw)).request.body
        validate_effective_request(effective,profile=X5)
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'budget.db',DirectSolLimits());ledger.claim('mock')
            with self.assertRaises(ValueError):CacheStableDirectSolBudgetGate(ledger,'mock',None)
            gate=CacheStableDirectSolBudgetGate(ledger,'mock',None,profile=X5)
            gate.prepare(effective)
            for location in ['state','tail']:
                bad=deepcopy(effective)
                if location=='state':
                    output=bad['input'][-2]['output'][0]
                    value=json.loads(output['text']);value['robot_state']['pose_site']='wrong'
                    output['text']=json.dumps(value)
                else:
                    output=bad['input'][-1]['content'][0]
                    value=json.loads(output['text'].split('\n',1)[1]);value['robot_profile']['control_hz']=50
                    output['text']='[RUA CURRENT OBSERVATION]\n'+json.dumps(value)
                with self.assertRaises(ValueError):gate.prepare(bad)
            for value in [float('nan'),float('inf'),True]:
                bad=deepcopy(effective);output=bad['input'][-1]['content'][0]
                meta=json.loads(output['text'].split('\n',1)[1]);meta['simulation_time_s']=value
                output['text']='[RUA CURRENT OBSERVATION]\n'+json.dumps(meta)
                with self.assertRaises(ValueError):gate.prepare(bad)
            self.assertEqual(ledger.snapshot()['generation_attempts'],0)
            ledger.db.close()
        bad=robot_state(state(1),profile=X5);bad['arms']['left'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'pose_invalid'):X5.validate_projected_state(bad)


if __name__=='__main__':unittest.main()
