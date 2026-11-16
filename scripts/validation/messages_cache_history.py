"""Native Messages cache/history invariants, no model or simulator access."""
from base64 import b64encode
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from cache_history import observation
from uni_robot_agent.context.history import ContextRequest
from uni_robot_agent.context.policies.append_only_observation import AppendOnlyObservationPolicy
from uni_robot_agent.context.policies.append_only_messages import (
    CACHE_CONTROL, BINDING_DROP, BINDING_BETA, persistent_messages, without_cache_control, validate_messages_effective)
from uni_robot_agent.integrations.robodojo.context_profile import ROBODOJO_X5_25HZ as X5
from uni_robot_agent.provider.budget.core import Ledger, Stop
from uni_robot_agent.provider.budget.messages import DirectFableLimits, CacheStableDirectFableBudgetGate, prepare_messages


def body():
    return {'model': 'claude-fable-5-2', 'system': 'Fixed robot instructions.',
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Task fixed.'}]}],
            'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'},
            'max_tokens': 1024, 'tools': [{'name': 'mcp__uni_robot_agent__move',
                                        'input_schema': {'type': 'object'}}]}


def pair(step, *, cameras=X5.cameras):
    _, parts = observation(step, cameras=cameras)
    native = []
    for p in parts:
        if p['type'] == 'input_text':
            native.append({'type': 'text', 'text': p['text']})
        else:
            native.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                           'data': b64encode(p['image_url'].encode()).decode()}})
    return [{'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': f'private fixture {step}', 'signature': f'signed{step}'},
                {'type': 'text', 'text': f'Brief intent {step}.'},
                {'type': 'tool_use', 'id': f'c{step}', 'name': 'mcp__uni_robot_agent__move', 'input': {'max_steps': 1}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': f'c{step}', 'content': native}]}]


def select(policy, raw):
    return policy.select(ContextRequest(deepcopy(raw), wire_format='messages')).request.body


class MessagesCacheChecks(unittest.TestCase):
    def test_eighty_turns_native_assistant_prefix_and_current_triplet(self):
        policy = AppendOnlyObservationPolicy(profile=X5)
        previous, raw = [], body()
        with tempfile.TemporaryDirectory() as d:
            ledger = Ledger(Path(d)/'ledger', DirectFableLimits());ledger.claim('mock')
            gate = CacheStableDirectFableBudgetGate(ledger, 'mock', None, profile=X5)
            try:
                for step in range(81):
                    if step: raw['messages'].extend(pair(step))
                    effective = select(policy, raw)
                    self.assertEqual(effective, select(policy, raw))
                    prefix = persistent_messages(effective)
                    self.assertEqual(prefix[:len(previous)], previous)
                    previous = prefix
                    before = [m for m in raw['messages'] if m['role'] == 'assistant']
                    after = [m for m in effective['messages'] if m['role'] == 'assistant']
                    self.assertEqual(after, before)
                    generation, count = gate.prepare(effective)
                    self.assertNotIn('cache_control', json.dumps(count))
                    self.assertEqual(without_cache_control(generation['messages']), count['messages'])
                    self.assertLessEqual(json.dumps(effective).count('"cache_control"'), 3)
                    if step:
                        tail = effective['messages'][-1]['content']
                        self.assertEqual(sum(p['type'] == 'image' for p in tail), 3)
                        self.assertNotIn('cache_control', json.dumps(tail))
                        self.assertEqual(json.loads(tail[0]['text'].split('\n', 1)[1])['native_step'], step)
                self.assertEqual(ledger.snapshot()['generation_attempts'], 0)
            finally: ledger.db.close()

    def test_native_history_and_header_changes_fail_before_commit(self):
        raw = body();raw['messages'].extend(pair(1))
        p = AppendOnlyObservationPolicy(profile=X5);select(p, raw)
        for mutation in ('system', 'thinking', 'signature', 'intent', 'tool_id', 'receipt'):
            bad = deepcopy(raw)
            if mutation == 'system':bad['system'] += 'changed'
            elif mutation == 'thinking':bad['thinking'] = {'type': 'disabled'}
            elif mutation == 'signature':bad['messages'][1]['content'][0]['signature'] = 'changed'
            elif mutation == 'intent':bad['messages'][1]['content'][1]['text'] = 'changed'
            elif mutation == 'tool_id':bad['messages'][1]['content'][2]['id'] = 'changed'
            else:bad['messages'][2]['content'][0]['content'][0]['text'] = '{}'
            reason = 'prepared_fable_adaptive' if mutation == 'thinking' else 'history_or_header_changed'
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, reason):
                select(p, bad)
        raw['messages'].extend(pair(2));select(p, raw)

    def test_partial_camera_and_bad_cache_do_not_pass_trusted_gate(self):
        p = AppendOnlyObservationPolicy(profile=X5);raw = body();select(p, raw)
        raw['messages'].extend(pair(1, cameras=('head', 'right_wrist')))
        with self.assertRaisesRegex(ValueError, 'camera_set_mismatch'):select(p, raw)
        raw['messages'][-2:] = pair(1)
        effective = select(p, raw)
        bad = deepcopy(effective);bad['messages'][-1]['content'][0]['cache_control'] = CACHE_CONTROL
        with self.assertRaisesRegex(ValueError, 'cache_boundaries'):validate_messages_effective(bad, profile=X5)
        bad = deepcopy(effective);bad['messages'][0]['content'][-1]['cache_control']['ttl'] = '1h'
        with self.assertRaisesRegex(ValueError, 'cache_boundaries'):validate_messages_effective(bad, profile=X5)

    def test_sdk_cache_markers_move_without_changing_semantics(self):
        p = AppendOnlyObservationPolicy(profile=X5);raw = body()
        old = select(p, raw)
        raw['messages'][0]['content'][0]['cache_control'] = {'type': 'ephemeral', 'ttl': '1h'}
        raw['messages'].extend(pair(1))
        current = select(p, raw)
        self.assertEqual(persistent_messages(current)[:1], persistent_messages(old))
        self.assertNotIn('1h', json.dumps(current))

    def test_inline_system_messages_remain_native_not_observation_heads(self):
        raw = body()
        inline = {'role': 'system', 'content': [{'type': 'text', 'text': 'Native CLI environment.'}]}
        raw['messages'].append(inline)
        p = AppendOnlyObservationPolicy(profile=X5)
        old = select(p, raw)
        self.assertEqual(old['messages'][1], inline)
        raw['messages'][1] = {'role': 'system', 'content': inline['content'][0]['text']}
        raw['messages'].extend(pair(1))
        new = select(p, raw)
        self.assertEqual(new['messages'][1], inline)
        self.assertEqual(persistent_messages(new)[:2], persistent_messages(old))

    def test_current_rgb_precedes_trailing_system_and_persistent_text_still_appends(self):
        raw = body();raw['messages'].extend(pair(1))
        inline = {'role': 'system', 'content': [{'type': 'text', 'text': 'Native CLI reminder.'}]}
        raw['messages'].append(inline)
        p = AppendOnlyObservationPolicy(profile=X5);old = select(p, raw)
        self.assertEqual(old['messages'][-1], inline)
        self.assertTrue(old['messages'][-2]['content'][0]['text'].startswith('[RUA CURRENT OBSERVATION]\n'))
        validate_messages_effective(old, profile=X5)
        bad = deepcopy(old);bad['messages'][-2:] = reversed(bad['messages'][-2:])
        with self.assertRaisesRegex(ValueError, 'inline_system_placement'):validate_messages_effective(bad, profile=X5)
        raw['messages'].extend(pair(2));new = select(p, raw)
        self.assertEqual(persistent_messages(new)[:len(persistent_messages(old))], persistent_messages(old))

    def test_consecutive_native_system_group_and_effort_guard(self):
        raw = body();raw['messages'].extend(pair(1))
        raw['messages'].extend([{'role':'system','content':[],'output_config':{'effort':'high'}},
                               {'role':'system','content':[{'type':'text','text':'Native reminder.'}]}])
        effective=select(AppendOnlyObservationPolicy(profile=X5),raw)
        validate_messages_effective(effective,profile=X5)
        prepare_messages(effective,DirectFableLimits())
        bad=deepcopy(effective);bad['messages'][-2]['output_config']['effort']='max'
        with self.assertRaisesRegex(Stop,'inline_effort_mismatch'):prepare_messages(bad,DirectFableLimits())

    def test_binding_controls_match_in_count_generation_and_headers(self):
        raw=body();raw['messages'].extend(pair(1))
        effective=select(AppendOnlyObservationPolicy(profile=X5),raw)
        self.assertEqual(effective['thinking']['block_binding'],BINDING_DROP)
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger',DirectFableLimits());ledger.claim('fixture')
            calls=[]
            gate=CacheStableDirectFableBudgetGate(ledger,'fixture',lambda *args:calls.append(args),profile=X5)
            try:
                prepared,count=gate.prepare(effective)
                self.assertEqual(prepared['thinking'],count['thinking'])
                for path,b in ((gate.count_path,count),(gate.generation_path,prepared)):
                    gate.forward_request(path,b,{'anthropic-beta':'existing-beta'})
                self.assertTrue(all(v[2]['anthropic-beta']=='existing-beta,'+BINDING_BETA for v in calls))
                bad=deepcopy(effective);bad['thinking'].pop('block_binding')
                with self.assertRaisesRegex(ValueError,'thinking_binding'):validate_messages_effective(bad,profile=X5)
            finally:ledger.db.close()

    def test_compaction_and_foreign_cache_fields_rejected(self):
        for update in ({'context_management': {'edits': [{'type': 'compact'}]}},
                       {'prompt_cache_options': {'mode': 'explicit', 'ttl': '30m'}}):
            p = AppendOnlyObservationPolicy(profile=X5)
            with self.assertRaises(ValueError):select(p, {**body(), **update})
        with self.assertRaisesRegex(ValueError, 'compaction'):
            AppendOnlyObservationPolicy(profile=X5).select(ContextRequest(body(), wire_format='messages', preserve_history=True))

    def test_only_archived_media_pruning_is_reconciled(self):
        p = AppendOnlyObservationPolicy(profile=X5);raw = body();raw['messages'].extend(pair(1))
        before = select(p, raw)
        content = raw['messages'][2]['content'][0]['content']
        # Both forms occur in the pinned CLI once its raw image window fills.
        images = [i for i, v in enumerate(content) if v['type'] == 'image']
        content[images[0]] = {'type': 'text', 'text': '[media removed: request limit. fixture]'}
        del content[images[1]]
        self.assertEqual(select(p, raw), before)
        bad = deepcopy(raw);bad['messages'][2]['content'][0]['content'][0]['text'] = '{}'
        with self.assertRaisesRegex(ValueError, 'history_or_header_changed'):select(p, bad)
        bad = deepcopy(raw)
        next(v for v in bad['messages'][2]['content'][0]['content'] if v['type'] == 'image')['source']['data'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'history_or_header_changed'):select(p, bad)


if __name__ == '__main__':unittest.main()
