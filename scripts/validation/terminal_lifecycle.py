"""Terminal planning/admission/result regression checks; no vendor or simulator."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import httpx

from uni_robot_agent.agents.contracts import AgentResult
from uni_robot_agent.artifacts.store import ArtifactStore
from uni_robot_agent.integrations.base import EpisodeOutcome
from uni_robot_agent.integrations.prepared_episode import PreparedEpisodeSession
from uni_robot_agent.provider.gateway import ProviderGateway
from uni_robot_agent.provider.protocols.messages import MessagesProtocol
from uni_robot_agent.provider.recording.exchanges import get_model_exchange_error
from uni_robot_agent.session import _finalize_run
from uni_robot_agent.tools.toolkit import Toolkit


class TerminalLifecycle(unittest.TestCase):
    def test_acknowledged_terminal_is_not_an_uncertain_ack(self):
        session=PreparedEpisodeSession.__new__(PreparedEpisodeSession)
        session.episode=SimpleNamespace(env=SimpleNamespace(uncertain=False),state=SimpleNamespace(ended=False))
        self.assertIsNone(session.terminal_reason())
        session.episode.state.ended=True
        self.assertEqual(session.terminal_reason(),'environment_terminal')
        session.episode.env.uncertain=True
        self.assertIsNone(session.terminal_reason())

    def test_terminal_callback_is_optional_and_not_a_task_verdict(self):
        toolkit=Toolkit()
        self.assertIsNone(toolkit.terminal_reason())
        toolkit.set_terminal_fn(lambda:'environment_terminal')
        self.assertIsNone(toolkit.task_succeeded())
        self.assertEqual(toolkit.terminal_reason(),'environment_terminal')

    def test_local_terminal_never_forwards_responses_or_messages(self):
        for messages in (False,True):
            with self.subTest(messages=messages):
                stopped=[]
                gateway=ProviderGateway('http://127.0.0.1:9/v1',
                    protocol=MessagesProtocol() if messages else None,
                    generation_guard=lambda:'environment_terminal',on_local_stop=stopped.append)
                gateway._client.send=Mock(side_effect=AssertionError('Must not contact any upstream'))
                endpoint=gateway.start()
                body=({'model':'claude-fable-5-2','max_tokens':1,'messages':[{'role':'user','content':'fixture'}]}
                    if messages else {'model':'gpt-6.1-sol','instructions':'Fixture','input':[{'role':'user','content':'fixture'}]})
                try:
                    reply=httpx.post(endpoint+('/messages' if messages else '/responses'),json=body,trust_env=False)
                    self.assertEqual(reply.status_code,409)
                    self.assertEqual(reply.json()['error']['code'],'rua_lifecycle_stop')
                finally:gateway.stop()
                gateway._client.send.assert_not_called()
                self.assertEqual(stopped,['environment_terminal'])
                self.assertIsNone(get_model_exchange_error(gateway.trace_records,interrupted=True))
                self.assertIsNotNone(get_model_exchange_error(gateway.trace_records,interrupted=False))

    def test_actual_result_writer_keeps_native_verdict_and_late_error_separate(self):
        for success,complete,uncertain in ((True,True,False),(False,True,False),(None,False,False),(True,True,True)):
            with self.subTest(success=success,complete=complete,uncertain=uncertain),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                outcome=EpisodeOutcome(success=success,metadata={'complete':complete,'native_score':1.0 if success else 0.0,'steps':2})
                integration=SimpleNamespace(finalize=lambda **kw:outcome)
                executor=Mock()
                if uncertain:executor.close.side_effect=RuntimeError('Unacknowledged physical execution')
                assembly=SimpleNamespace(executor=executor,observation_profile=None,
                    integration_session=integration,allow_automatic_retry=False,artifacts=ArtifactStore(root),session_dir=root)
                args=SimpleNamespace(integration='fixture',agent_backend='codex',model='fixture',instruction='fixture',
                    max_turns=0,episode_timeout_s=60,reasoning_effort='high')
                result=AgentResult(error='late SDK error',provider_error='late provider error')
                subscriptions=SimpleNamespace(close=lambda:None,errors={})
                returned=_finalize_run(assembly,subscriptions,args,result,None,time.time())
                saved=json.loads((root/'result.json').read_text())
                expected=success if complete and not uncertain else None
                self.assertEqual(saved['task_success'],expected)
                self.assertEqual(saved['error'],'late SDK error')
                self.assertEqual(saved['provider_error'],'late provider error')
                self.assertEqual(returned.exit_code,1)
                if complete and not uncertain:
                    self.assertEqual(saved['score'],1.0 if success else 0.0)
                    self.assertIs(saved['native_complete'],True)


if __name__=='__main__':unittest.main()
