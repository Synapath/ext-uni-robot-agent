"""Cancellation reasons survive the public session boundary."""
import unittest
from unittest.mock import Mock
from uni_robot_agent.session import Session


class SessionCancellationTests(unittest.TestCase):
    def test_claude_cancel_is_cooperative_and_first_reason_wins(self):
        from uni_robot_agent.agents.claude_code.agent import ClaudeCodeAgent
        agent=ClaudeCodeAgent(model='fixture')
        agent._loop=Mock();agent._cancel_scope=Mock()
        agent.cancel('wall_time_limit');agent.cancel('cancelled')
        self.assertEqual(agent._cancel_reason,'wall_time_limit')
        agent._loop.call_soon_threadsafe.assert_called_once_with(agent._cancel_scope.cancel)

    def test_default_and_explicit_stop_reasons_keep_the_same_api(self):
        session=Session.__new__(Session);session._agent=Mock();session._closed=False
        session.cancel();session._agent.cancel.assert_called_once_with('interrupted')
        session._agent.reset_mock();session.cancel(reason='wall_time_limit')
        session._agent.cancel.assert_called_once_with('wall_time_limit')
        session._agent.reset_mock();session._closed=True;session.cancel()
        session._agent.cancel.assert_not_called()


if __name__=='__main__':unittest.main()
