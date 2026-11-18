"""Deferred episode clocks preserve persistent admission and old defaults."""
from pathlib import Path
import tempfile
import unittest
from uni_robot_agent.provider.budget import Ledger, Limits, Stop


class BudgetClockTests(unittest.TestCase):
    def test_deferred_clock_is_one_shot_and_never_refunds_setup_or_old_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = [100.0]
            ledger = Ledger(Path(tmp)/'clock.sqlite3', Limits(wall_seconds=2700), clock=lambda: now[0])
            try:
                ledger.claim('deferred', defer_clock=True)
                now[0] += 3600  # Deployment/debug time cannot consume rollout time.
                with self.assertRaisesRegex(Stop, 'episode_clock_not_started'):
                    ledger.check_generation('deferred')
                self.assertEqual(ledger.start_clock('deferred'), 3700)
                ledger.reserve('deferred', 10, 10)
                with self.assertRaisesRegex(Stop, 'episode_clock_already_started'):
                    ledger.start_clock('deferred')
                now[0] += 2699
                ledger.check('deferred')
                now[0] += 1
                with self.assertRaisesRegex(Stop, 'wall_time_limit'):
                    ledger.check('deferred')
                self.assertEqual(ledger.snapshot()['generation_attempts'], 1)
                self.assertGreater(ledger.snapshot()['unsettled_reserved_usd'], 0)
                ledger.stop('deferred', 'wall_time_limit')
                with self.assertRaisesRegex(Stop, 'wall_time_limit'):
                    ledger.start_clock('deferred')
            finally:
                ledger.db.close()

    def test_default_claim_and_closed_deferred_claim_cannot_be_rearmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Ledger(Path(tmp)/'clock.sqlite3',Limits())
            try:
                ledger.claim('ordinary')
                ledger.check('ordinary')
                with self.assertRaisesRegex(Stop,'episode_clock_already_started'):
                    ledger.start_clock('ordinary')
                with self.assertRaisesRegex(Stop,'episode_not_claimed'):
                    ledger.start_clock('unknown')
            finally:ledger.db.close()
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Ledger(Path(tmp)/'clock.sqlite3',Limits())
            try:
                ledger.claim('closed',defer_clock=True);ledger.stop('closed','setup_failed')
                with self.assertRaisesRegex(Stop,'setup_failed'):ledger.start_clock('closed')
            finally:ledger.db.close()


if __name__=='__main__':unittest.main()
