"""CPU boundary checks; real native reset still requires simulator validation."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from uni_robot_agent.integrations.robodojo.environment.native_compat import reset_episode, validate_seed
from uni_robot_agent.integrations.robodojo import deployment
from uni_robot_agent.services.deployment.config import DeploymentConfig


class SeedChecks(unittest.TestCase):
    def test_reset_keeps_exact_layout_for_each_seed(self):
        for seed in (0, 1, 2, 3):
            with self.subTest(seed=seed):
                env = SimpleNamespace(seed_manager=SimpleNamespace(), env_seeds=[])
                def reset(*, seed):
                    env.env_seeds = seed
                    env.actual = seed[0]
                env.reset = reset
                env.get_env_seed = lambda slot: env.actual
                reset_episode(env, Path('/fixed-layout.json'), seed=seed)
                self.assertEqual(env.env_seeds, [seed])
                self.assertEqual(env.seed_manager.seed_info, {seed: {'scene_layout': '/fixed-layout.json'}})

    def test_wrong_native_seed_is_not_published(self):
        env = SimpleNamespace(seed_manager=SimpleNamespace(), env_seeds=[0],
                              reset=lambda **kwargs: None, get_env_seed=lambda slot: 0)
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            reset_episode(env, Path('/fixed-layout.json'), seed=2)

    def test_seed_type_and_numpy_range(self):
        for seed in (-1, 2**32, True, 1.5, '1', None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                validate_seed(seed)

    def test_launch_plan_forwards_seed_and_hash_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root/'task/RoboDojo/config').mkdir(parents=True)
            (root/'task/RoboDojo/config/stack_bowls.yml').write_text('')
            layout = root/'frozen.json';layout.write_text('{}')
            runtime = SimpleNamespace(python='python', environment={},
                                      data={'native_root': str(root), 'assets_dir': str(root)})
            for seed in (None, 0, 1, 2, 3):
                args = SimpleNamespace(seed=seed,command='deploy',task='stack_bowls',native_root=root,
                    layout=layout,layout_sha256=None,port=18760,host='127.0.0.1',
                    depth=False,camera_calibration=True,runtime_profile=None,rua_source_root=None)
                deployment.validate(args)
                plan = deployment.plan(args,runtime,root,'test')
                self.assertIn('--seed', plan.command)
                self.assertEqual(plan.command[plan.command.index('--seed')+1],str(seed or 0))
                self.assertIn(str(layout),plan.command)
                self.assertNotIn('--layout-set', plan.command)
                self.assertNotIn('--continuous-recording', plan.command)

    def test_continuous_runtime_keeps_layout_set_and_seed_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root/'task/RoboDojo/config/general_pickup.yml'
            config_path.parent.mkdir(parents=True)
            config_path.touch()
            layout = root/'layout.json'
            layout.write_text('{}')
            args = DeploymentConfig(integration='robodojo', task='general_pickup',
                layout=layout, layout_set=2, seed=7, continuous_recording=True).resolve()
            runtime = SimpleNamespace(python='python', environment={},
                                      data={'native_root': str(root), 'assets_dir': str(root)})
            plan = deployment.plan(args, runtime, root, 'test')
            self.assertIn('--continuous-recording', plan.command)
            self.assertEqual(plan.command[plan.command.index('--layout-set')+1], '2')
            self.assertEqual(plan.command[plan.command.index('--seed')+1], '7')
            self.assertEqual(plan.expected['metadata.layout_set'], 2)
            self.assertEqual(plan.expected['metadata.seed'], 7)


if __name__ == '__main__':
    unittest.main()
