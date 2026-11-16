"""Prepared RoboDojo dual-X5 context contract; no native runtime imports."""
from uni_robot_agent.context.robot_profile import RobotContextProfile

ROBODOJO_X5_25HZ = RobotContextProfile(
    name='robodojo_x5_25hz_v1', integration='robodojo', control_hz=25,
    arm_names=('left', 'right'), cameras=('head', 'left_wrist', 'right_wrist'),
    pose_site='x5_grasp_site',
)
