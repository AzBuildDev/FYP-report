"""Portable semantic IDs and a local checkpoint descriptor. No model/hardware imports."""
from pathlib import Path

LEGACY_PROFILE = 'legacy_red'
RED_BLUE_PROFILE = 'red_blue_30k'
ROOT = Path(__file__).resolve().parents[2]
RED_BLUE_CHECKPOINT = '模型/红蓝混合120段_20261028_均衡选色_bs64_30000/best_val'
RED_BLUE_SHA256 = '9a902835afd04a453866d346241789d8e3b6e350fcd03446ee6888739cdda310'
RED_TEXT = 'Pick up the red cube and place it in the target area.'
BLUE_TEXT = 'Pick up the blue cube and place it in the target area.'

def task_texts(profile=LEGACY_PROFILE):
    # Return a fresh mapping; callers cannot mutate the shared allowlist.
    if profile == LEGACY_PROFILE:
        return {'red_cube_to_zone_A': RED_TEXT}
    if profile == RED_BLUE_PROFILE:
        return {'red_cube_to_zone_A': RED_TEXT, 'blue_cube_to_zone_A': BLUE_TEXT}
    raise ValueError('unknown_policy_profile')

def policy_binding(profile):
    task_texts(profile)  # reject unknown profiles
    if profile == LEGACY_PROFILE:
        return None
    return {
        'profile': RED_BLUE_PROFILE,
        'checkpoint_relative_path': RED_BLUE_CHECKPOINT,
        'checkpoint_sha256': RED_BLUE_SHA256,
        'checkpoint_step': 30000,
        'validation_status': 'offline_trained_pending_hardware_validation',
        'checkpoint_loaded': False,
        'hardware_validated': False,
        'input_images': ['observation.images.front', 'observation.images.wrist'],
        'image_shape_chw': [3, 480, 640],
        'state_names': ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                        'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos'],
        'action_shape': [50, 6],
        'target_binding': 'zone_A denotes the demonstrated target area; physical binding unverified',
    }
