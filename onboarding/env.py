"""One task definition shared by training, evaluation, and tests."""

from pathlib import Path

import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "scene.xml"
MAX_EPISODE_STEPS = 1_000
FRAME_SKIP = 2
RESET_NOISE = 0.01


def make_env(render_mode: str | None = None, max_episode_steps: int = MAX_EPISODE_STEPS) -> gym.Env:
    """Use the custom scene with Gymnasium's supplied balancing task."""
    # MEMBER TODO 3: Return gym.make("InvertedPendulum-v5", ...).
    # Pass in values for the xml_file, frame_skip, reset_noise_scale, max_episode_steps, render_mode, width, height, and camera_name arguments
    # All of these values are provided through existing variables or function inputs
    # Set width=640, height=480, and camera_name="side" for consistent videos.
    # Gymnasium supplies the reward/reset rules and adds the TimeLimit wrapper.
    raise NotImplementedError("Section 3: connect the custom scene to Gymnasium")
