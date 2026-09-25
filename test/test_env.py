from pathlib import Path

import gymnasium as gym
import numpy as np
from numpy.testing import assert_allclose
import pytest
from stable_baselines3.common.env_checker import check_env

from onboarding.env import make_env


def test_environment_contract(env: gym.Env) -> None:
    observation, info = env.reset(seed=7)
    assert observation.shape == (4,) and observation.dtype == np.float64
    assert isinstance(info, dict) and env.observation_space.contains(observation)
    assert np.isneginf(env.observation_space.low).all()
    assert np.isposinf(env.observation_space.high).all()
    assert env.action_space.shape == (1,) and env.action_space.dtype == np.float32
    assert_allclose(env.action_space.low, [-3])
    assert_allclose(env.action_space.high, [3])
    assert_allclose(observation, np.concatenate([env.unwrapped.data.qpos, env.unwrapped.data.qvel]))
    assert env.spec.max_episode_steps == 1000
    assert env.unwrapped.frame_skip == 2
    assert_allclose(env.unwrapped.dt, 0.04)
    assert env.metadata["render_fps"] == 25
    env.action_space.seed(1)
    for _ in range(100):
        previous_time = env.unwrapped.data.time
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert_allclose(env.unwrapped.data.time - previous_time, 0.04)
        assert np.isfinite(obs).all() and env.observation_space.contains(obs)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert reward in (0, 1) and isinstance(info, dict)
        if terminated or truncated:
            env.reset()
    check_env(env, warn=True, skip_render_check=True)


def test_seeded_reset(env: gym.Env) -> None:
    first, _ = env.reset(seed=19)
    env.step(np.array([1.0]))
    repeat, _ = env.reset(seed=19)
    assert_allclose(first, repeat, rtol=0, atol=0)
    samples = np.array([env.reset(seed=seed)[0] for seed in range(30)])
    assert (np.abs(samples) <= 0.01).all()
    assert (samples.std(axis=0) > 0).all()


@pytest.mark.parametrize("angle, expected_reward", [(0.0, 1), (0.3, 0), (-0.3, 0)])
def test_reward_and_angle_failure(env: gym.Env, angle: float, expected_reward: int) -> None:
    env.reset(seed=1)
    env.unwrapped.set_state(np.array([0.0, angle]), np.zeros(2))
    _, reward, terminated, truncated, _ = env.step(np.zeros(1))
    assert reward == expected_reward
    assert terminated == (expected_reward == 0)
    assert not truncated


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_observation(env: gym.Env, monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    env.reset(seed=1)
    # Test health classification without putting NaNs into the physics engine.
    monkeypatch.setattr(env.unwrapped, "do_simulation", lambda *args: None)
    monkeypatch.setattr(env.unwrapped, "_get_obs", lambda: np.array([value, 0, 0, 0]))
    _, reward, terminated, truncated, _ = env.unwrapped.step(np.zeros(1))
    assert reward == 0 and terminated and not truncated


def test_cart_position_is_not_failure(env: gym.Env) -> None:
    env.reset(seed=0)
    env.unwrapped.set_state(np.array([1.20, 0]), np.zeros(2))
    _, reward, terminated, _, _ = env.step(np.zeros(1))
    assert reward == 1 and not terminated


def test_time_limit() -> None:
    env = make_env(max_episode_steps=3)
    try:
        env.reset(seed=0)
        env.unwrapped.set_state(np.zeros(2), np.zeros(2))
        for step in range(1, 4):
            _, reward, terminated, truncated, _ = env.step(np.zeros(1))
            assert reward == 1 and not terminated
            assert truncated == (step == 3)
        env.reset(seed=0)
        assert not env.step(np.zeros(1))[3]
    finally:
        env.close()


def test_scene_path_is_independent_of_shell_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    env = make_env()
    try:
        assert env.unwrapped.model.body("cart").mass == 10
    finally:
        env.close()
