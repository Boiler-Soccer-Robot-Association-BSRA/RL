import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from onboarding.evaluate import choose_action, evaluate, load_policy, run_episode
from onboarding.train import train


def test_training_smoke(trained_run: tuple[Path, Path]) -> None:
    root, policy = trained_run
    assert policy.is_file()
    metadata = json.loads((root / "results/smoke/run.json").read_text())
    assert metadata["actual_steps"] == metadata["requested_steps"] == 256
    assert metadata["device"] == "cpu" and metadata["seed"] == 0
    assert metadata["task"]["max_episode_steps"] == 1000
    assert metadata["versions"]["gymnasium"]
    assert metadata["lockfile_sha256"] and metadata["policy_sha256"]
    event = next((root / "runs").rglob("events.out.tfevents.*"))
    log = EventAccumulator(str(event)).Reload()
    assert log.Scalars("rollout/ep_rew_mean")
    assert log.Scalars("rollout/ep_len_mean")
    with pytest.raises(FileExistsError):
        train("smoke", smoke=True, output_dir=root)


def test_policy_round_trip(trained_run: tuple[Path, Path], env: gym.Env) -> None:
    _, policy_path = trained_run
    policy = load_policy(policy_path)
    observation, _ = env.reset(seed=10_000)
    action = choose_action(env, observation, policy)
    assert env.action_space.contains(action)
    assert np.array_equal(action, choose_action(env, observation, policy))
    for seed in [10_000, 10_001]:
        row = run_episode(env, seed, policy)
        assert np.isfinite(row["return"]) and 1 <= row["length"] <= 1000
        assert row["terminated"] or row["truncated"]


def test_evaluation_outputs(trained_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    root, policy = trained_run
    # Ordinary evaluation must neither render nor start a video encoder.
    import imageio_ffmpeg
    from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Core evaluation attempted rendering or video encoding")
    monkeypatch.setattr(imageio_ffmpeg, "write_frames", forbidden)
    monkeypatch.setattr(InvertedPendulumEnv, "render", forbidden)
    seeds = (10_000, 10_001)
    summary = evaluate("smoke", policy, seeds=seeds, output_dir=root)
    assert json.loads((root / "results/smoke/summary.json").read_text()) == summary
    csv_path = root / "results/smoke/episodes.csv"
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and summary["video"] is None
    for label in ["random", "ppo"]:
        group = [row for row in rows if row["policy"] == label]
        assert [int(row["seed"]) for row in group] == list(seeds)
        stats = summary["results"][label]
        assert stats["episodes"] == 2
        for field in ["return", "length"]:
            values = [float(row[field]) for row in group]
            assert stats[f"{field}_mean"] == pytest.approx(np.mean(values))
            assert stats[f"{field}_std"] == pytest.approx(np.std(values))
        assert stats["time_limit_fraction"] == np.mean([row["truncated"] == "True" for row in group])
        assert all(row["terminated"] == "True" or row["truncated"] == "True" for row in group)
    original = csv_path.read_text()
    evaluate("smoke", policy, seeds=seeds, output_dir=root)
    assert csv_path.read_text() == original  # Replaces outputs; no duplicate rows.
    baseline = evaluate("random-debug", seeds=seeds, output_dir=root)
    assert baseline["results"]["random"] == summary["results"]["random"]
    with pytest.raises(ValueError, match="separate run name"):
        evaluate("smoke", seeds=seeds, output_dir=root)
    with pytest.raises(ValueError, match="distinct"):
        evaluate("invalid", seeds=(1, 1), output_dir=root)


def test_evaluation_rejects_mismatched_policy(trained_run: tuple[Path, Path], tmp_path: Path) -> None:
    root, _ = trained_run
    wrong_policy = tmp_path / "wrong.zip"
    wrong_policy.write_bytes(b"not the recorded policy")
    with pytest.raises(ValueError, match="differs"):
        evaluate("smoke", wrong_policy, seeds=(10_000,), output_dir=root)
