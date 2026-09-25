from collections.abc import Iterator
from pathlib import Path

import gymnasium as gym
import pytest

from onboarding.env import make_env
from onboarding.train import train


@pytest.fixture
def env() -> Iterator[gym.Env]:
    environment = make_env()
    yield environment
    environment.close()


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("training")
    policy = train("smoke", smoke=True, output_dir=root)
    return root, policy
