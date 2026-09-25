"""Inspect the XML directly, before completing the environment."""

import argparse
from pathlib import Path

import mujoco

from onboarding.viewer import view_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "assets/scene.xml")
    parser.add_argument("--control", type=float, choices=[-0.1, 0.0, 0.1], default=0.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    print(f"qpos={model.nq}, qvel={model.nv}, actuators={model.nu}, timestep={model.opt.timestep}s")
    view_model(model, args.control, args.seconds)


if __name__ == "__main__":
    main()
