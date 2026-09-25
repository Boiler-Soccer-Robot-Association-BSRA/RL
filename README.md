# BSRA Reinforcement Learning

This is the learning and research hub for the Reinforcement Learning subteam of the [Boiler Soccer Robots Association](https://github.com/Fabricio-Giusti/BSRA-Documentation) Programming team. We study how agents can learn useful humanoid-soccer behaviors in simulation and, eventually, transfer them to a robot.

## Leadership

- **RL Lead:** Quinn Winkler (qwinkler@purdue.edu)
- **Programming Director:** Adhitya Vasudevan (adhiv2007@gmail.com, vasude29@purdue.edu)

## Start here

1. **[Complete the orientation](docs/ORIENTATION.md)** — allow roughly 60–90 minutes to learn the shared RL vocabulary. No installation, coding, or prior RL experience is required.
2. **[Work through CartPole onboarding](docs/ONBOARDING.md)** — build a MuJoCo model, connect it to a Gymnasium task, train a PPO policy, and evaluate it against random actions. You need basic Python functions, classes, and imports; the guide introduces the Git workflow. Budget about 8–12 focused hours over two weeks, with a third week for setup or debugging. This estimate still needs a beginner pilot.
3. **Complete your handoff** — fill in [docs/REPORT.md](docs/REPORT.md), review and merge your assignment PR within your own fork, and share the completed repository with the RL lead. Follow the [handoff instructions](docs/ONBOARDING.md#6-report-and-github-handoff), then move to a separate project repository when you receive a team assignment.

## How the onboarding works

This is an assignment starter: complete the `MEMBER TODO` blocks in the XML model, environment, training, and evaluation files. Tests for unfinished sections are expected to fail. Follow the guide in order and run each section's checks as you go; the setup diagnostic works before any TODOs are completed.

Work in **your own fork** and keep your completed solution there. The handoff includes your code, report, compact results, and a reviewed, merged fork PR. There is no required policy score: explain your results and support them with checks and evaluation evidence.

See the [setup guide](docs/ONBOARDING.md#1-setup-and-git-workflow) for installation and platform requirements, including WSL2 for Windows and the pending Apple Silicon runtime pilot.

## Repository guide

| Location | What you will find |
| --- | --- |
| [docs/ORIENTATION.md](docs/ORIENTATION.md) | First readings and vocabulary checklist |
| [docs/ONBOARDING.md](docs/ONBOARDING.md) | Assignment steps, setup, checks, and submission instructions |
| [docs/REPORT.md](docs/REPORT.md) | Template for explaining your task, experiment, and results |
| [resources/](resources/README.md) | Curated reading list and toolchain overview |
| [assets/](assets/) | Cart-pole mechanism and scene XML to complete |
| [onboarding/](onboarding/) | Python exercise code, including the member TODOs |
| [scripts/](scripts/) | Commands for setup checks, viewing, training, and evaluation |
| [test/](test/) | Checks for the model, environment, and training/evaluation pipeline |

## Toolchain

We use **MuJoCo** for physics, **Gymnasium** for the environment interface, **Stable-Baselines3** for RL algorithms, **PyTorch** as the learning backend, and **TensorBoard** for experiment tracking. **uv** manages the Python environment and dependencies. See [the toolchain overview](resources/toolchain.md) for each learning tool's role and follow the onboarding guide for installation.

## Contributing

Use a focused pull request to improve the learning material, fix a starter or check, or suggest a resource. A new resource should include why it helps, what to read, any prerequisites, and the date it was checked. Prefer improving or replacing an entry over growing an unranked link list.

This repository is licensed under the [MIT License](LICENSE).
