# Orientation reading list

These four resources are the complete pre-meeting list. Together they provide motivation, core vocabulary, and a first look at the APIs we will use. No installation or coding is expected.

Last reviewed: **2026-09-17**

## 1. Learning Agile Soccer Skills for a Bipedal Robot

- **Why:** Shows the long-term motivation for studying learned humanoid control.
- **Watch:** [The full video](https://www.youtube.com/watch?v=chMwFy6kXhs).
- **Take away:** Impressive robot behavior depends on simulation, training choices, and successful transfer—not RL alone.

## 2. Spinning Up: Key Concepts in RL

- **Why:** Provides an approachable vocabulary for the agent–environment loop.
- **Read:** [What Can RL Do? and Key Concepts and Terminology](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).
- **Skip for now:** Detailed equations or neural-network examples that slow you down.
- **Take away:** Be able to explain agent, environment, observation, action, reward, return, policy, value function, and episode.

## 3. Gymnasium: Basic Usage

- **Why:** Gymnasium defines the environment interface used in our exercises.
- **Skim:** [Initializing Environments through Wrappers](https://gymnasium.farama.org/introduction/basic_usage/), focusing on `reset()`, `step()`, and episode endings.
- **Take away:** Recognize the observation–action loop and the difference between termination and truncation.

## 4. Stable-Baselines3: Getting Started

- **Why:** Stable-Baselines3 will provide the first algorithms we train instead of asking beginners to implement them from scratch.
- **Skim:** [The example workflow](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html); do not run it yet.
- **Take away:** Recognize the steps to create, train, save, load, and use a policy.

The [orientation checklist](../docs/ORIENTATION.md) turns these readings into discussion prompts. Later material will be added only when it becomes relevant to onboarding or a team project.
