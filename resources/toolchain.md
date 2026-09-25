# Toolchain overview

Our tools form one learning pipeline:

`MuJoCo simulation → Gymnasium environment → Stable-Baselines3 agent → TensorBoard results`

**MuJoCo** simulates bodies, joints, contacts, sensors, and actuators. Its MJCF/XML files describe the physical model and scene.

**Gymnasium** turns a task into a consistent RL interface. It defines observations, actions, rewards, resets, termination, and truncation around a simulator such as MuJoCo.

**Stable-Baselines3 (SB3)** supplies tested implementations of algorithms such as PPO. It handles the training loop, saving and loading policies, and common evaluation utilities.

**PyTorch** provides tensors, neural networks, gradients, and optimization underneath SB3. Our first assignment uses it through SB3; direct PyTorch model code is not required.

**TensorBoard** displays training metrics over time so we can inspect and compare named runs. A graph is evidence to examine, not proof that a policy works.

In short: MuJoCo supplies the simulated world, Gymnasium defines the task, SB3 learns a policy with PyTorch, and TensorBoard helps us inspect the experiment.

Follow the [onboarding setup guide](../docs/ONBOARDING.md#1-setup-and-git-workflow) to install the supplied environment and check that the tools work together.
