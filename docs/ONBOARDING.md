# CartPole onboarding

CartPole is a balancing task: a pole is attached by a free hinge to a cart that
moves left and right. You control the cart's motor, and the pole moves in response
to gravity and the cart's motion. The objective is to keep the pole upright for
as long as possible within an episode.

You will build the physical model in MuJoCo, connect it to Gymnasium's supplied
balancing task, and train a PPO policy (a learned controller). The policy receives
the cart and pole positions and velocities, then chooses a motor control. It
learns from rewards for keeping the pole upright. You will evaluate the saved
policy against random actions to see what it learned.

By the end, you should be able to explain how simulation, observations, actions,
rewards, and training results connect across the toolchain, and support your
conclusions with a repeatable experiment. Finish with a short report and a pull
request in your own fork.

Start after the [orientation](ORIENTATION.md). You need basic Python functions,
classes, and imports; Git and RL experience are not required. Budget about
**6–8 focused hours over two weeks**.

**Before starting:** read the [toolchain overview](../resources/toolchain.md).
Be able to explain how MuJoCo, Gymnasium, Stable-Baselines3 (SB3), and TensorBoard
contribute to this exercise, and answer the toolchain question in
[REPORT.md](REPORT.md#setup). As you work through each stage, use the report
questions to trace which tool uses the inputs you write, where its outputs come
from, and what uses those outputs next.

| Section        | Outcome                                      |
| -------------- | -------------------------------------------- |
| 1. Setup       | Working fork, branch, and Python environment |
| 2. Model       | Two composed XML files that load and move    |
| 3. Environment | A checked, documented RL task                |
| 4. Training    | Saved policy and TensorBoard logs            |
| 5. Evaluation  | Baseline comparison and one recorded episode |
| 6. Handoff     | Report and reviewed, merged fork PR          |

**Member files:** complete `MEMBER TODO` blocks in `assets/cartpole.xml`,
`assets/scene.xml`, `onboarding/env.py`, `onboarding/train.py`, and
`onboarding/evaluate.py`; fill in [REPORT.md](REPORT.md).
`scripts/` contains commands you run; `onboarding/` contains the Python you edit.
After cloning, run assignment commands from the repository root. Command-line
paths in this guide are relative to that root, including `docs/REPORT.md`.
The supplied `test/` checks describe the assignment contract. Tests for unfinished
sections are expected to fail; the setup diagnostic works immediately.

## 1. Setup and Git workflow

**Files to edit:** [REPORT.md](REPORT.md) (Setup).

Install
[Git](https://docs.github.com/en/get-started/git-basics/set-up-git) and
[uv](https://docs.astral.sh/uv/getting-started/installation/) in the environment
where you will run the exercise. New to terminals? Read the
[Ubuntu command-line introduction](https://ubuntu.com/tutorials/command-line-for-beginners)
through working with directories. `pwd` shows your directory; `cd` changes it.

Throughout this onboarding, look up any Git or uv command or flag you do not
recognize before running it. Use the documentation or built-in help to understand
what it does and how it affects your repository or Python environment.

| Platform | Setup |
| --- | --- |
| Windows | Use **WSL2 with Ubuntu**. Follow [Microsoft's setup guide](https://learn.microsoft.com/en-us/windows/wsl/setup/environment), then check `wsl -l -v` in PowerShell. Run all assignment commands inside Ubuntu. Keep the clone under a Linux path such as `~/dev/BSRA-RL`. Check [WSLg](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) before the viewer step. |
| Linux x86-64 | Use the native terminal and CPU dependencies supplied here. |
| Apple Silicon macOS | Use native ARM64 Python through uv. CPU training is configured. Use the `mjpython` viewer commands below. This platform still needs a runtime pilot of this lockfile. |
| Intel Mac / other architectures | This lockfile does not target these systems. Arrange access to a Linux x86-64 machine with a maintainer before starting. |


### Fork, clone, and install

Follow [GitHub's fork guide](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo)
to fork [grmpn/BSRA-RL](https://github.com/grmpn/BSRA-RL) into your account.
A **fork** is your GitHub repository; a **clone** is its local copy.
Configure your [Git name](https://docs.github.com/en/get-started/git-basics/setting-your-username-in-git),
[email](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address),
and authentication using the Git setup guide above. Replace `YOUR-USERNAME`:

```bash
git clone https://github.com/YOUR-USERNAME/BSRA-RL.git
cd BSRA-RL
git remote -v
git switch -c onboarding/cartpole
git status
uv sync
```

A **branch** is a named line of development within a repository.
`git switch -c onboarding/cartpole` creates a branch named `onboarding/cartpole`
from your current commit and switches to it (`-c` means create). Your new commits
go on this branch, so you can work on the assignment while `main` stays unchanged
until you merge your work into it.

### How you know you've finished

Run the setup diagnostic. It checks package imports, CPU calculations, the video
encoder, and a small physics simulation that works before you complete any TODOs.

```bash
uv run python scripts/check_setup.py
```

Check rendering separately. `--viewer` opens an interactive window; `--rgb`
checks that MuJoCo can render an image for video recording.

```bash
uv run python scripts/check_setup.py --viewer
uv run python scripts/check_setup.py --rgb
```

On macOS, replace the first command with
`uv run mjpython scripts/check_setup.py --viewer`; ordinary training uses Python.
MuJoCo requires this launcher for its
[passive viewer](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

**Finished when:** all diagnostic checks pass; a falling sphere appears for three
seconds and the window closes. The fixture is independent of your XML.
If a viewer fails, resolve the display/WSLg setup before working on the model.

Add your OS, setup results, and any fixes to `docs/REPORT.md`, then practice:

```bash
git status
git diff
git add docs/REPORT.md
git commit -m "Record onboarding setup"
git push -u origin onboarding/cartpole
```

A commit saves a local revision; a push uploads it to your fork. Confirm the
commit appears on GitHub. Repeat this cycle after each section, staging the files
you edited. The final workflow is **branch → push → PR into your fork → review → merge**.

## 2. XML model and scene creation

**Files to edit:** [assets/cartpole.xml](../assets/cartpole.xml),
[assets/scene.xml](../assets/scene.xml), and [REPORT.md](REPORT.md) (Model and task).

**Before starting:** setup passes. Read MuJoCo's
[modeling introduction](https://mujoco.readthedocs.io/en/stable/modeling.html)
on bodies, joints, geoms, and actuators; use the
[XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) for attributes.

Sketch the cart, pole, slide axis, hinge axis, and motor. Then fill the two XML
files. `scene.xml` is the top-level `<mujoco>` model; its `<include>` inserts the
contents of `cartpole.xml`'s `<mujocoinclude>` root. Put the mechanism's
`<worldbody>` and `<actuator>` sections in that included file. The scene owns
global options, scenery, lighting, and a camera.

| Element / required name | Specification |
| --- | --- |
| Body `cart`, geom `cart_geom` | Body at `0 0 0.20`; box full dimensions `0.30 × 0.20 × 0.16 m`, mass `10 kg` |
| Body `pole`, geom `pole_geom` | Child of cart at `0 0 0.08`; capsule from `0 0 0` to `0 0 0.75`, radius `0.04 m`, mass `5 kg` |
| Joint `slider` | Slide along `1 0 0`, limited to `[-1.25, 1.25] m`, damping `1` |
| Joint `hinge` | Hinge along `0 1 0`, limited to `[-90, 90]` degrees, damping `1` |
| Motor `cart_motor` | Acts on `slider`, limited control `[-3, 3]`, gear `100` |
| Scene | Degree compiler units; gravity `0 0 -9.81`, RK4 integrator, timestep `0.02 s` |
| Scenery | Floor, rail, light, and camera named `side`; disable collisions for **all** geoms with `contype="0" conaffinity="0"` |

Box `size` values are **half-extents**. Capsule endpoint separation excludes its
rounded caps. With these placements, zero joint positions put the pole upright;
the hinge is unactuated. XML angles use degrees here, while runtime joint angles
use radians. Colors, camera placement, and scenery dimensions are your choice.

### How you know you've finished

Run the model tests. They check XML composition, body dimensions and masses,
joint and motor settings, simulation options, and motion under opposite controls.

```bash
uv run python -m pytest test/test_model.py
```

Inspect the model in the viewer, first with no control and then with opposite
controls. This checks that the scene looks right and the axes behave as intended.

```bash
uv run python scripts/view_model.py
uv run python scripts/view_model.py --control 0.1 --seconds 1
uv run python scripts/view_model.py --control -0.1 --seconds 1
```

On macOS use `uv run mjpython scripts/view_model.py ...`.
Each viewer trial starts from zero. Rotate the camera to check the axes.

**Finished when:** tests pass, opposite controls move the cart in opposite directions,
and the pole rotates about the hinge. It need not balance without a policy.

Save a small screenshot in `results/model.png`, link it in the report, and explain
which XML file owns the mechanism. Use `scripts/view_model.py` to identify which
tool loads the XML and produces the simulated motion, then answer the report's
model questions. Commit the two XML files and report evidence.

## 3. Environment setup and inspection

**Files to edit:** [onboarding/env.py](../onboarding/env.py) and
[REPORT.md](REPORT.md#environment-stage-3) (Environment).

**Before starting:** model tests pass. Review
[Gymnasium basic usage](https://gymnasium.farama.org/introduction/basic_usage/) and
the [InvertedPendulum task](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/).

MuJoCo computes motion from your XML model and the applied controls. Gymnasium
connects that simulation to RL: it accepts actions, returns observations and
rewards, and starts new episodes with `reset()`. Your job is to connect the model
and understand these interactions. The observation and reward functions are
already supplied.

Complete `make_env()` in `onboarding/env.py`, using the supplied path and task
constants. Pass the rendering mode and episode limit through to `gym.make()`.
The latter applies the `TimeLimit` wrapper.

### Understand the supplied task

Read `_get_obs()`, `reset_model()`, and `step()` in the
[pinned upstream implementation](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.3/gymnasium/envs/mujoco/inverted_pendulum_v5.py).
For the physics, follow `do_simulation()` to `_step_mujoco_simulation()` in the
[MuJoCo environment base class](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.3/gymnasium/envs/mujoco/mujoco_env.py)
to see how actions reach the simulator. In this code, `self.data` is MuJoCo's
simulation data: `qpos` holds joint positions, `qvel` holds joint velocities, and
`ctrl` holds actuator controls.

Use that code, your XML, and the constants in `onboarding/env.py` to answer the
five questions in [REPORT.md](REPORT.md#environment-stage-3): **observations,
actions, physics, reset, and reward**. Keep each answer short and name the code
or setting that supports it.

During a rollout, reset after either `terminated` (task failure) or `truncated`
(the `TimeLimit` episode limit). XML joint travel limits constrain the motion;
they do not themselves end an episode.

### How you know you've finished

Run the environment tests. They check the Gymnasium API, observation and action
spaces, timing, seeded resets, rewards, termination, and the time limit. They also
check that loading the scene does not depend on your terminal's working directory.

```bash
uv run python -m pytest test/test_env.py
```

Run random-action episodes to check the environment before completing the policy
evaluation TODOs. `--trace` prints observations, actions, rewards, and ending
flags; `--video` records a rollout so you can inspect the custom scene.

```bash
uv run python scripts/evaluate.py --random-only --run-name random-debug --episodes 3 --trace
uv run python scripts/evaluate.py --random-only --run-name random-debug --episodes 3 --video
```

Open `videos/random-debug/episode-10000.mp4`.

**Finished when:** API and behavior tests pass; random episodes end and reset; the custom
scene is visible. SB3 warns that the action range is not normalized to `[-1, 1]`.
Keep `[-3, 3]` for this task and record that warning.

## 4. Training and experiment tracking

**Files to edit:** [onboarding/train.py](../onboarding/train.py) and
[REPORT.md](REPORT.md) (Training).

**Before starting:** environment checks pass. Read the introduction and example in
[SB3's PPO guide](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
PPO collects experience and updates a policy; SB3 handles the PyTorch network and
optimizer. You supply the environment and experiment settings.

Complete the three training TODOs in `onboarding/train.py`: construct PPO with
`MlpPolicy`, learn, and save. Use the provided monitored environment, seed, CPU
device, TensorBoard path, and `settings`. `Monitor` supplies episode return/length
statistics. The default settings are explicit in `PPO_SETTINGS`; keep them for
your first run. The supplied code writes versions, task hashes, source revision,
requested/actual steps, runtime, and output paths to `results/<run>/run.json`.

The smoke run is 256 steps and checks the pipeline, not learning. The full
100,000-step budget is a starting point, not a promised score. PPO collects full
rollouts, so with 2,048 steps per rollout it actually collects **100,352** steps.
Run names cannot be reused for training; choose a new name to preserve earlier work.

### How you know you've finished

Run the training test. It checks that a short PPO run saves a policy, records
training metadata and TensorBoard episode statistics, and rejects a reused run
name.

```bash
uv run python -m pytest test/test_pipeline.py -k training_smoke
```

Check the training script with a smoke run. It writes the policy, metadata, and
logs under the `smoke` run name so you can inspect the outputs.

```bash
uv run python scripts/train.py --run-name smoke --smoke
```

Commit your code before the full run so its metadata points to a reproducible
revision, then train the policy you will evaluate in Stage 5:

```bash
uv run python scripts/train.py --run-name first-run --seed 0 --steps 100000
```

Read SB3's [TensorBoard integration guide](https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/tensorboard.html)
and [metric definitions](https://stable-baselines3.readthedocs.io/en/v2.7.1/common/logger.html#rollout).
Use the report questions to distinguish the environment's rewards, `Monitor`'s
episode statistics, SB3's logging, and TensorBoard's display. Open TensorBoard
to inspect the recorded learning statistics:

```bash
uv run tensorboard --logdir runs
```

Open TensorBoard's printed local URL. Look at `rollout/ep_rew_mean` and
`rollout/ep_len_mean`. The horizontal axis is environment steps, not episodes;
each statistic averages up to the most recent 100 training episodes. Set UI
smoothing to zero for the report and save a small curve image at
`results/first-run/learning-curve.png`. Explain improvement, variability, or a plateau.
TensorBoard's "TensorFlow installation not found" notice is expected; these
scalar plots work without TensorFlow.

**Finished when:** the training test passes, `models/first-run/policy.zip` exists,
and TensorBoard has real episode statistics. Record elapsed time and settings.

## 5. Evaluation and interpretation

**Files to edit:** [onboarding/evaluate.py](../onboarding/evaluate.py) and
[REPORT.md](REPORT.md) (Evaluation).

**Before starting:** a saved policy and matching `run.json` exist. Complete the
two evaluation TODOs in `onboarding/evaluate.py`: load the policy on CPU and
predict deterministic actions. Evaluation runs in a fresh process and never learns.

Use the same run name as training. The evaluation script verifies the policy and task
against the training record, then compares PPO and random actions on reset seeds
**10000–10019**. Reserve these seeds for evaluation. Random action sampling has
its own seed (`reset seed + 20000`). Both agents get the same starting-state seeds.

`results/first-run/episodes.csv` records each return, length, seed, and ending flags.
`summary.json` contains means, population standard deviations (`ddof=0`), and the
fraction reaching the time limit. Repeating evaluation replaces these outputs;
it does not append episodes. Random-only debugging uses a separate run name.

`--video` records the **actual PPO episode with seed 10000**, selected in advance.
That episode is counted once in the results. The supplied encoder streams 640 × 480
frames into `videos/first-run/episode-10000.mp4` at 25 fps, derived from the control
interval. Other episodes run without rendering. Without `--video`, no encoder
starts and the new summary records no video.

Compare the baseline and policy in the report. A training curve and a good-looking
video do not replace multi-episode evaluation. These episodes measure varied starts
for one trained policy, not variation across training seeds. There is **no required
return threshold**.
If learning is weak, use checks and logs to support one diagnosis; do not select
a better-looking video seed or hide failed episodes.

### How you know you've finished

Run the pipeline tests. They check smoke training, saved-policy loading,
deterministic actions, evaluation statistics and output files, repeatable results,
and rejection of a policy that does not match its training record.

```bash
uv run python -m pytest test/test_pipeline.py
```

Evaluate your saved policy against the random baseline and record the predetermined
rollout. This also checks video recording, which the automated tests do not cover.

```bash
uv run python scripts/evaluate.py --policy models/first-run/policy.zip --run-name first-run --video
```

**Finished when:** tests pass, the video plays, and your report compares the
baseline and PPO results using the generated CSV and JSON files. Explain how
`onboarding/evaluate.py` uses the saved policy and environment to produce those
results and the video.

## 6. Report and GitHub handoff

**Files to edit:** [REPORT.md](REPORT.md) (Reproduce and review, Feedback, and any
remaining TODOs).

Finish `docs/REPORT.md`: setup, model/task explanation, reproduction commands, learning
curve, baseline comparison, rollout observations, one limitation, and onboarding
feedback. Explain what carries over to humanoid control and what this exercise
leaves out.

### How you know you've finished

Run the full test suite before publishing your handoff. It checks the model,
environment, smoke training, saved-policy loading, and evaluation outputs together.
Use your earlier manual checks as evidence that the viewer and video also work.

```bash
uv run python -m pytest
```

After the tests pass, review and commit your report, code, and compact results:

```bash
git status
git diff
git add assets onboarding docs/REPORT.md results
git diff --staged
git commit -m "Complete CartPole onboarding report"
git push
```

Commit compact JSON/CSV results and images. `.venv`, `models`, `runs`, and `videos`
are ignored. Keep the saved policy, TensorBoard logs, and rollout video locally.
Share them directly with the RL lead if requested; uploading them or creating a
GitHub release is not required for completion.

Follow [GitHub's PR instructions](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request).
Inspect the repository selector: **base repository `YOUR-USERNAME/BSRA-RL`, base
branch `main`, compare branch `onboarding/cartpole`**. Member solutions stay in
member forks. Example description:

> Completes the custom CartPole model and PPO experiment. Report and results:
> docs/REPORT.md. Checks: setup, model/environment tests, smoke train, saved-policy
> evaluation, and playable seed-10000 video. Limitation: one training seed.

Review the diff, record the review in the PR, then merge it through
GitHub. Your completed work and finalized report should now be on your fork's
`main` branch.

```bash
git switch main
git pull --ff-only origin main
```

**Finished when:** a reviewer can clone your fork, run `uv sync` and the tests,
and understand your experiment from the report and committed results. The report,
compact results and images, and reviewed, merged fork PR must all be present.

## Submit your onboarding

Post a link to your completed forked repository in the **software channel on
Discord** and ping **@Quinn**. Make sure the repository includes your completed
code and finalized `docs/REPORT.md` on `main`.

You can also send Quinn the repository link by **Discord DM**, or **present your
work to Quinn during a meeting** and share the link then.
