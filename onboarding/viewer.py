"""Provided viewer; independent of the unfinished Gymnasium exercise."""

import threading
import time

import mujoco
import mujoco.viewer


def view_model(model: mujoco.MjModel, control: float = 0.0, seconds: float = 10.0) -> None:
    """Start each bounded trial from the upright zero state. Close with Escape."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # MuJoCo 3.6 closes its window asynchronously. Wait for the viewer's
    # threads before Python exits and GLFW tears down the display.
    existing_threads = set(threading.enumerate())
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.5]
                viewer.cam.distance = 3.5
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -15
            end = time.monotonic() + seconds
            while viewer.is_running() and time.monotonic() < end:
                start = time.monotonic()
                if model.nu:
                    data.ctrl[:] = control
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - start)))
    finally:
        for thread in set(threading.enumerate()) - existing_threads:
            thread.join()
