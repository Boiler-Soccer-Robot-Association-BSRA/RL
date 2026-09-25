import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from numpy.testing import assert_allclose

from onboarding.env import SCENE


def test_model_contract() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    assert (model.nq, model.nv, model.nu) == (2, 2, 1)
    cart, pole = model.geom("cart_geom"), model.geom("pole_geom")
    assert cart.type == mujoco.mjtGeom.mjGEOM_BOX
    assert pole.type == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert_allclose(cart.size, [0.15, 0.10, 0.08])
    assert_allclose(pole.size[:2], [0.04, 0.375])
    assert_allclose(model.body("cart").mass, 10)
    assert_allclose(model.body("pole").mass, 5)
    assert_allclose(model.body("cart").pos, [0, 0, 0.20])
    assert_allclose(model.body("pole").pos, [0, 0, 0.08])
    slider, hinge = model.joint("slider"), model.joint("hinge")
    assert slider.type == mujoco.mjtJoint.mjJNT_SLIDE
    assert hinge.type == mujoco.mjtJoint.mjJNT_HINGE
    assert bool(slider.limited) and bool(hinge.limited)
    assert_allclose(slider.axis, [1, 0, 0])
    assert_allclose(hinge.axis, [0, 1, 0])
    assert_allclose(slider.range, [-1.25, 1.25])
    assert_allclose(hinge.range, np.deg2rad([-90, 90]))
    assert_allclose(model.dof_damping, [1, 1])
    assert_allclose(model.qpos0, [0, 0])
    motor = model.actuator("cart_motor")
    assert motor.trnid[0] == slider.id
    assert motor.trntype == mujoco.mjtTrn.mjTRN_JOINT
    assert bool(motor.ctrllimited)
    assert_allclose(motor.ctrlrange, [-3, 3])
    assert_allclose(motor.gear, [100, 0, 0, 0, 0, 0])
    assert_allclose(motor.gainprm[0], 1)
    assert_allclose(motor.biasprm, 0)
    assert_allclose(model.opt.gravity, [0, 0, -9.81])
    assert_allclose(model.opt.timestep, 0.02)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    assert not model.geom_contype.any() and not model.geom_conaffinity.any()
    assert model.camera("side").id >= 0
    # The included mechanism owns bodies/actuation; scene owns presentation.
    scene = ET.parse(SCENE).getroot()
    assert scene.find("include").attrib["file"] == "cartpole.xml"
    mechanism = ET.parse(SCENE.with_name("cartpole.xml")).getroot()
    assert mechanism.find("worldbody/body[@name='cart']") is not None
    assert mechanism.find("actuator/motor[@name='cart_motor']") is not None


def test_motion() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    positions = []
    for control in [-0.1, 0.1]:
        data = mujoco.MjData(model)
        data.ctrl[:] = control
        for _ in range(10):
            mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert data.ncon == 0
        positions.append(data.qpos[0])
    assert positions[0] < 0 < positions[1]
    assert_allclose(positions[0], -positions[1], atol=1e-8)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # At zero hinge angle the capsule's long axis is vertical.
    pole_rotation = data.geom("pole_geom").xmat.reshape(3, 3)
    assert_allclose(np.abs(pole_rotation[:, 2]), [0, 0, 1], atol=1e-10)
