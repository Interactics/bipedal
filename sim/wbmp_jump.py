"""Simulate the planned jump:  python sim/wbmp_jump.py
Plan it first (once, about 2 min):  python sim/jump_plan.py  -> sim/jump_plan.npz

Tracks the whole-body plan (centroidal dynamics + full kinematics, see jump_plan.py) in MuJoCo:
position servos on the planned joint angles, plus the planned joint torques as feedforward
(sent as a Goal Position offset, as in zmp_walk).
"""
import sys

import mujoco
import mujoco.viewer
import numpy as np

from run_sim import OUT, ROOT, STALL_TORQUE, build, update_markers

PLAN = ROOT / "sim" / "jump_plan.npz"
CROUCH_TIME = 1.0  # s, zero pose -> first planned posture before the jump
SETTLE_TIME = 1.5  # s, standing after landing


def load_plan():
    if not PLAN.exists():
        sys.exit(f"No plan yet: run  python sim/jump_plan.py  first (writes {PLAN.relative_to(ROOT)}).")
    return dict(np.load(PLAN))


def make_jump(model, plan):
    """Control callback for the plan, and the total time in seconds."""
    actuators = [model.actuator(name).id for name in plan["joints"]]
    t, q, v, torque = plan["t"], plan["q"], plan["v"], plan["torque"]
    kp = model.actuator_gainprm[actuators, 0]
    kv = -model.actuator_biasprm[actuators, 2]
    commands = q + (torque + kv * v) / kp  # same Goal Position offset as zmp_walk
    commands[0] = q[0] + torque[0] / kp

    def controller(model, data):
        s = data.time - CROUCH_TIME
        if s < 0:  # crouch from the zero pose into the first planned posture
            x = data.time / CROUCH_TIME
            ctrl = commands[0] * x * x * (3 - 2 * x)
        elif s >= t[-1]:
            ctrl = commands[-1]
        else:  # joint targets: linear between knots; velocity and torque: held over each step
            k = np.searchsorted(t, s, side="right")
            f = (s - t[k - 1]) / (t[k] - t[k - 1])
            ctrl = commands[k] + (q[k - 1] - q[k]) * (1 - f)
        data.ctrl[actuators] = ctrl
        update_markers(model, data)

    return controller, CROUCH_TIME + t[-1] + SETTLE_TIME


def check_jump(model, plan, duration):
    """Runs the jump headless: leaves the ground, clears the floor, lands upright and stands."""
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    soles = [model.geom(f"{side}_sole").id for side in "lr"]
    airborne, longest, clearance, lowest, peak, saturated, steps = 0.0, 0.0, 0.0, 1.0, 0.0, 0, 0
    while data.time < duration:
        mujoco.mj_step(model, data)
        lowest = min(lowest, data.site("imu").xmat[8])
        touching = any(floor in c.geom for c in data.contact[:data.ncon])
        airborne = 0.0 if touching else airborne + model.opt.timestep
        longest = max(longest, airborne)
        if not touching:
            bottoms = [data.geom_xpos[g][2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g] for g in soles]
            clearance = max(clearance, min(bottoms))
        peak = max(peak, np.abs(data.actuator_force).max())
        saturated += np.any(np.abs(data.actuator_force) > STALL_TORQUE - 1e-6)
        steps += 1
    flight = np.flatnonzero(~plan["stance"])
    planned = plan["t"][flight[-1]] - plan["t"][flight[0] - 1]
    print(f"jump: airborne {longest * 1e3:.0f} ms (planned {planned * 1e3:.0f}) | sole clearance "
          f"{clearance * 1e3:.1f} mm (planned {plan['clearance'] * 1e3:.0f}) | min upright {lowest:.3f} | "
          f"peak servo torque {peak:.2f} N·m, at the limit {saturated / steps * 100:.1f}% of the time | "
          f"standing at the end: {data.ncon >= 2 and data.site('imu').xmat[8] > 0.99}")
    assert lowest > 0.9, f"tipped over: upright {lowest:.3f}"
    assert longest > 0.5 * planned, f"airborne only {longest * 1e3:.0f} ms"
    assert data.site("imu").xmat[8] > 0.99 and data.ncon >= 2, "did not land and stand"


if __name__ == "__main__":
    plan = load_plan()
    build()
    op3 = mujoco.MjModel.from_xml_path(str(OUT))
    jump, seconds = make_jump(op3, plan)
    mujoco.set_mjcb_control(jump)  # after loading: the compiler would trigger it too
    try:
        check_jump(op3, plan, seconds)
    except AssertionError as failure:  # still open the viewer, to see what went wrong
        print(f"CHECK FAILED: {failure}")
    mujoco.viewer.launch(op3)
