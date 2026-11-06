"""ZMP walking along a circle:  python sim/zmp_walk.py

0. Footsteps along the path (a circle, drawn as a yellow line on the floor), turning with it.
1. Footsteps -> ZMP reference that stays inside the support foot.
2. ZMP reference -> CoM path: preview control on the cart-table (LIPM) model, Kajita 2003.
3. CoM path + foot paths -> joint angles: numerical whole-body IK on the MuJoCo model.
4. Joint angles -> position actuators, open loop (no balance feedback yet).

Viewer markers: red = CoM, green = measured ZMP, blue = reference ZMP.
Reset in the viewer (Backspace) replays the walk.
"""
import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are

from run_sim import OUT, SPAWN_CLEARANCE, build, marker, measured_zmp, update_markers

# Gait knobs.
DT = 0.01  # s, planner sample time
STEP_TIME = 0.5  # s per step
DOUBLE_SUPPORT = 0.2  # share of each step with both feet down, while the ZMP moves across
RADIUS = 0.25  # m, circle the feet midpoint walks along, turning left
N_STEPS = 52  # one full lap
STEP_LENGTH = 2 * np.pi * RADIUS / N_STEPS  # m along the path, ≈ 30 mm
STEP_HEIGHT = 0.02  # m, swing foot lift
COM_HEIGHT = 0.15  # m, CoM above the floor while walking (knees bent; standing is 0.165)
ZMP_INSET = 0.01  # m, ZMP target this far inside the sole center, toward the body
PREVIEW_TIME = 1.6  # s
CROUCH_TIME = 1.0  # s, zero pose -> walking posture
HOLD_TIME = 1.0  # s, standing still before the first and after the last step
FILTER_PASSES = 2  # dynamics filter passes; full-robot ZMP error 12 -> 3 mm with 2
FEEDFORWARD = True  # add the planned joint torques to the servo commands
G = 9.81


def smooth(s):
    return s * s * (3 - 2 * s)


def path_pose(s):
    """Walking path at arc length s: (x, y, heading) of the feet midpoint, relative to its start.

    A circle of RADIUS, starting along +x and turning left (counter-clockwise from above).
    """
    heading = s / RADIUS
    return np.array([RADIUS * np.sin(heading), RADIUS * (1 - np.cos(heading)), heading])


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def plan(sole0, com0):
    """Footsteps -> per-sample ZMP reference (n, 2) and sole paths {side: (n, 4)}: x, y, z, yaw.

    The reference starts under the CoM and moves to the middle of the feet, so the
    preview controller starts without a jump.
    """
    start = (sole0["l"] + sole0["r"]) / 2

    def foot_at(side, s):
        x, y, heading = path_pose(s)
        offset = sole0[side][1] - start[1]  # + for the left foot
        return np.array([start[0] + x - np.sin(heading) * offset,
                         start[1] + y + np.cos(heading) * offset, sole0[side][2], heading])

    arc = {"l": 0.0, "r": 0.0}
    feet = {side: foot_at(side, 0.0) for side in "lr"}
    zmp, paths = [], {"l": [], "r": []}
    zmp_now = com0.copy()

    def segment(duration, zmp_to, swing=None, swing_to=None):
        nonlocal zmp_now
        n = round(duration / DT)
        for k in range(1, n + 1):
            s = k / n
            zmp.append(zmp_now + (zmp_to - zmp_now) * s)
            for side in "lr":
                p = feet[side].copy()
                if side == swing:
                    p[[0, 1, 3]] += (swing_to[[0, 1, 3]] - p[[0, 1, 3]]) * smooth(s)
                    p[2] += STEP_HEIGHT * np.sin(np.pi * s) ** 2
                paths[side].append(p)
        if swing:
            feet[swing] = swing_to
        zmp_now = zmp_to

    def on_foot(side):
        x, y, _, yaw = feet[side]
        inward = (1 if side == "l" else -1) * ZMP_INSET  # toward the body, across the foot
        return np.array([x + np.sin(yaw) * inward, y - np.cos(yaw) * inward])

    segment(HOLD_TIME, (feet["l"][:2] + feet["r"][:2]) / 2)
    for i in range(N_STEPS + 1):  # the last step only brings the feet together
        swing, support = ("l", "r") if i % 2 == 0 else ("r", "l")
        arc[swing] = arc[support] + (STEP_LENGTH if i < N_STEPS else 0.0)
        target = foot_at(swing, arc[swing])
        segment(STEP_TIME * DOUBLE_SUPPORT, on_foot(support))
        segment(STEP_TIME * (1 - DOUBLE_SUPPORT), on_foot(support), swing, target)
    segment(STEP_TIME * DOUBLE_SUPPORT, (feet["l"][:2] + feet["r"][:2]) / 2)
    segment(HOLD_TIME + PREVIEW_TIME, zmp_now)  # the preview needs to see a still future
    return np.array(zmp), {side: np.array(p) for side, p in paths.items()}


def preview_com(zmp_ref, com0):
    """Cart-table preview control (Kajita 2003): CoM xy (n, 2) and the model's ZMP (n, 2)."""
    A = np.array([[1, DT, DT ** 2 / 2], [0, 1, DT], [0, 0, 1]])
    B = np.array([[DT ** 3 / 6], [DT ** 2 / 2], [DT]])
    C = np.array([[1, 0, -COM_HEIGHT / G]])
    A1 = np.block([[np.ones((1, 1)), C @ A], [np.zeros((3, 1)), A]])
    B1 = np.vstack([C @ B, B])
    Q, R = np.diag([1.0, 0, 0, 0]), np.array([[1e-6]])
    P = solve_discrete_are(A1, B1, Q, R)
    S = R + B1.T @ P @ B1
    K = np.linalg.solve(S, B1.T @ P @ A1)
    gi, gx = K[0, 0], K[0, 1:]
    Ac = A1 - B1 @ K
    X = -Ac.T @ P @ np.array([[1.0], [0], [0], [0]])
    n_preview = round(PREVIEW_TIME / DT)
    gd = np.zeros(n_preview)
    gd[0] = -gi
    for j in range(1, n_preview):
        gd[j] = np.linalg.solve(S, B1.T @ X)[0, 0]
        X = Ac.T @ X

    x = np.zeros((3, 2))  # rows: position, velocity, acceleration; columns: x, y
    x[0] = com0
    error, com, zmp = np.zeros(2), [], []
    for k in range(len(zmp_ref) - n_preview):
        error += (C @ x)[0] - zmp_ref[k]
        u = -gi * error - gx @ x - gd @ zmp_ref[k:k + n_preview]
        x = A @ x + B @ u[None]
        com.append(x[0].copy())
        zmp.append((C @ x)[0])
    return np.array(com), np.array(zmp)


def solve_ik(model, com, feet):
    """qpos per sample (n, nq) so the whole-body CoM and both soles follow their paths.

    Damped least squares over the root pose and the 12 joints. Soles stay flat and turn with
    their yaw; the pelvis stays upright and faces the average of the two feet.
    """
    data = mujoco.MjData(model)
    root = model.body("root").id
    soles = {side: model.geom(f"{side}_sole").id for side in "lr"}
    mujoco.mj_kinematics(model, data)
    flat = {side: data.geom_xmat[g].reshape(3, 3).copy() for side, g in soles.items()}
    upright = data.xmat[root].reshape(3, 3).copy()
    # Seed with bent knees so the IK can't bend them backwards (signs: CLAUDE.md table).
    for name, angle in {"l_hip_pitch": -0.2, "l_knee": 0.4, "l_ankle_pitch": -0.2,
                        "r_hip_pitch": 0.2, "r_knee": -0.4, "r_ankle_pitch": 0.2}.items():
        data.qpos[model.joint(name).qposadr[0]] = angle

    def turn_error(R, R_target):  # rotation vector that takes R to R_target (small angles)
        return 0.5 * np.cross(R.T, R_target.T).sum(0)

    jac_com, jac_pos, jac_rot = (np.zeros((3, model.nv)) for _ in range(3))
    out, worst = [], 0.0
    for k in range(len(com)):
        heading = (feet["l"][k, 3] + feet["r"][k, 3]) / 2
        for _ in range(50):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_jacSubtreeCom(model, data, jac_com, root)
            errors = [np.array([*com[k], COM_HEIGHT]) - data.subtree_com[root]]
            jacs = [jac_com.copy()]
            mujoco.mj_jacBody(model, data, jac_pos, jac_rot, root)
            errors.append(turn_error(data.xmat[root].reshape(3, 3), rot_z(heading) @ upright))
            jacs.append(jac_rot.copy())
            for side, g in soles.items():
                mujoco.mj_jacGeom(model, data, jac_pos, jac_rot, g)
                errors += [feet[side][k, :3] - data.geom_xpos[g],
                           turn_error(data.geom_xmat[g].reshape(3, 3), rot_z(feet[side][k, 3]) @ flat[side])]
                jacs += [jac_pos.copy(), jac_rot.copy()]
            e = np.concatenate(errors)
            if np.abs(e).max() < 1e-7:
                break
            J = np.vstack(jacs)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(len(e)), e)
            mujoco.mj_integratePos(model, data.qpos, dq, 1.0)
        worst = max(worst, np.abs(e).max())
        out.append(data.qpos.copy())
    return np.array(out), worst


def derivatives(model, qpos):
    """qvel and qacc (n, nv) of a qpos trajectory, by finite differences."""
    qvel = np.zeros((len(qpos), model.nv))
    for k in range(1, len(qpos)):
        mujoco.mj_differentiatePos(model, qvel[k], DT, qpos[k - 1], qpos[k])
    return qvel, np.gradient(qvel, DT, axis=0)


def multibody_zmp(model, qpos):
    """ZMP the full robot needs for a qpos trajectory (n, nq), from inverse dynamics."""
    n = len(qpos)
    qvel, qacc = derivatives(model, qpos)
    data = mujoco.MjData(model)
    root = model.body("root").id
    zmp = []
    for k in range(n):
        data.qpos[:], data.qvel[:], data.qacc[:] = qpos[k], qvel[k], qacc[k]
        mujoco.mj_inverse(model, data)
        # The free joint's generalized force is the wrench the ground must supply:
        # force in world axes, torque in root axes, both about the root origin o.
        force = data.qfrc_inverse[:3]
        torque = data.xmat[root].reshape(3, 3) @ data.qfrc_inverse[3:6]
        o = data.xpos[root]
        zmp.append([o[0] - (torque[1] + o[2] * force[0]) / force[2],
                    o[1] + (torque[0] - o[2] * force[1]) / force[2]])
    return np.array(zmp)


def feedforward(model, qpos, feet, zmp_ref):
    """Joint torques (n, nu) and joint velocities (n, nu) the plan needs.

    mj_inverse gives the total generalized force. Its 6 free-joint rows can only come from
    the ground through the soles in contact (J^T F); whatever remains in the joint rows is
    what the servos must supply. In double support the ground wrench is split by least
    norm, weighted by where the ZMP reference sits between the feet.
    """
    qvel, qacc = derivatives(model, qpos)
    data = mujoco.MjData(model)
    soles = {side: model.geom(f"{side}_sole").id for side in "lr"}
    ground = min(feet["l"][0, 2], feet["r"][0, 2])  # sole-center height when standing
    act_dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    torques = []
    for k in range(len(qpos)):
        data.qpos[:], data.qvel[:], data.qacc[:] = qpos[k], qvel[k], qacc[k]
        mujoco.mj_inverse(model, data)
        left, right = feet["l"][k, :2], feet["r"][k, :2]
        share = np.clip((zmp_ref[k] - right) @ (left - right) / ((left - right) @ (left - right)), 0, 1)
        jacs, weights = [], []
        for side, weight in (("l", share), ("r", 1 - share)):
            if feet[side][k, 2] > ground + 1e-4:  # swing foot
                continue
            mujoco.mj_jacGeom(model, data, jac_pos, jac_rot, soles[side])
            jacs.append(np.vstack([jac_pos, jac_rot]))
            weights.append(max(weight, 1e-3))
        J = np.vstack(jacs)  # (6 per foot in contact, nv), wrench at the sole center
        W = np.repeat(weights, 6)
        A = J[:, :6].T  # how each sole wrench pushes the free joint
        wrench = W * (A.T @ np.linalg.solve((A * W) @ A.T, data.qfrc_inverse[:6]))
        torques.append((data.qfrc_inverse - J.T @ wrench)[act_dof])
    return np.array(torques), qvel[:, act_dof]


def at(samples, t):
    """Per-sample rows linearly interpolated at plan time t, clamped to the plan."""
    s = np.clip(t / DT, 0, len(samples) - 1.001)
    k = int(s)
    return samples[k] * (1 - (s - k)) + samples[k + 1] * (s - k)


def make_walk(model):
    """Plan the whole walk. Returns the control callback and the walk's duration in seconds."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    com0 = data.subtree_com[model.body("root").id][:2]
    sole0 = {side: data.geom(f"{side}_sole").xpos - [0, 0, SPAWN_CLEARANCE] for side in "lr"}
    zmp_ref, feet = plan(sole0, com0)
    com, _ = preview_com(zmp_ref, com0)
    n = len(com)
    zmp_ref, feet = zmp_ref[:n], {side: p[:n] for side, p in feet.items()}
    qpos, ik_error = solve_ik(model, com, feet)
    cart_table_error = np.linalg.norm(multibody_zmp(model, qpos) - zmp_ref, axis=1).mean()

    # Dynamics filter (Kajita 2003): the cart-table model ignores the swinging legs, so shift
    # the CoM by the preview-control answer to the full robot's ZMP error, and repeat.
    for _ in range(FILTER_PASSES):
        error = multibody_zmp(model, qpos) - zmp_ref
        still_future = np.repeat(error[-1:], round(PREVIEW_TIME / DT), axis=0)
        com_shift, _ = preview_com(np.vstack([error, still_future]), np.zeros(2))
        com = com - com_shift
        qpos, ik_error = solve_ik(model, com, feet)
    filtered_error = np.linalg.norm(multibody_zmp(model, qpos) - zmp_ref, axis=1).mean()
    print(f"plan: {n * DT:.1f} s | full-robot ZMP vs reference: mean {cart_table_error * 1e3:.1f} mm "
          f"-> {filtered_error * 1e3:.1f} mm after the dynamics filter | "
          f"IK residual max {ik_error * 1e3:.3f} mm")

    act_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    targets = qpos[:, act_qpos]
    commands = targets.copy()
    if FEEDFORWARD:
        # A position servo pushes kp (ctrl - q) - kv qdot. Shifting ctrl by (tau + kv qdot_ref) / kp
        # turns that into tau + kp (q_ref - q) + kv (qdot_ref - qdot): PD around the plan plus
        # the planned torque. On the XM430 this is an offset on Goal Position.
        torque, qdot = feedforward(model, qpos, feet, zmp_ref)
        kp, kv = model.actuator_gainprm[:, 0], -model.actuator_biasprm[:, 2]
        commands += (torque + kv * qdot) / kp
        print(f"feedforward: peak torque {np.abs(torque).max():.2f} N·m "
              f"({model.joint(int(model.actuator_trnid[np.abs(torque).max(0).argmax(), 0])).name})")
    ref_marker = marker(model, "zmp_ref")

    def controller(model, data):
        t = data.time - CROUCH_TIME
        if t < 0:  # crouch from the zero pose into the first planned posture
            data.ctrl[:] = commands[0] * smooth(data.time / CROUCH_TIME)
        else:
            data.ctrl[:] = at(commands, t)
        data.mocap_pos[ref_marker] = (*at(zmp_ref, t), 0.003)
        update_markers(model, data)

    controller.zmp_ref, controller.targets = zmp_ref, targets
    return controller, CROUCH_TIME + n * DT


def path_start(model):
    """Feet midpoint at spawn, where the path starts."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return (data.geom("l_sole").xpos[:2] + data.geom("r_sole").xpos[:2]) / 2


def scene_with_path():
    """The scene plus a yellow line on the floor along the walking path (visual only)."""
    start = path_start(mujoco.MjModel.from_xml_path(str(OUT)))
    spec = mujoco.MjSpec.from_file(str(OUT))
    points = start + np.array([path_pose(s)[:2] for s in np.linspace(0, N_STEPS * STEP_LENGTH, 200)])
    for a, b in zip(points[:-1], points[1:]):
        heading = np.arctan2(*(b - a)[::-1])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, contype=0, conaffinity=0,
                                size=[np.linalg.norm(b - a) / 2 + 0.001, 0.004, 0.0005],
                                pos=[*(a + b) / 2, 0.0005], rgba=[1, 0.85, 0.1, 1],
                                quat=[np.cos(heading / 2), 0, 0, np.sin(heading / 2)])
    return spec.compile()


def check_walk(model, controller, duration):
    """Walks the plan headless: stays upright, turns the full path, CoM near the line, ZMP close."""
    data = mujoco.MjData(model)
    act_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    center = path_start(model) + [0, RADIUS]
    mujoco.mj_forward(model, data)
    errors, lags, off_line, lowest = [], [], [], 1.0
    turned, heading = 0.0, 0.0
    while data.time < duration:
        mujoco.mj_step(model, data)
        imu = data.site("imu").xmat
        lowest = min(lowest, imu[8])
        new_heading = np.arctan2(imu[3], imu[0])
        turned += (new_heading - heading + np.pi) % (2 * np.pi) - np.pi
        heading = new_heading
        t = data.time - CROUCH_TIME
        if t < 0:
            continue
        lags.append(np.abs(data.qpos[act_qpos] - at(controller.targets, t)))
        com = data.subtree_com[model.body("root").id][:2]
        off_line.append(abs(np.linalg.norm(com - center) - RADIUS))
        zmp = measured_zmp(model, data)
        if zmp is not None:
            errors.append(np.linalg.norm(zmp - at(controller.zmp_ref, t)))
    planned = N_STEPS * STEP_LENGTH / RADIUS
    print(f"walk: turned {np.degrees(turned):.0f} deg (planned {np.degrees(planned):.0f}) | "
          f"CoM off the line: mean {np.mean(off_line) * 1e3:.1f} mm, max {np.max(off_line) * 1e3:.1f} mm | "
          f"min upright {lowest:.3f} | joint lag mean {np.degrees(np.mean(lags)):.2f} deg, "
          f"max {np.degrees(np.max(lags)):.1f} deg | ZMP vs reference: mean "
          f"{np.mean(errors) * 1e3:.1f} mm, max {np.max(errors) * 1e3:.1f} mm")
    assert lowest > 0.95, f"fell or tipped: upright {lowest:.3f}"
    assert abs(turned - planned) < 0.1 * planned, f"turned {np.degrees(turned):.0f} deg"
    assert np.max(off_line) < 0.05, f"CoM {np.max(off_line) * 1e3:.0f} mm off the line"


if __name__ == "__main__":
    build()
    op3 = scene_with_path()
    walk, seconds = make_walk(op3)
    mujoco.set_mjcb_control(walk)  # after loading: the compiler would trigger it too
    try:
        check_walk(op3, walk, seconds)
    except AssertionError as failure:  # still open the viewer, to see what went wrong
        print(f"CHECK FAILED: {failure}")
    mujoco.viewer.launch(op3)
