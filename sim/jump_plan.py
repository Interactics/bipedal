"""Plan the jump offline:  python sim/jump_plan.py   (about 2 min) -> sim/jump_plan.npz
Then simulate it:  python sim/wbmp_jump.py

Dai, Valenzuela, Tedrake, "Whole-body Motion Planning with Centroidal Dynamics and Full
Kinematics", Humanoids 2014. Contact sequence is fixed (paper section II-D):
stance (crouch, push) -> flight -> stance (land), feet back on the same spot.

CasADi + IPOPT; Pinocchio gives exact derivatives of the kinematics.
Decision variables at every knot k: q, v, CoM r, r', r'', angular momentum h, h',
corner contact forces F (4 sole corners per foot), and the time step dt[k].
Constraints (paper eq. 7a-7k):
  m r'' = sum F + m g                    centroidal dynamics
  h'    = sum (c - r) x F
  h     = A(q) v                         full kinematics -> centroidal momentum
  q[k]  = q[k-1] (+) v[k] dt[k]          backward Euler (on the quaternion manifold)
  h[k] - h[k-1] = h'[k] dt[k],  r'[k] - r'[k-1] = r''[k] dt[k]   (7e, 7g)
  m r'  = A(q) v (linear part)           replaces the trapezoidal (7f), see plan_jump
  r = com(q),  c = corners(q)            stance: corners on their footprint; flight: F = 0
  friction pyramid, joint ranges
Added for the XM430: joint torque (full inverse dynamics) under the torque-speed line,
and a trunk tilt limit.
Cost (6): posture error + velocity + CoM acceleration + contact force, weighted by dt.
The saved plan holds joint angles, joint velocities and the feedforward joint torques
(inverse dynamics with the planned contact forces).
"""
import time

import casadi as ca
import mujoco
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from run_sim import OUT, RAW, ROOT, STALL_TORQUE, build

PLAN = ROOT / "sim" / "jump_plan.npz"

# Jump knobs.
CLEARANCE = 0.005  # m, every sole corner this high at mid-flight (10 mm is infeasible for the XM430 with the margins below)
N_PUSH, N_FLIGHT, N_LAND = 20, 10, 20  # knots per phase
DT_RANGE = (0.01, 0.05)  # s, bounds on each stance time step
FLIGHT_DT_RANGE = (0.002, 0.05)  # s, in the air: 11 steps x 10 ms would force a 110 ms flight
NO_LOAD_SPEED = 4.8  # rad/s, XM430-W350 at 12 V (46 rpm); less under load and on 3S
MAX_TILT = 10  # deg, trunk tilt from upright; open-loop landing can't recover a big tilt
TORQUE_MARGIN = 0.8  # plan within this fraction of the XM430 torque-speed line, the rest is for tracking
MU = 0.7  # friction coefficient (MuJoCo default is 1)
SQUAT = 0.2  # rad, nominal posture: hip -/+SQUAT, knee +/-2 SQUAT, ankle -/+SQUAT (L/R)
W_POSTURE, W_ORIENT, W_VEL, W_ACC, W_FORCE = 1.0, 10.0, 1e-2, 1e-3, 1e-3  # cost weights
G = np.array([0, 0, -9.81])


class Robot:
    """Pinocchio model of the robot (raw export + free-flyer root) and its links to MuJoCo."""

    def __init__(self, mj_model):
        self.mj = mj_model
        self.model = pin.buildModelFromMJCF(str(RAW), pin.JointModelFreeFlyer(), "root")[0]
        self.joints = list(self.model.names)[2:]  # same names as the MuJoCo joints / actuators
        self.mj_qpos = [mj_model.joint(n).qposadr[0] for n in self.joints]
        self.mj_dof = [mj_model.joint(n).dofadr[0] for n in self.joints]
        self.actuators = [mj_model.actuator(n).id for n in self.joints]
        self.model.armature[6:] = mj_model.dof_armature[self.mj_dof]
        self.mass = sum(inertia.mass for inertia in self.model.inertias)

        # One frame per sole-box bottom corner (foot frame: +y points down, out of the sole).
        self.corners = []
        for side in "lr":
            body = self.model.getFrameId(f"{side}_foot_link_1")
            sole = mj_model.geom(f"{side}_sole")
            for sx, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                local = sole.pos + sole.size * [sx, 1, sz]
                frame = self.model.frames[body]
                placement = frame.placement * pin.SE3(np.eye(3), local)
                self.corners.append(self.model.addFrame(
                    pin.Frame(f"{side}_corner_{len(self.corners)}", frame.parentJoint, body,
                              placement, pin.FrameType.OP_FRAME)))
        self.data = self.model.createData()

    def to_mujoco(self, q):
        """Pinocchio q (x y z, qx qy qz qw, joints) -> MuJoCo qpos (x y z, qw qx qy qz, joints)."""
        qpos = np.zeros(self.mj.nq)
        qpos[:3], qpos[3:7] = q[:3], [q[6], q[3], q[4], q[5]]
        qpos[self.mj_qpos] = q[7:]
        return qpos

    def corner_positions(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        return np.array([self.data.oMf[f].translation for f in self.corners])

    def nominal(self):
        """Squat posture, upright, soles flat on z = 0 where they are at spawn."""
        data = mujoco.MjData(self.mj)
        mujoco.mj_kinematics(self.mj, data)
        q = np.zeros(self.model.nq)
        q[3:7] = [*data.qpos[4:7], data.qpos[3]]
        for name, angle in {"l_hip_pitch": -SQUAT, "l_knee": 2 * SQUAT, "l_ankle_pitch": -SQUAT,
                            "r_hip_pitch": SQUAT, "r_knee": -2 * SQUAT, "r_ankle_pitch": SQUAT}.items():
            q[7 + self.joints.index(name)] = angle
        spawn_feet = (data.geom("l_sole").xpos + data.geom("r_sole").xpos)[:2] / 2
        corners = self.corner_positions(q)
        q[:2] += spawn_feet - corners[:, :2].mean(axis=0)
        q[2] -= corners[:, 2].min()
        return q


def check_models(robot):
    """Pinocchio and MuJoCo must agree: mass, CoM, corners, and joint torques (rnea vs mj_inverse)."""
    mj, data = robot.mj, mujoco.MjData(robot.mj)
    rng = np.random.default_rng(0)
    q = robot.nominal()
    q[7:] += rng.uniform(-0.3, 0.3, 12)
    data.qpos[:] = robot.to_mujoco(q)
    data.qvel[robot.mj_dof] = rng.uniform(-2, 2, 12)  # base at rest: both conventions agree
    data.qacc[robot.mj_dof] = rng.uniform(-20, 20, 12)
    flags = mj.opt.disableflags
    mj.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # rigid-body dynamics only
    mujoco.mj_inverse(mj, data)
    mj.opt.disableflags = flags
    v, a = np.zeros(robot.model.nv), np.zeros(robot.model.nv)
    v[6:], a[6:] = data.qvel[robot.mj_dof], data.qacc[robot.mj_dof]
    tau = pin.rnea(robot.model, robot.data, q, v, a)
    com_error = np.linalg.norm(pin.centerOfMass(robot.model, robot.data, q) - data.subtree_com[mj.body("root").id])
    corners = robot.corner_positions(q)
    mj_corners = []
    for side in "lr":
        sole = data.geom(f"{side}_sole")
        R = sole.xmat.reshape(3, 3)
        mj_corners += [sole.xpos + R @ (mj.geom(f"{side}_sole").size * [sx, 1, sz])
                       for sx, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    assert abs(robot.mass - mj.body_mass.sum()) < 1e-6, "mass differs"
    assert com_error < 1e-9, f"CoM differs by {com_error}"
    assert np.abs(corners - mj_corners).max() < 1e-9, "sole corners differ"
    assert np.abs(tau[6:] - data.qfrc_inverse[robot.mj_dof]).max() < 1e-9, "joint torques differ"
    print("models: Pinocchio == MuJoCo (mass, CoM, sole corners, joint torques)")


def plan_jump(robot):
    """Solve the trajectory optimization. Returns knot times and per-knot q, v, F (8 corners)."""
    model = cpin.Model(robot.model)
    data = model.createData()
    nq, nv, nc = robot.model.nq, robot.model.nv, len(robot.corners)
    q_s, v_s, dq_s = ca.SX.sym("q", nq), ca.SX.sym("v", nv), ca.SX.sym("dq", nv)
    cpin.framesForwardKinematics(model, data, q_s)
    kinematics = ca.Function("kinematics", [q_s], [cpin.centerOfMass(model, data, q_s),
                                                   ca.horzcat(*[data.oMf[f].translation for f in robot.corners]),
                                                   data.oMi[1].rotation[2, 1]])  # root's CAD up axis . world z
    cpin.computeCentroidalMomentum(model, data, q_s, v_s)
    momentum = ca.Function("momentum", [q_s, v_s], [data.hg.linear, data.hg.angular])
    integrate = ca.Function("integrate", [q_s, dq_s], [cpin.integrate(model, q_s, dq_s)])
    # Joint torques from full inverse dynamics with the corner forces (beyond the paper, which
    # has no torque limits; the XM430 needs them).
    a_s, f_s = ca.SX.sym("a", nv), ca.SX.sym("f", 3, nc)
    tau = cpin.rnea(model, data, q_s, v_s, a_s)
    cpin.computeJointJacobians(model, data, q_s)
    cpin.updateFramePlacements(model, data)
    for j, frame in enumerate(robot.corners):
        tau -= cpin.getFrameJacobian(model, data, frame, pin.LOCAL_WORLD_ALIGNED)[:3, :].T @ f_s[:, j]
    torque = ca.Function("torque", [q_s, v_s, a_s, f_s], [tau[6:]])

    q_nom = robot.nominal()
    footprint = robot.corner_positions(q_nom).T  # (3, nc)
    n = N_PUSH + N_FLIGHT + N_LAND
    stance = [k < N_PUSH or k >= N_PUSH + N_FLIGHT for k in range(n)]

    opti = ca.Opti()
    Q, V = opti.variable(nq, n), opti.variable(nv, n)
    R, RD, RDD = opti.variable(3, n), opti.variable(3, n), opti.variable(3, n)
    H, HD = opti.variable(3, n), opti.variable(3, n)
    F = opti.variable(3 * nc, n)
    DT = opti.variable(n - 1)  # DT[k - 1] is the step from knot k - 1 to knot k

    low, high = np.full(12, -1.2), np.full(12, 1.2)  # rad, until real joint ranges are chosen
    low[robot.joints.index("l_knee")], high[robot.joints.index("l_knee")] = 0.0, 2.3  # no hyperextension
    low[robot.joints.index("r_knee")], high[robot.joints.index("r_knee")] = -2.3, 0.0
    cost = 0
    for k in range(n):
        com, corners, upright = kinematics(Q[:, k])
        opti.subject_to(upright >= np.cos(np.radians(MAX_TILT)))
        forces = ca.reshape(F[:, k], 3, nc)
        opti.subject_to(R[:, k] == com)  # (7h)
        linear, angular = momentum(Q[:, k], V[:, k])
        opti.subject_to(H[:, k] == angular)  # (7c)
        # ponytail: in place of the paper's trapezoidal (7f). With backward-Euler q, (7f) lets r'
        # drift from the real CoM velocity, and the full-body dynamics no longer match the
        # centroidal ones (77 N base residual at take-off). m r' = A(q) v (linear) ties them.
        opti.subject_to(robot.mass * RD[:, k] == linear)
        opti.subject_to(robot.mass * RDD[:, k] == ca.sum2(forces) + robot.mass * G)  # (7a)
        opti.subject_to(HD[:, k] == sum(ca.cross(corners[:, j] - R[:, k], forces[:, j]) for j in range(nc)))  # (7b)
        if stance[k] and k > 0:  # knot 0 is fixed by q[0] = q_nom
            # (7i, 7j): no slip, sole flat on its footprint. 6 equations per foot, not 12: four
            # corners of one rigid sole are redundant, and IPOPT fails on redundant constraints.
            # Corner order per foot: (+x,+z), (+x,-z), (-x,-z), (-x,+z); sole +z points forward.
            for first in (0, 4):
                opti.subject_to(corners[:, first] == footprint[:, first])  # pins position
                opti.subject_to(corners[1:, first + 1] == footprint[1:, first + 1])  # pitch, yaw
                opti.subject_to(corners[2, first + 3] == footprint[2, first + 3])  # roll
        if stance[k]:
            opti.subject_to(forces[2, :] >= 0)  # (7k): friction pyramid
            for axis in (0, 1):
                opti.subject_to(opti.bounded(-MU / np.sqrt(2) * forces[2, :], forces[axis, :],
                                             MU / np.sqrt(2) * forces[2, :]))
        else:
            opti.subject_to(F[:, k] == 0)
            opti.subject_to(corners[2, :] >= (CLEARANCE if k == N_PUSH + N_FLIGHT // 2 else 0.0))
        opti.subject_to(opti.bounded(low, Q[7:, k], high))
        if k > 0:
            dt = DT[k - 1]
            # XM430 torque-speed line: |tau| <= stall (1 - |w| / no-load speed), all four sign cases.
            t_k = torque(Q[:, k], V[:, k], (V[:, k] - V[:, k - 1]) / dt, forces)
            for sign_t in (1, -1):
                for sign_w in (1, -1):
                    opti.subject_to(sign_t * t_k + STALL_TORQUE * sign_w * V[6:, k] / NO_LOAD_SPEED
                                    <= TORQUE_MARGIN * STALL_TORQUE)
            dt_low, dt_high = DT_RANGE if stance[k] and stance[k - 1] else FLIGHT_DT_RANGE
            opti.subject_to(opti.bounded(dt_low, dt, dt_high))  # not bounded(*DT_RANGE, dt)!
            opti.subject_to(Q[:, k] == integrate(Q[:, k - 1], V[:, k] * dt))  # (7d)
            opti.subject_to(H[:, k] - H[:, k - 1] == HD[:, k] * dt)  # (7e)
            opti.subject_to(RD[:, k] - RD[:, k - 1] == RDD[:, k] * dt)  # (7g)
            cost += dt * (W_POSTURE * ca.sumsqr(Q[7:, k] - q_nom[7:]) + W_ORIENT * ca.sumsqr(Q[3:6, k] - q_nom[3:6])
                          + W_VEL * ca.sumsqr(V[:, k]) + W_ACC * ca.sumsqr(RDD[:, k]) + W_FORCE * ca.sumsqr(F[:, k]))
    # Start and end at rest in the nominal posture (r' = 0 and h = 0 follow from v = 0).
    opti.subject_to(Q[:, 0] == q_nom)
    opti.subject_to(V[:, 0] == 0)
    opti.subject_to(V[:, -1] == 0)
    opti.subject_to(RDD[:, 0] == 0)  # knot 0 has no (7g): without these its forces are free
    opti.subject_to(HD[:, 0] == 0)
    opti.subject_to(Q[7:, -1] == q_nom[7:])
    opti.minimize(cost)

    # Initial guess: standing still, weight shared by the corners.
    opti.set_initial(Q, np.tile(q_nom, (n, 1)).T)
    opti.set_initial(R, np.tile(pin.centerOfMass(robot.model, robot.data, q_nom), (n, 1)).T)
    opti.set_initial(F, np.array([np.tile([0, 0, robot.mass * 9.81 / nc], nc) * s for s in stance]).T)
    opti.set_initial(DT, 0.03)
    opti.solver("ipopt", {"print_time": False}, {"max_iter": 3000, "print_level": 0, "sb": "yes"})
    start = time.perf_counter()
    solution = opti.solve()
    print(f"plan: IPOPT {opti.stats()['return_status']} in {opti.stats()['iter_count']} iterations, "
          f"{time.perf_counter() - start:.1f} s")
    dt = np.atleast_1d(solution.value(DT))
    return {"t": np.concatenate([[0], np.cumsum(dt)]), "q": solution.value(Q).T, "v": solution.value(V).T,
            "F": solution.value(F).T.reshape(n, nc, 3), "stance": np.array(stance)}


def joint_torques(robot, plan):
    """Inverse dynamics with the planned contact forces: joint torques (n, 12), base residual (n,)."""
    model, data = robot.model, robot.data
    q, v, F, t = plan["q"], plan["v"], plan["F"], plan["t"]
    a = np.zeros_like(v)
    a[1:] = np.diff(v, axis=0) / np.diff(t)[:, None]  # backward differences, like (7g)
    torques, residual = [], []
    for k in range(len(q)):
        tau = pin.rnea(model, data, q[k], v[k], a[k])
        pin.computeJointJacobians(model, data, q[k])
        pin.updateFramePlacements(model, data)
        for j, frame in enumerate(robot.corners):
            J = pin.getFrameJacobian(model, data, frame, pin.LOCAL_WORLD_ALIGNED)[:3]
            tau -= J.T @ F[k, j]
        torques.append(tau[6:])
        residual.append(np.linalg.norm(tau[:6]))  # 0 if centroidal and full dynamics agree
    return np.array(torques), np.array(residual)


if __name__ == "__main__":
    build()
    robot = Robot(mujoco.MjModel.from_xml_path(str(OUT)))
    check_models(robot)
    plan = plan_jump(robot)
    torque, residual = joint_torques(robot, plan)
    t, q, v = plan["t"], plan["q"], plan["v"]
    flight = t[N_PUSH + N_FLIGHT - 1] - t[N_PUSH - 1]
    com = np.array([pin.centerOfMass(robot.model, robot.data, qk) for qk in q])
    speed_use = np.abs(torque) / STALL_TORQUE + np.abs(v[:, 6:]) / NO_LOAD_SPEED  # <= 1 under the line
    print(f"plan: {t[-1]:.2f} s, flight {flight * 1e3:.0f} ms, CoM rise {(com[:, 2].max() - com[0, 2]) * 1e3:.1f} mm | "
          f"peak joint speed {np.abs(v[:, 6:]).max():.2f} rad/s, peak torque {np.abs(torque).max():.2f} N·m | "
          f"torque-speed use {speed_use.max() * 100:.0f}% of the XM430 line | "
          f"base residual (centroidal vs full dynamics) max {residual.max():.3f} N")
    np.savez(PLAN, joints=robot.joints, t=t, q=q[:, 7:], v=v[:, 6:], torque=torque,
             stance=plan["stance"], clearance=CLEARANCE)
    print(f"saved {PLAN.relative_to(ROOT)}")
