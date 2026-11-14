"""Residual RL environment around the ZMP line walk (forward 10 steps, back 10, stand up).

The policy runs at 50 Hz and adds a small correction (±ACTION_SCALE rad) to zmp_walk's servo command,
so a zero action replays the ZMP walk exactly. It sees what the real robot can measure: joint angles and
velocities, the IMU (gravity direction + gyro), the gait phase and its last action.
Each episode randomizes total mass, foot friction, servo gains, a 0-1 step action delay, sensor noise and
pushes, so the policy has to learn corrections that keep the walk on the reference path.

Self-check:  python sim/rl_env.py   (zero action must walk the whole plan, a random policy must not crash)
"""
import gymnasium as gym
import mujoco
import numpy as np

from run_sim import OUT, build
from zmp_walk import STEP_TIME, make_walk

POLICY_DT = 0.02  # s, 50 Hz like a real RL loop; the sim steps at 2 ms -> 10 substeps
ACTION_SCALE = 0.15  # rad (~9 deg) of correction at |action| = 1
MASS_RANGE = (1.0, 2.2)  # x the CAD mass (1.506 kg) -> 1.5..3.3 kg; the real robot is ~3 kg
FRICTION_RANGE = (0.5, 1.2)
GAIN_RANGE = (0.8, 1.2)  # x the servo kp / kv placeholders in run_sim.py
PUSH_FORCE = 3.0  # N, max horizontal push on the pelvis, held for PUSH_TIME, every ~2-4 s
PUSH_TIME = 0.1  # s
NOISE = {"q": 0.005, "qvel": 0.1, "gravity": 0.02, "gyro": 0.05}  # rad, rad/s, unit vector, rad/s


class WalkEnv(gym.Env):
    def __init__(self, randomize=True, seed=None):
        self.model = mujoco.MjModel.from_xml_path(str(OUT))  # run build() once before making envs
        self.data = mujoco.MjData(self.model)
        self.randomize = randomize
        self.substeps = round(POLICY_DT / self.model.opt.timestep)
        self.nominal = {k: getattr(self.model, k).copy() for k in
                        ("body_mass", "body_inertia", "geom_friction", "actuator_gainprm", "actuator_biasprm")}
        self.root = self.model.body("root").id
        floor = np.flatnonzero(self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        self.contact_geoms = [*floor, self.model.geom("l_sole").id, self.model.geom("r_sole").id]

        # Reference: zmp_walk's commands, plus where the root goes when the ZMP walk plays them (nominal model).
        walk, duration = make_walk(self.model, "line")
        self.horizon = int(duration / POLICY_DT)
        scratch = mujoco.MjData(self.model)
        self.ref_ctrl = np.zeros((self.horizon + 1, self.model.nu))
        self.ref_root = np.zeros((self.horizon + 1, 3))  # x, y, yaw
        for k in range(self.horizon + 1):
            scratch.time = k * POLICY_DT
            walk(self.model, scratch)
            self.ref_ctrl[k] = scratch.ctrl
            self.ref_root[k] = self._root_pose(scratch)
            scratch.ctrl[:] = self.ref_ctrl[k]
            mujoco.mj_step(self.model, scratch, nstep=self.substeps)
        self.act_qpos = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        self.act_qvel = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]

        n = self.model.nu
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3 * n + 9,), np.float32)
        self._seed = seed

    def _root_pose(self, data):
        R = data.site("imu").xmat
        return np.array([*data.body("root").xpos[:2], np.arctan2(R[3], R[0])])

    def _obs(self):
        d, rng = self.data, self.np_random
        R = d.site("imu").xmat.reshape(3, 3)
        k = min(self.k, self.horizon)
        phase = 2 * np.pi * k * POLICY_DT / (2 * STEP_TIME)  # one left + right step per cycle
        noise = lambda key, size: rng.normal(0, NOISE[key], size) if self.randomize else 0.0
        obs = np.concatenate([
            d.qpos[self.act_qpos] - self.ref_ctrl[k] + noise("q", self.model.nu),
            0.1 * (d.qvel[self.act_qvel] + noise("qvel", self.model.nu)),
            R.T @ [0, 0, -1] + noise("gravity", 3),
            d.sensor("imu_gyro").data + noise("gyro", 3),
            [np.sin(phase), np.cos(phase), k / self.horizon],
            self.last_action,
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self._seed)
        self._seed = None
        m, rng = self.model, self.np_random
        for key, value in self.nominal.items():
            getattr(m, key)[:] = value
        if self.randomize:
            scale = rng.uniform(*MASS_RANGE)  # ponytail: uniform scale; model the battery/boards when weighed
            m.body_mass[:] *= scale
            m.body_inertia[:] *= scale
            m.geom_friction[self.contact_geoms, 0] = rng.uniform(*FRICTION_RANGE)
            gain = rng.uniform(*GAIN_RANGE, m.nu)
            m.actuator_gainprm[:, 0] *= gain
            m.actuator_biasprm[:, 1:3] *= gain[:, None]
        mujoco.mj_resetData(m, self.data)
        mujoco.mj_forward(m, self.data)
        self.k = 0
        self.delay = self.randomize and rng.random() < 0.5
        self.last_action = np.zeros(m.nu)
        self.pending = np.zeros(m.nu)
        self.next_push = rng.uniform(2, 4)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        applied = self.pending if self.delay else action
        self.pending = action
        d, rng = self.data, self.np_random
        t = self.k * POLICY_DT
        if self.randomize and t >= self.next_push:
            angle = rng.uniform(0, 2 * np.pi)
            d.xfrc_applied[self.root, :2] = rng.uniform(0, PUSH_FORCE) * np.array([np.cos(angle), np.sin(angle)])
            self.push_until, self.next_push = t + PUSH_TIME, t + rng.uniform(2, 4)
        if t >= getattr(self, "push_until", 0):
            d.xfrc_applied[self.root] = 0
        d.ctrl[:] = self.ref_ctrl[self.k] + ACTION_SCALE * applied
        mujoco.mj_step(self.model, d, nstep=self.substeps)
        self.k += 1

        pose, ref = self._root_pose(d), self.ref_root[self.k]
        path_error = np.linalg.norm(pose[:2] - ref[:2])
        yaw_error = (pose[2] - ref[2] + np.pi) % (2 * np.pi) - np.pi
        upright = d.site("imu").xmat[8]  # IMU z . world z
        reward = (1.0
                  + np.exp(-(path_error / 0.03) ** 2)
                  + 0.5 * np.exp(-(yaw_error / 0.1) ** 2)
                  + 0.5 * np.exp(-((1 - upright) / 0.01) ** 2)
                  - 0.05 * np.sum(action ** 2)
                  - 0.05 * np.sum((action - self.last_action) ** 2))
        self.last_action = action
        fell = upright < 0.8
        if fell:
            reward -= 50
        return self._obs(), float(reward), bool(fell), self.k >= self.horizon, {"path_error": path_error}


def run(env, policy, seed=0):
    """One episode -> (steps survived, mean path error [mm])."""
    obs, _ = env.reset(seed=seed)
    errors, done = [], False
    while not done:
        obs, _, fell, truncated, info = env.step(policy(obs))
        errors.append(info["path_error"])
        done = fell or truncated
    return env.k, 1e3 * np.mean(errors)


if __name__ == "__main__":
    build()
    from gymnasium.utils.env_checker import check_env
    env = WalkEnv(randomize=False)
    check_env(WalkEnv(), skip_render_check=True)
    zero = lambda obs: np.zeros(env.model.nu)
    steps, error = run(env, zero)
    print(f"zero action, nominal: {steps}/{env.horizon} steps, path error {error:.1f} mm")
    assert steps == env.horizon and error < 10, "zero action must replay the ZMP walk"
    noisy = WalkEnv()
    results = [run(noisy, zero, seed=s) for s in range(5)]
    print("zero action, randomized (steps, path mm):", [(s, round(e, 1)) for s, e in results])
    rng = np.random.default_rng(0)
    steps, _ = run(noisy, lambda obs: rng.uniform(-1, 1, env.model.nu))
    print(f"random action, randomized: {steps}/{env.horizon} steps")
