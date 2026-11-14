"""Pure-RL walking: the policy sets all 12 servo targets itself (no ZMP plan) and follows a velocity command.

N robots step in parallel inside MuJoCo's own thread pool (mujoco.rollout), so this is one process with numpy
arrays of shape (N, ...): no pipes, no pickling. On the M1: ~17-25k policy steps/s for 128 robots.

Policy, 50 Hz:  obs (47) -> action (12) in [-1, 1];  servo target = DEFAULT + ACTION_SCALE * action.
  obs = joint angle - default (12), joint velocity (12), gravity direction in the IMU frame (3), gyro (3),
        command vx, vy, yaw rate (3), gait clock sin/cos (2), last action (12)   <- all measurable on the robot
What makes the gait look natural is the reward (REWARD below): follow the commanded velocity, lift each foot
along a smooth arc in step with a clock (left, right, left ...; both feet down when told to stand), keep the
body upright and level, feet flat and not sliding, small smooth torques. The style is ours to tune there.

Servo model (XM430-W350 in position mode = PWM control): torque = clip(kp * error, +-stall) - (stall / no-load
speed) * velocity. The second term is the motor's back-EMF; it is what makes the real servo slow at speed.
Every episode randomizes mass, pelvis load and its position, friction, battery voltage, servo gain / damping /
rotor inertia, encoder zero offsets, IMU tilt, command latency (0-20 ms), sensor noise, and pushes.

Self-check:  python sim/rl_walk_env.py   (must stand 20 s with zero action; prints reward terms and speed)
"""
import copy
import time

import gymnasium as gym
import mujoco
import numpy as np
from mujoco import rollout
from scipy.spatial import ConvexHull
from stable_baselines3.common.vec_env import VecEnv

from run_sim import OUT, ROOT, STALL_TORQUE, build

POLICY_DT = 0.02  # s, 50 Hz
SIM_DT = 0.004  # s. ponytail: 2x faster than run_sim's 0.002; rl_walk_play.py re-checks the policy at 0.002
EPISODE = 1000  # policy steps = 20 s
NO_LOAD_SPEED = 4.8  # rad/s, XM430-W350 at 12 V (46 rpm)

# Standing pose the actions are added to: knees bent by KNEE, feet flat under the hips.
# Signs (CLAUDE.md): left hip_pitch + = leg back, knee + = bend, ankle_pitch + = toe down; right pitch axes flipped.
KNEE = 0.4  # rad
DEFAULT = np.array([0, 0, -KNEE / 2, KNEE, -KNEE / 2, 0, 0, 0, KNEE / 2, -KNEE, KNEE / 2, 0])
ACTION_SCALE = np.tile([0.25, 0.25, 0.5, 0.5, 0.5, 0.25], 2)  # rad at |action| = 1; yaw and roll joints move less

PERIOD = 0.7  # s for one left + one right step
SWING = 0.4  # share of the period each foot is in the air -> 10 % double support between steps
STEP_HEIGHT = 0.03  # m, peak of the swing foot's arc
COMMAND = {"low": np.array([-0.1, -0.08, -0.8]), "high": np.array([0.3, 0.08, 0.8])}  # vx, vy [m/s], yaw [rad/s]
STAND_SHARE = 0.1  # of the episodes get the command 0, 0, 0
COMMAND_STEPS = 250  # a new command every 5 s

REWARD = {  # weight of each term in WalkVecEnv._reward
    "alive": 0.5,
    "track_lin": 1.5,  # exp(-(velocity error / 0.1 m/s)^2)
    "track_yaw": 0.75,  # exp(-(yaw rate error)^2 / 0.1)
    "feet_phase": 1.0,  # exp(-(foot height error / 2 cm)^2): lift the right foot at the right time
    "tilt": -2.0,  # body not upright
    "height": -50.0,  # (pelvis height - standing height)^2: no deep crouch
    "bounce": -0.5,  # vertical velocity^2
    "wobble": -0.05,  # roll / pitch rate^2
    "torque": -5e-4,  # sum of torque^2
    "action_rate": -0.02,  # sum of (action - last action)^2: no jitter
    "hip_pose": -0.5,  # hip yaw / roll away from 0: toes point forward
    "slip": -0.5,  # foot sliding while it touches the floor
    "foot_flat": -2.0,  # sole not parallel to the floor
    "feet_close": -10.0,  # feet nearer than 8 cm sideways
    "stand": -0.5,  # joints away from DEFAULT when the command is to stand
}

RANDOM = {
    "mass": (0.9, 1.3),  # x CAD mass (1.506 kg): cables, screws, print infill
    "pelvis_load": (0.0, 1.2),  # kg on the pelvis: battery, boards -> 1.4 .. 3.2 kg total (weigh it, then narrow)
    "pelvis_com": 0.015,  # m, +- shift of the pelvis centre of mass
    "friction": (0.4, 1.2),
    "voltage": (0.8, 1.05),  # x 12 V: scales servo gain and stall torque (3S battery sag)
    "kp": (0.9, 1.1), "damping": (0.7, 1.3), "armature": (0.5, 1.5),  # x nominal, per joint
    "encoder": 0.03,  # rad, +- zero offset per joint (hand calibration)
    "imu_tilt": 0.035,  # +- error of the measured gravity direction (IMU mounting)
    "latency": (0.0, POLICY_DT),  # s between reading the sensors and the servos getting the new target
    "push": 0.3,  # m/s, +- velocity kick on the pelvis
    "push_steps": (150, 300),  # policy steps between kicks
    "noise": {"q": 0.005, "dq": 0.3, "gravity": 0.03, "gyro": 0.1},
}

Q, DQ = slice(8, 20), slice(26, 38)  # joint angles / velocities in mjSTATE_FULLPHYSICS: [time, qpos 19, qvel 18]
ROOT_VEL = slice(20, 22)


def make_model(sim_dt=SIM_DT):
    """op3_scene.xml with hull-only meshes (35 MB -> 1.8 MB per copy), the servo model, and reward sensors."""
    spec = mujoco.MjSpec.from_file(str(OUT))
    record = np.dtype([("normal", "<3f4"), ("verts", "<9f4"), ("attr", "<u2")])
    for mesh in spec.meshes:  # MuJoCo collides meshes as convex hulls anyway; the full meshes are only for looks
        verts = np.fromfile(ROOT / "model" / "OP3_sim" / mesh.file, dtype=record, offset=84)["verts"]
        verts = verts.reshape(-1, 3).astype(float)
        mesh.file, mesh.uservert = "", verts[ConvexHull(verts).vertices].flatten()
    S, O = mujoco.mjtSensor, mujoco.mjtObj
    spec.add_sensor(name="base_vel", type=S.mjSENS_VELOCIMETER, objtype=O.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="base_pos", type=S.mjSENS_FRAMEPOS, objtype=O.mjOBJ_SITE, objname="imu")
    for side in "lr":
        sole = spec.geom(f"{side}_sole")
        spec.body(f"{side}_foot_link_1").add_site(name=f"{side}_touch", type=mujoco.mjtGeom.mjGEOM_BOX,
                                                 pos=sole.pos, size=np.array(sole.size) + 0.003)
        spec.add_sensor(name=f"{side}_touch", type=S.mjSENS_TOUCH, objtype=O.mjOBJ_SITE, objname=f"{side}_touch")
        for name, kind in (("pos", S.mjSENS_FRAMEPOS), ("vel", S.mjSENS_FRAMELINVEL), ("down", S.mjSENS_FRAMEYAXIS)):
            spec.add_sensor(name=f"{side}_foot_{name}", type=kind, objtype=O.mjOBJ_GEOM, objname=f"{side}_sole")
    for actuator in spec.actuators:
        spec.add_sensor(name=f"torque_{actuator.name}", type=S.mjSENS_ACTUATORFRC, objtype=O.mjOBJ_ACTUATOR,
                        objname=actuator.name)
    spec.option.timestep = sim_dt
    model = spec.compile()
    model.actuator_biasprm[:, 2] = 0  # the XM430's Position D gain is 0 by default ...
    model.dof_damping[6:] = STALL_TORQUE / NO_LOAD_SPEED  # ... its damping is the back-EMF: 0.85 N·m·s/rad
    return model


class WalkVecEnv(VecEnv):
    render_mode = None

    def __init__(self, n_envs=128, randomize=True, seed=0, sim_dt=SIM_DT, nthread=8):
        self.nominal = make_model(sim_dt)
        m = self.nominal
        self.models = [copy.copy(m) for _ in range(n_envs)]
        self.pool = rollout.Rollout(nthread=nthread)
        self.datas = [mujoco.MjData(m) for _ in range(nthread)]
        self.scratch = mujoco.MjData(m)
        self.nstep = round(POLICY_DT / sim_dt)
        self.sim_dt = sim_dt
        self.randomize, self.rng = randomize, np.random.default_rng(seed)
        self.fixed_command = None  # set to (vx, vy, yaw) to evaluate one command
        self.sensor = {m.sensor(i).name: slice(int(m.sensor_adr[i]), int(m.sensor_adr[i] + m.sensor_dim[i]))
                       for i in range(m.nsensor)}
        self.torque = slice(self.sensor["torque_l_hip_yaw"].start, self.sensor["torque_r_ankle_roll"].stop)
        self.base = m.body("base_link").id
        self.contact_geoms = [m.geom("floor").id, m.geom("l_sole").id, m.geom("r_sole").id]

        # Standing state: DEFAULT pose, soles 0.5 mm above the floor.
        d = self.scratch
        d.qpos[7:] = DEFAULT
        mujoco.mj_kinematics(m, d)
        lowest = min(d.geom(f"{s}_sole").xpos[2] - np.abs(d.geom(f"{s}_sole").xmat.reshape(3, 3)[2])
                     @ m.geom(f"{s}_sole").size for s in "lr")
        d.qpos[2] -= lowest - 0.0005
        mujoco.mj_forward(m, d)
        for s in "lr":
            assert abs(d.geom(f"{s}_sole").xmat.reshape(3, 3)[2, 1]) > 0.999, "DEFAULT pose: soles are not flat"
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.state0 = np.zeros(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, d, self.state0, spec)
        self.height0 = d.site("imu").xpos[2]
        self.sole_z = m.geom("l_sole").size[1]  # sole centre height when the foot is flat on the floor

        N, nu = n_envs, m.nu
        self.state = np.tile(self.state0, (N, 1))
        self.state_out = np.zeros((N, self.nstep, len(self.state0)))
        self.sensor_out = np.zeros((N, self.nstep, m.nsensordata))
        self.phase, self.ep_step = np.zeros(N), np.zeros(N, int)
        self.command, self.last_action = np.zeros((N, 3)), np.zeros((N, nu))
        self.prev_target = np.tile(DEFAULT, (N, 1))
        self.encoder, self.imu_tilt = np.zeros((N, nu)), np.zeros((N, 2))
        self.latency, self.push_in = np.zeros(N, int), np.full(N, 10 ** 9)
        self.gravity, self.gyro = np.tile([0.0, 0, -1], (N, 1)), np.zeros((N, 3))
        self.sums, self.steps = {}, 0
        super().__init__(N, gym.spaces.Box(-np.inf, np.inf, (3 * nu + 11,), np.float32),
                         gym.spaces.Box(-1.0, 1.0, (nu,), np.float32))

    # --- randomization and reset ---------------------------------------------------------------------------
    def _randomize_model(self, i):
        m, n, rng, R = self.models[i], self.nominal, self.rng, RANDOM
        scale = rng.uniform(*R["mass"])
        m.body_mass[:], m.body_inertia[:] = n.body_mass * scale, n.body_inertia * scale
        b, load = self.base, rng.uniform(*R["pelvis_load"])
        m.body_inertia[b] *= (m.body_mass[b] + load) / m.body_mass[b]  # ponytail: load spread like the pelvis itself
        m.body_mass[b] += load
        m.body_ipos[b] = n.body_ipos[b] + rng.uniform(-R["pelvis_com"], R["pelvis_com"], 3)
        m.geom_friction[self.contact_geoms, 0] = rng.uniform(*R["friction"])
        volt = rng.uniform(*R["voltage"])
        kp = n.actuator_gainprm[:, 0] * volt * rng.uniform(*R["kp"], m.nu)
        m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
        m.actuator_forcerange[:] = n.actuator_forcerange * volt
        m.dof_damping[6:] = n.dof_damping[6:] * rng.uniform(*R["damping"], m.nu)
        m.dof_armature[6:] = n.dof_armature[6:] * rng.uniform(*R["armature"], m.nu)
        mujoco.mj_setConst(m, self.scratch)

    def _sample_commands(self, k):
        if self.fixed_command is not None:
            return np.tile(self.fixed_command, (k, 1)).astype(float)
        command = self.rng.uniform(COMMAND["low"], COMMAND["high"], (k, 3))
        command[self.rng.random(k) < STAND_SHARE] = 0
        return command

    def _reset(self, idx):
        k, rng, R = len(idx), self.rng, RANDOM
        self.state[idx] = self.state0
        self.encoder[idx], self.imu_tilt[idx], self.latency[idx] = 0, 0, 0
        if self.randomize:
            self.state[idx, Q] += rng.uniform(-0.05, 0.05, (k, 12))
            for i in idx:
                self._randomize_model(i)
            self.encoder[idx] = rng.uniform(-R["encoder"], R["encoder"], (k, 12))
            self.imu_tilt[idx] = rng.uniform(-R["imu_tilt"], R["imu_tilt"], (k, 2))
            self.latency[idx] = np.round(rng.uniform(*R["latency"], k) / self.sim_dt)
            self.push_in[idx] = rng.integers(*R["push_steps"], k)
        self.phase[idx], self.ep_step[idx] = rng.random(k), 0
        self.command[idx] = self._sample_commands(k)
        self.last_action[idx] = 0
        self.prev_target[idx] = DEFAULT + self.encoder[idx]
        self.gravity[idx], self.gyro[idx] = [0, 0, -1], 0

    def reset(self):
        self._reset(np.arange(self.num_envs))
        if self.randomize:  # spread the episode ends, so the robots don't all reset in the same step
            self.ep_step[:] = self.rng.integers(0, EPISODE, self.num_envs)
        return self._obs()

    # --- observation and reward ------------------------------------------------------------------------------
    def _obs(self):
        N, rng = self.num_envs, self.rng
        noise = (lambda key, n: rng.normal(0, RANDOM["noise"][key], (N, n))) if self.randomize else (lambda key, n: 0)
        gravity = self.gravity + noise("gravity", 3)
        gravity[:, :2] += self.imu_tilt
        clock = 2 * np.pi * self.phase
        return np.concatenate([
            self.state[:, Q] - self.encoder - DEFAULT + noise("q", 12),
            0.1 * (self.state[:, DQ] + noise("dq", 12)),
            gravity,
            0.25 * (self.gyro + noise("gyro", 3)),
            self.command,
            np.sin(clock)[:, None], np.cos(clock)[:, None],
            self.last_action,
        ], axis=1).astype(np.float32)

    def swing_height(self):
        """Target height of the (left, right) sole above the floor for the current clock, (N, 2)."""
        leg_phase = (self.phase[:, None] + [0.0, 0.5]) % 1
        arc = STEP_HEIGHT * np.sin(np.pi * np.clip(leg_phase / SWING, 0, 1)) ** 2
        return arc * (np.abs(self.command).sum(1) > 0)[:, None]  # stand: both feet stay down

    def _reward(self, s, action):
        get = lambda name: s[:, self.sensor[name]]
        w, x, y, z = get("imu_quat").T
        lateral = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1)  # body +Y in world
        velocity, q = get("base_vel"), self.state[:, Q]
        foot_z = np.stack([get("l_foot_pos")[:, 2], get("r_foot_pos")[:, 2]], axis=1) - self.sole_z
        touching = np.concatenate([get("l_touch"), get("r_touch")], axis=1) > 0
        foot_speed = np.stack([(get(f"{k}_foot_vel")[:, :2] ** 2).sum(1) for k in "lr"], axis=1)
        foot_tilt = sum((get(f"{k}_foot_down")[:, :2] ** 2).sum(1) for k in "lr")
        apart = np.abs(((get("l_foot_pos") - get("r_foot_pos")) * lateral).sum(1))
        standing = np.abs(self.command).sum(1) == 0
        return {
            "alive": np.ones(self.num_envs),
            "track_lin": np.exp(-((self.command[:, :2] - velocity[:, :2]) ** 2).sum(1) / 0.1 ** 2),
            "track_yaw": np.exp(-(self.command[:, 2] - self.gyro[:, 2]) ** 2 / 0.1),
            "feet_phase": np.exp(-((foot_z - self.swing_height()) ** 2).sum(1) / 0.02 ** 2),
            "tilt": (self.gravity[:, :2] ** 2).sum(1),
            "height": (get("base_pos")[:, 2] - self.height0) ** 2,
            "bounce": velocity[:, 2] ** 2,
            "wobble": (self.gyro[:, :2] ** 2).sum(1),
            "torque": (s[:, self.torque] ** 2).sum(1),
            "action_rate": ((action - self.last_action) ** 2).sum(1),
            "hip_pose": (q[:, [0, 1, 6, 7]] ** 2).sum(1),
            "slip": (foot_speed * touching).sum(1),
            "foot_flat": foot_tilt,
            "feet_close": np.clip(0.08 - apart, 0, None),
            "stand": ((q - DEFAULT) ** 2).sum(1) * standing,
        }

    # --- step --------------------------------------------------------------------------------------------------
    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        N, rng = self.num_envs, self.rng
        action = np.clip(self.actions, -1, 1).astype(float)
        target = DEFAULT + ACTION_SCALE * action
        target[:, 3], target[:, 9] = np.maximum(target[:, 3], 0), np.minimum(target[:, 9], 0)  # knees only bend one way
        target += self.encoder  # the servo's zero is off by this much
        late = np.arange(self.nstep)[None, :, None] < self.latency[:, None, None]
        ctrl = np.where(late, self.prev_target[:, None], target[:, None])
        self.prev_target = target
        if self.randomize:
            self.push_in -= 1
            hit = self.push_in <= 0
            self.state[hit, ROOT_VEL] += rng.uniform(-RANDOM["push"], RANDOM["push"], (hit.sum(), 2))
            self.push_in[hit] = rng.integers(*RANDOM["push_steps"], hit.sum())
        self.pool.rollout(self.models, self.datas, self.state, ctrl, state=self.state_out, sensordata=self.sensor_out)
        self.state = self.state_out[:, -1].copy()
        s = self.sensor_out[:, -1]

        w, x, y, z = s[:, self.sensor["imu_quat"]].T
        self.gravity = -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1)
        self.gyro = s[:, self.sensor["imu_gyro"]].copy()
        self.phase = (self.phase + POLICY_DT / PERIOD) % 1
        self.ep_step += 1
        terms = self._reward(s, action)
        reward = sum(REWARD[name] * value for name, value in terms.items())
        for name, value in terms.items():
            self.sums[name] = self.sums.get(name, 0.0) + REWARD[name] * value.mean()
        self.steps += 1
        self.last_action = action

        fell = (self.gravity[:, 2] > -0.5) | (s[:, self.sensor["base_pos"]][:, 2] < 0.5 * self.height0) \
            | ~np.isfinite(self.state).all(1)
        timeout = (self.ep_step >= EPISODE) & ~fell
        done = fell | timeout
        new_command = (self.ep_step % COMMAND_STEPS == 0) & ~done
        self.command[new_command] = self._sample_commands(new_command.sum())
        reward = np.where(np.isfinite(reward), reward, 0.0)
        obs = self._obs()
        infos = [{} for _ in range(N)]
        if done.any():
            idx = np.flatnonzero(done)
            for i in idx:
                infos[i] = {"terminal_observation": np.nan_to_num(obs[i]), "TimeLimit.truncated": bool(timeout[i])}
            self._reset(idx)
            obs[idx] = self._obs()[idx]
        return obs, reward.astype(np.float32), done, infos

    def pop_stats(self):
        """Mean weighted reward per term since the last call."""
        stats = {name: total / max(self.steps, 1) for name, total in self.sums.items()}
        self.sums, self.steps = {}, 0
        return stats

    # --- the rest of SB3's VecEnv interface -----------------------------------------------------------------
    def close(self):
        self.pool.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


if __name__ == "__main__":
    build()
    env = WalkVecEnv(n_envs=4, randomize=False)
    com = env.scratch.subtree_com[env.nominal.body("root").id]
    soles = np.mean([env.scratch.geom(f"{s}_sole").xpos for s in "lr"], axis=0)
    print(f"standing pose: pelvis {env.height0 * 1e3:.0f} mm high, CoM {1e3 * (com[0] - soles[0]):+.1f} mm ahead of "
          f"the sole centres, mass {env.nominal.body_subtreemass[env.nominal.body('root').id]:.3f} kg")
    env.fixed_command = (0, 0, 0)
    env.reset()
    falls = 0
    for _ in range(EPISODE - 1):
        _, _, done, _ = env.step(np.zeros((4, 12), np.float32))
        falls += done.sum()
    print("zero action, stand 20 s: falls", falls, "| reward terms per step:",
          {k: round(v, 3) for k, v in env.pop_stats().items() if abs(v) > 5e-4})
    assert falls == 0, "the robot must stand in the DEFAULT pose with zero action"
    env.close()

    env = WalkVecEnv(n_envs=128)
    obs = env.reset()
    rng, t, n = np.random.default_rng(0), time.perf_counter(), 300
    for _ in range(n):
        obs, reward, done, _ = env.step(rng.uniform(-1, 1, (128, 12)).astype(np.float32))
    rate = 128 * n / (time.perf_counter() - t)
    assert np.isfinite(obs).all() and np.isfinite(reward).all()
    print(f"random actions, randomized, 128 robots: {rate:.0f} policy steps/s, obs and rewards finite")
    env.close()
