"""Record an RL policy's forward walk for the real robot, to replay like the ZMP walk (no policy on the robot):
  python sim/export_rl_walk.py sim/runs/<run> [vx m/s = 0.15] [walk seconds = 8]
Writes hardware/motions/walk_rl.csv in export_walk.py's format (one row per 10 ms, servo commands in rad, 0 = home):
1 s home -> bent-knee pose, 1 s stand, walk, 1.5 s stop, 1 s back to home. Play:  cd hardware && ./build/walk motions/walk_rl.csv
The rows are the servo targets the policy sent in simulation on the nominal robot. On the robot nothing reacts:
it is a recording. The self-check replays it blind on 64 randomized robots (mass, friction, servo gains, pushes).
"""
import pickle
import sys
from pathlib import Path

import numpy as np

from rl_blind import BlindVecEnv
from rl_walk_env import ACTION_SCALE, DEFAULT, POLICY_DT, WalkVecEnv
from rl_walk_play import load
from run_sim import ROOT, build
JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]  # as export_walk.py (importing it runs it)

if __name__ == "__main__":
    folder = Path(sys.argv[1])
    vx, seconds = (float(sys.argv[2]) if len(sys.argv) > 2 else 0.15), (float(sys.argv[3]) if len(sys.argv) > 3 else 8.0)
    build()
    blind = pickle.load(open(folder / "vecnormalize.pkl", "rb")).obs_rms.mean.size == 41
    env = (BlindVecEnv if blind else WalkVecEnv)(1, randomize=False)
    policy, obs, actions = load(folder, env), env.reset(), []
    for duration, command in ((1.0, 0), (seconds, vx), (1.5, 0)):
        env.fixed_command = (command, 0, 0)
        env.command[:] = env.fixed_command
        for _ in range(round(duration / POLICY_DT)):
            actions.append(np.clip(policy(obs), -1, 1)[0])
            obs, _, done, _ = env.step(actions[-1][None])
            assert not done[0], "the policy fell in the nominal simulation"
    actions = np.array(actions)
    names = [env.nominal.actuator(i).name for i in range(12)]
    env.close()

    targets = DEFAULT + ACTION_SCALE * actions
    targets[:, 3], targets[:, 9] = np.maximum(targets[:, 3], 0), np.minimum(targets[:, 9], 0)  # as WalkVecEnv does
    ease = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, 50))[:, None]  # 1 s, smooth start and stop
    plan = np.vstack([ease * targets[0], targets, (1 - ease) * targets[-1]])
    t = np.arange(0, (len(plan) - 1) * POLICY_DT, 0.01)  # 50 Hz -> the 10 ms rows walk.cpp expects
    rows = np.column_stack([t, *(np.interp(t, np.arange(len(plan)) * POLICY_DT, plan[:, j]) for j in range(12))])
    ids = [(3 if name.startswith("l_") else 4) * 10 + JOINTS.index(name[2:]) + 1 for name in names]
    csv = ROOT / "hardware" / "motions" / "walk_rl.csv"
    np.savetxt(csv, rows, fmt="%.5f", delimiter=",", header=",".join(["t", *map(str, ids)]), comments="")
    print(f"wrote {csv}: {len(rows)} rows, {t[-1]:.1f} s | peak |command| "
          f"{np.degrees(np.abs(rows[:, 1:]).max(0)).round(0).tolist()} deg")

    # Self-check: the same targets with nothing reacting, on 64 different robots. This is what the real robot gets.
    for label, randomize in (("nominal robot", False), ("64 randomized robots, with pushes", True)):
        env = WalkVecEnv(64, randomize=randomize, seed=2)
        env.reset()
        env.ep_step[:] = 0
        x0, yaw0, fell = env.state[:, 1].copy(), 0, np.zeros(64, bool)
        for a in actions:
            _, _, done, _ = env.step(np.tile(a, (64, 1)).astype(np.float32))
            x = np.where(fell | done, np.nan, env.state[:, 1])  # a robot that fell has been reset
            fell |= done
        moved = (x - x0)[~fell]
        print(f"replay on {label}: fell {fell.sum()} / 64 | forward "
              f"{np.mean(moved):.2f} m (min {np.min(moved):.2f}, max {np.max(moved):.2f})" if len(moved) else
              f"replay on {label}: all fell")
        env.close()
