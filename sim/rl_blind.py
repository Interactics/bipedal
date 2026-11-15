"""The pure-RL walk WITHOUT an IMU: the policy sees only what the servos and a timer give.
  python sim/rl_blind.py                                   self-check
  python sim/rl_blind.py train [million_steps] [run_name]  -> sim/runs/<run_name>  (default 100, blind)
  python sim/rl_blind.py play sim/runs/<run_name> [--video out.mp4 | --view] [--cmd vx vy yaw]

Same robot, reward, randomization and PPO settings as rl_walk_env.py / rl_walk_train.py; the only difference is
the observation: 41 numbers = the 47 of the IMU walk minus gravity direction (3) and gyro (3).
  obs = joint angle - default (12), joint velocity (12), command (3), gait clock sin/cos (2), last action (12)
The reward still uses the body's tilt and speed: it only exists in simulation, where they are known.
ponytail: one frame of observation, same inputs for the critic. If it cannot hold its balance, the upgrades are
a few past frames in the observation (lets it feel a tilt building up) and a critic that sees the IMU.
"""
import sys

import gymnasium as gym
import numpy as np
import torch

from rl_walk_env import ACT_MIRROR, OBS_MIRROR, SymmetricNets, SymmetricPolicy, WalkVecEnv

KEEP = np.delete(np.arange(47), np.r_[24:30])  # drop gravity (24:27) and gyro (27:30)
# The IMU entries mirror onto themselves, so the mirror map of the rest is the old one, renumbered.
BLIND_MIRROR = (np.searchsorted(KEEP, OBS_MIRROR[0][KEEP]), OBS_MIRROR[1][KEEP])


class BlindVecEnv(WalkVecEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(KEEP),), np.float32)

    def _obs(self):
        return super()._obs()[:, KEEP]


class BlindNets(SymmetricNets):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.obs_perm = torch.as_tensor(BLIND_MIRROR[0])
        self.obs_sign = torch.as_tensor(BLIND_MIRROR[1], dtype=torch.float32)


class BlindPolicy(SymmetricPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = BlindNets(self.features_dim, self.action_space.shape[0])


def check():
    full = np.random.default_rng(0).normal(size=47)
    assert np.allclose((full[OBS_MIRROR[0]] * OBS_MIRROR[1])[KEEP], full[KEEP][BLIND_MIRROR[0]] * BLIND_MIRROR[1])
    env = BlindVecEnv(n_envs=4, randomize=False)
    obs = env.reset()
    assert obs.shape == (4, 41) and np.allclose(obs[:, :24], 0, atol=1e-6), obs.shape  # standing in DEFAULT
    env.close()
    nets, o = BlindNets(41, 12), torch.randn(8, 41)
    perm, sign = torch.as_tensor(ACT_MIRROR[0]), torch.as_tensor(ACT_MIRROR[1], dtype=torch.float32)
    assert torch.allclose(nets.forward_actor(nets.mirror(o)), nets.forward_actor(o)[:, perm] * sign, atol=1e-6)
    print("ok: 41 observations without the IMU, mirrored situation -> mirrored action")


if __name__ == "__main__":
    import rl_blind as B  # so a saved model names its classes rl_blind.*, not __main__.*
    from run_sim import build

    mode = sys.argv.pop(1) if len(sys.argv) > 1 else "check"
    if mode == "train":
        import rl_walk_train
        sys.argv[1:] = (sys.argv[1:] + ["100", "blind"][len(sys.argv) - 1:])[:2] + [a for a in sys.argv if a == "--cpu"]
        rl_walk_train.main(B.BlindVecEnv, B.BlindPolicy, B.BLIND_MIRROR)
    elif mode == "play":
        import rl_walk_play
        rl_walk_play.main(B.BlindVecEnv)
    else:
        build()
        B.check()
