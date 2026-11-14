"""Train the residual walking policy with PPO on the Mac CPU:  python sim/rl_train.py [million_steps] [run_name]
Writes sim/runs/<run_name>/: checkpoints, model.zip, vecnormalize.pkl, TensorBoard logs.
Watch it:  tensorboard --logdir sim/runs     Replay:  python sim/rl_play.py sim/runs/<run_name>
"""
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from rl_env import WalkEnv
from run_sim import build

N_ENVS = 8  # M1, measured: 6 -> ~3.1k, 8 -> ~4.5-5k, 12-24 no better (pipe overhead per step); Metal (mps) 5x slower


def make(rank):
    def thunk():
        torch.set_num_threads(1)
        return WalkEnv(seed=rank)
    return thunk


if __name__ == "__main__":
    steps = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    run = Path(__file__).parent / "runs" / (sys.argv[2] if len(sys.argv) > 2 else "residual")
    build()  # once, before the workers load op3_scene.xml
    env = VecNormalize(VecMonitor(SubprocVecEnv([make(i) for i in range(N_ENVS)])), norm_reward=False)
    model = PPO("MlpPolicy", env, n_steps=2048, batch_size=4096, n_epochs=5, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                # small start noise (std e^-1 x 0.15 rad ~ 3 deg): the first policy is almost the ZMP walk
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0),
                device="cpu", tensorboard_log=str(run), verbose=1)
    save = CheckpointCallback(max(1_000_000 // N_ENVS, 1), str(run / "checkpoints"), save_vecnormalize=True)
    try:
        model.learn(int(steps * 1e6), callback=save, tb_log_name="ppo")
    finally:  # Ctrl+C still keeps what was learned
        model.save(run / "model")
        env.save(str(run / "vecnormalize.pkl"))
        print(f"saved {run}")
