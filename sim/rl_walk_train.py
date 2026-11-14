"""Train the pure-RL walk on the Mac:  python sim/rl_walk_train.py [million_steps] [run_name] [--cpu]
Writes sim/runs/<run_name>/: model.zip, vecnormalize.pkl, checkpoints/, TensorBoard logs. Running it again with
the same name continues that run. Ctrl+C saves and stops.
Watch:  tensorboard --logdir sim/runs      Replay:  python sim/rl_walk_play.py sim/runs/<run_name> --video out.mp4

M1 layout (measured): 128 robots step in MuJoCo's thread pool; the policy runs on the CPU for every step
(tiny batches: the GPU round trip costs more than the math); the PPO update runs on the GPU through Metal
(batches of 8192: ~3x faster than the CPU). --cpu keeps the update on the CPU too.
"""
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from rl_walk_env import WalkVecEnv
from run_sim import build

N_ENVS, N_STEPS = 128, 256  # 32768 samples per update


class MetalPPO(PPO):
    """PPO that collects on the CPU and does its gradient steps on the Mac GPU."""

    def _move(self, device):
        params = list(self.policy.parameters())
        self.policy.to(device)
        assert all(a is b for a, b in zip(params, self.policy.parameters())), "optimizer would lose the parameters"
        for state in self.policy.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        self.rollout_buffer.device = torch.device(device)

    def train(self):
        self._move("mps")
        try:
            super().train()
        finally:
            self._move("cpu")


class RewardTerms(BaseCallback):
    """Logs each reward term (weighted, mean per step) to TensorBoard under reward/."""

    def __init__(self, walk):
        super().__init__()
        self.walk = walk

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        for name, value in self.walk.pop_stats().items():
            self.logger.record(f"reward/{name}", value)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    steps = float(args[0]) if args else 100
    run = Path(__file__).parent / "runs" / (args[1] if len(args) > 1 else "walk")
    algo = PPO if "--cpu" in sys.argv or not torch.backends.mps.is_available() else MetalPPO
    torch.set_num_threads(4)  # the 4 performance cores; 8 threads measured no faster
    build()
    walk = WalkVecEnv(N_ENVS)
    if (run / "model.zip").exists():
        env = VecNormalize.load(str(run / "vecnormalize.pkl"), VecMonitor(walk))
        model = algo.load(run / "model", env=env, device="cpu")
        print(f"continuing {run} from {model.num_timesteps / 1e6:.1f} M steps")
    else:
        env = VecNormalize(VecMonitor(walk), clip_obs=10.0)
        model = algo("MlpPolicy", env, n_steps=N_STEPS, batch_size=8192, n_epochs=5, learning_rate=3e-4,
                     gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005, max_grad_norm=1.0,
                     # start noise e^-1 x ACTION_SCALE ~ 10 deg on the pitch joints, 5 deg on yaw / roll
                     policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ELU, log_std_init=-1.0),
                     device="cpu", tensorboard_log=str(run), verbose=1)
    callbacks = [RewardTerms(walk), CheckpointCallback(5_000_000 // N_ENVS, str(run / "checkpoints"),
                                                      save_vecnormalize=True)]
    try:
        model.learn(int(steps * 1e6), callback=callbacks, tb_log_name="ppo", reset_num_timesteps=False)
    except KeyboardInterrupt:
        pass
    finally:
        model.save(run / "model")
        env.save(str(run / "vecnormalize.pkl"))
        print(f"saved {run} at {model.num_timesteps / 1e6:.1f} M steps")
