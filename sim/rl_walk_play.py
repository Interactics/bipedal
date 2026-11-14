"""Evaluate, watch or record a trained pure-RL walk:
  python   sim/rl_walk_play.py sim/runs/<run>                     table: 64 randomized robots x 10 s per command
  python   sim/rl_walk_play.py sim/runs/<run> --video out.mp4     one robot through TOUR (stand, walk, turn, back)
  mjpython sim/rl_walk_play.py sim/runs/<run> --view              same in the viewer (macOS needs mjpython)
  ... --cmd 0.2 0 0     hold one command (vx m/s, vy m/s, yaw rad/s) instead of TOUR
Uses the newest of model.zip and the checkpoints, so it also works while training runs.
The table also runs at a 2 ms physics step (training uses 4 ms): the policy must not depend on the step size.
"""
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from rl_walk_env import POLICY_DT, WalkVecEnv
from run_sim import OUT, build

COMMANDS = [(0, 0, 0), (0.1, 0, 0), (0.2, 0, 0), (0.3, 0, 0), (-0.1, 0, 0), (0, 0.08, 0), (0, 0, 0.8), (0.2, 0, 0.5)]
TOUR = [(2, (0, 0, 0)), (6, (0.2, 0, 0)), (4, (0.15, 0, 0.6)), (3, (0.3, 0, 0)), (3, (-0.1, 0, 0)), (2, (0, 0, 0))]


def load(folder, env):
    files = [folder / "model.zip", *(folder / "checkpoints").glob("rl_model_*_steps.zip")]
    newest = max((f for f in files if f.exists()), key=lambda f: f.stat().st_mtime)
    stats = folder / "vecnormalize.pkl" if newest.name == "model.zip" else \
        newest.with_name(newest.name.replace("rl_model_", "rl_model_vecnormalize_").replace(".zip", ".pkl"))
    norm = VecNormalize.load(str(stats), env)
    norm.training = False
    model = PPO.load(newest, device="cpu")
    print(f"policy: {newest.relative_to(folder)} at {model.num_timesteps / 1e6:.1f} M steps")
    return lambda obs: model.predict(norm.normalize_obs(obs), deterministic=True)[0]


def table(folder):
    for sim_dt in (0.004, 0.002):
        env = WalkVecEnv(64, seed=1, sim_dt=sim_dt)
        policy = load(folder, env) if sim_dt == 0.004 else policy
        print(f"physics step {sim_dt * 1e3:.0f} ms | command (vx, vy, yaw) -> measured, robots that fell in 10 s")
        for command in COMMANDS:
            env.fixed_command = command
            obs = env.reset()
            env.ep_step[:] = 0
            fell, measured = np.zeros(64, bool), []
            for k in range(500):
                obs, _, done, _ = env.step(policy(obs))
                fell |= done
                if k >= 100:  # after the start-up
                    s = env.sensor_out[:, -1]
                    measured.append(np.c_[s[:, env.sensor["base_vel"]][:, :2], s[:, env.sensor["imu_gyro"]][:, 2]])
            vx, vy, yaw = np.mean(measured, axis=(0, 1))
            print(f"  {command[0]:+.2f} {command[1]:+.2f} {command[2]:+.1f} -> {vx:+.2f} {vy:+.2f} {yaw:+.1f} | "
                  f"fell {fell.sum():2d} / 64")
        env.close()


def show(folder, video, view, command):
    env = WalkVecEnv(1, randomize=False)
    policy = load(folder, env)
    tour = [(10, command)] if command else TOUR
    full = mujoco.MjModel.from_xml_path(str(OUT))  # the detailed meshes, only for drawing
    data = mujoco.MjData(full)
    obs = env.reset()

    def frames():
        nonlocal obs
        for seconds, cmd in tour:
            env.fixed_command = cmd
            env.command[:] = cmd
            for _ in range(round(seconds / POLICY_DT)):
                env.ep_step[:] = 0  # no 20 s time limit here: done then only means a fall
                obs, _, done, _ = env.step(policy(obs))
                data.qpos[:] = env.state[0, 1:20]
                mujoco.mj_forward(full, data)
                yield cmd, env.sensor_out[0, -1, env.sensor["base_vel"]][0], bool(done[0])

    if view:
        with mujoco.viewer.launch_passive(full, data) as viewer:
            for _ in frames():
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(POLICY_DT)
        return
    full.vis.global_.offwidth, full.vis.global_.offheight = 1280, 720
    renderer, cam = mujoco.Renderer(full, 720, 1280), mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 130, -15, 1.0
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                               "1280x720", "-r", "50", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt",
                               "yuv420p", video], stdin=subprocess.PIPE)
    look, falls = None, 0
    for cmd, speed, fell in frames():
        falls += fell
        target = np.array([*data.body("root").xpos[:2], 0.17])
        look = target if look is None or fell else look + 0.1 * (target - look)
        cam.lookat[:] = look
        renderer.update_scene(data, cam)
        image = Image.fromarray(renderer.render())
        ImageDraw.Draw(image).text(
            (20, 16), f"command: forward {cmd[0]:+.2f} m/s, sideways {cmd[1]:+.2f} m/s, turn {cmd[2]:+.1f} rad/s\n"
                      f"measured forward speed {speed:+.2f} m/s" + (f"   falls: {falls}" if falls else ""),
            font=font, fill="white", stroke_width=3, stroke_fill="black")
        ffmpeg.stdin.write(image.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"wrote {video} ({falls} falls)")


if __name__ == "__main__":
    folder, args = Path(sys.argv[1]), sys.argv
    build()
    command = tuple(float(x) for x in args[args.index("--cmd") + 1:args.index("--cmd") + 4]) if "--cmd" in args else None
    if "--video" in args or "--view" in args:
        show(folder, args[args.index("--video") + 1] if "--video" in args else None, "--view" in args, command)
    else:
        table(folder)
