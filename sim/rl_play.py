"""Compare a trained policy with the plain ZMP walk:  mjpython sim/rl_play.py sim/runs/<run> [--view]
(mjpython: macOS needs it for the viewer. During training it uses the newest checkpoint.)
Prints steps survived and mean path error for both, over the same 20 randomized episodes
(mass, friction, gains, delay, pushes). --view then opens the viewer on one policy episode;
--video out.mp4 saves both walks side by side on the same episode.
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env import POLICY_DT, WalkEnv, run
from run_sim import build

if __name__ == "__main__":
    folder = Path(sys.argv[1])
    model_file, norm_file = folder / "model.zip", folder / "vecnormalize.pkl"
    if not model_file.exists():  # still training: use the newest checkpoint
        steps = max(int(p.stem.split("_")[-2]) for p in (folder / "checkpoints").glob("rl_model_*_steps.zip"))
        model_file = folder / "checkpoints" / f"rl_model_{steps}_steps.zip"
        norm_file = folder / "checkpoints" / f"rl_model_vecnormalize_{steps}_steps.pkl"
        print(f"checkpoint at {steps / 1e6:.0f} M steps")
    build()
    env = WalkEnv()
    norm = VecNormalize.load(str(norm_file), DummyVecEnv([lambda: env]))
    norm.training = False
    model = PPO.load(model_file, device="cpu")
    policy = lambda obs: model.predict(norm.normalize_obs(obs), deterministic=True)[0]
    zero = lambda obs: np.zeros(env.model.nu)
    for name, act in (("ZMP walk only", zero), ("ZMP + policy ", policy)):
        results = np.array([run(env, act, seed=1000 + s) for s in range(20)])
        print(f"{name}: survived {np.mean(results[:, 0] == env.horizon) * 100:3.0f} % | "
              f"path error mean {results[:, 1].mean():5.1f} mm, worst {results[:, 1].max():5.1f} mm")
    if "--video" in sys.argv:  # side by side, same randomized episode, real time at 25 fps
        out = sys.argv[sys.argv.index("--video") + 1]
        env.model.vis.global_.offwidth, env.model.vis.global_.offheight = 640, 720
        renderer, cam = mujoco.Renderer(env.model, 720, 640), mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 135, -20, 0.9
        clips = []
        for act in (zero, policy):
            obs, _ = env.reset(seed=1000)
            frames, done = [], False
            while not done:
                obs, _, fell, truncated, info = env.step(act(obs))
                done = fell or truncated
                if env.k % 2 == 0:
                    cam.lookat[:] = [*env.data.body("root").xpos[:2], 0.15]
                    renderer.update_scene(env.data, cam)
                    frames.append((renderer.render(), info["path_error"]))
            clips.append(frames)
        ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", "1280x720", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "20",
                                   "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        for (a, ea), (b, eb) in zip(*clips):  # ponytail: stops at the shorter clip if one falls
            image = Image.fromarray(np.hstack([a, b]))
            draw = ImageDraw.Draw(image)
            for x, label, error in ((20, "ZMP walk only", ea), (660, f"ZMP + RL policy", eb)):
                draw.text((x, 16), f"{label}\noff path {error * 1e3:4.1f} mm", font=font, fill="white",
                          stroke_width=3, stroke_fill="black")
            ffmpeg.stdin.write(image.tobytes())
        ffmpeg.stdin.close()
        ffmpeg.wait()
        print(f"wrote {out}")
    if "--view" in sys.argv:
        obs, _ = env.reset(seed=1000)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            done = False
            while viewer.is_running() and not done:
                obs, _, fell, truncated, _ = env.step(policy(obs))
                done = fell or truncated
                viewer.sync()
                time.sleep(POLICY_DT)
