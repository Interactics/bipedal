# Vertical 1080x1920 30 fps MP4 of the line walk (forward, back, stand), real time, slow orbiting camera.
#   python tools/reels.py out.mp4 [preview.png]   (from any folder)
import os, subprocess, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sim"))
import numpy as np, mujoco
import zmp_walk as w
W, H, FPS, LENGTH = 1080, 1920, 30, 15.0  # ponytail: Reels length; the walk is 15.1 s, the last 0.1 s is the end of standing up
w.build()
m = w.scene_with_path("line")
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
m.vis.headlight.ambient[:] = 0.35; m.vis.headlight.diffuse[:] = 0.6
m.vis.quality.shadowsize = 8192; m.vis.quality.offsamples = 8
walk, sec = w.make_walk(m, "line"); mujoco.set_mjcb_control(walk)
d = mujoco.MjData(m); root = m.body("root").id
r = mujoco.Renderer(m, H, W); cam = mujoco.MjvCamera(); opt = mujoco.MjvOption(); opt.sitegroup[:] = 0
total = LENGTH
look = None
def frame():
    global look
    com = d.subtree_com[root].copy(); com[2] = 0.14
    look = com if look is None else look + 0.15 * (com - look)   # smooth follow
    s = d.time / total
    cam.lookat[:] = look
    cam.azimuth = 120 + 120 * s          # orbit from front-left to front-right
    cam.elevation = -24
    cam.distance = 1.0
    r.update_scene(d, cam, opt); return r.render()
if len(sys.argv) > 2:                    # preview frame only
    while d.time < 4: mujoco.mj_step(m, d)
    from PIL import Image; Image.fromarray(frame()).save(sys.argv[2]); sys.exit()
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", sys.argv[1]], stdin=subprocess.PIPE)
n = 0
while n < round(total * FPS):
    mujoco.mj_step(m, d)
    if n < d.time * FPS: ff.stdin.write(frame().tobytes()); n += 1
ff.stdin.close(); ff.wait()
print(n, "frames", n / FPS, "s, upright", d.site("imu").xmat[8])
