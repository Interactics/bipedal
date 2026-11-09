"""Export a ZMP walk for the real robot:  python sim/export_walk.py [circle|line]
Writes hardware/motions/walk.csv (circle) or walk_line.csv (forward + backward): a header of servo IDs, then one row per 10 ms: t [s] and the 12 servo
commands [rad, sim sign convention, 0 = home pose], including the crouch from home and standing back up at the end. The commands are
exactly what zmp_walk sends to its MuJoCo position servos (plan + torque feedforward offset).
hardware/build/walk plays the file.
"""
import sys

import mujoco
import numpy as np

from run_sim import OUT, ROOT, build
from zmp_walk import DT, make_walk

JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
GAIT = sys.argv[1] if len(sys.argv) > 1 else "circle"
CSV = ROOT / "hardware" / "motions" / ("walk.csv" if GAIT == "circle" else f"walk_{GAIT}.csv")

build()
model = mujoco.MjModel.from_xml_path(str(OUT))
walk, duration = make_walk(model, GAIT)
data = mujoco.MjData(model)
rows = []
for t in np.arange(0, duration, DT):  # ponytail: replay the sim's own controller, no duplicated timing logic
    data.time = t
    walk(model, data)
    rows.append([t, *data.ctrl])

names = [model.actuator(i).name for i in range(model.nu)]
ids = [(3 if name.startswith("l_") else 4) * 10 + JOINTS.index(name[2:]) + 1 for name in names]
np.savetxt(CSV, rows, fmt="%.5f", delimiter=",", header=",".join(["t", *map(str, ids)]), comments="")
peak = np.abs(np.array(rows)[:, 1:]).max(0)
print(f"wrote {CSV}: {len(rows)} rows, {duration:.1f} s | ids {ids} | peak |command| "
      f"{np.degrees(peak).round(1).tolist()} deg")
