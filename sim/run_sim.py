"""Run the simulation:  python sim/run_sim.py

Builds sim/op3_scene.xml from the raw ACDC4Robot export, checks that the robot
stands, then opens the MuJoCo viewer. The raw export is never edited.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "model" / "OP3_sim" / "OP3_sim.xml"
OUT = ROOT / "sim" / "op3_scene.xml"

# Tuning knobs. Only the torque limit comes from a datasheet.
STALL_TORQUE = 4.1  # N·m, XM430-W350 at 12 V (ROBOTIS e-Manual); lower on 3S
KP = 20.0  # N·m/rad, placeholder until the servo gains are matched
KV = 0.5  # N·m·s/rad, placeholder
ARMATURE = 0.01  # kg·m², placeholder for reflected rotor inertia, identify later
SOLE_THICKNESS = 0.008  # m, foot plate
SPAWN_CLEARANCE = 0.001  # m, soles above the floor at spawn
EXPECTED_MASS = 1.506  # kg, from CLAUDE.md


def foot_vertices(name):
    """Vertices of a link's binary STL, in meters, in that link's frame."""
    record = np.dtype([("normal", "<3f4"), ("verts", "<9f4"), ("attr", "<u2")])
    tris = np.fromfile(ROOT / "model" / "OP3_sim" / "meshes" / f"{name}.stl", dtype=record, offset=84)
    return tris["verts"].reshape(-1, 3) * 1e-3


def spawn_height():
    """Height of the raw model's origin that puts the soles SPAWN_CLEARANCE above z = 0."""
    model = mujoco.MjModel.from_xml_path(str(RAW))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    lowest = np.inf  # raw model: +Y is up
    for name in ("l_foot_link_1", "r_foot_link_1"):
        body = model.body(name).id
        world = data.xpos[body] + foot_vertices(name) @ data.xmat[body].reshape(3, 3).T
        lowest = min(lowest, world[:, 1].min())
    return -lowest + SPAWN_CLEARANCE


def build():
    tree = ET.parse(RAW)
    mjcf = tree.getroot()
    mjcf.find("compiler").set("meshdir", "../model/OP3_sim")
    world = mjcf.find("worldbody")

    # Floor.
    asset = mjcf.find("asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", width="512",
                  height="512", rgb1=".25 .3 .35", rgb2=".35 .4 .45")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="8 8", texuniform="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", material="grid")

    # Free root body: CAD (+Z fwd, +X left, +Y up) -> MuJoCo (+X fwd, +Y left, +Z up).
    base = world.find("body[@name='base_link']")
    world.remove(base)
    root = ET.SubElement(world, "body", name="root", pos=f"0 0 {spawn_height():.6f}",
                         quat="0.5 0.5 0.5 0.5")
    ET.SubElement(root, "freejoint", name="root")
    # Inverse quat, so the IMU reads in MuJoCo axes (Z up), not CAD axes.
    # ponytail: at the root origin until the real BMI088 position is modelled.
    ET.SubElement(root, "site", name="imu", size="0.005", quat="0.5 -0.5 -0.5 -0.5")
    root.append(base)

    # Markers, moved every step by update_markers(): red = CoM, green = measured ZMP,
    # blue = reference ZMP (only zmp_walk.py sets it). Parked below the floor until used.
    for name, rgba, shape in (("com", "1 0 0 1", {"type": "sphere", "size": "0.012"}),
                              ("zmp", "0 .8 0 1", {"type": "cylinder", "size": "0.012 0.001"}),
                              ("zmp_ref", ".2 .4 1 1", {"type": "cylinder", "size": "0.007 0.002"})):
        marker = ET.SubElement(world, "body", name=name, mocap="true", pos="0 0 -1")
        ET.SubElement(marker, "site", name=name, rgba=rgba, **shape)

    # One position actuator per joint. No joint ranges yet, see CLAUDE.md.
    actuators = ET.SubElement(mjcf, "actuator")
    for joint in base.iter("joint"):
        joint.set("armature", str(ARMATURE))
        ET.SubElement(actuators, "position", name=joint.get("name"), joint=joint.get("name"),
                      kp=str(KP), kv=str(KV), forcerange=f"{-STALL_TORQUE} {STALL_TORQUE}")

    # Collision: only one box per sole. Foot frame in the zero pose: +Y points down.
    for name in ("l_foot_link_1", "r_foot_link_1"):
        v = foot_vertices(name)
        low, high = v.min(axis=0), v.max(axis=0)
        center = (low + high) / 2
        center[1] = high[1] - SOLE_THICKNESS / 2
        half = (high - low) / 2
        half[1] = SOLE_THICKNESS / 2
        foot = base.find(f".//body[@name='{name}']")
        ET.SubElement(foot, "geom", name=f"{name[0]}_sole", type="box", group="3",
                      pos=" ".join(f"{x:.6f}" for x in center),
                      size=" ".join(f"{x:.6f}" for x in half))

    # Self-collision on the link meshes. MuJoCo collides meshes as convex hulls, so parts
    # can touch a little early. Bit 1 = floor, bit 2 = other links. The foot meshes skip the
    # floor because the sole boxes already handle it.
    for geom in base.iter("geom"):
        if geom.get("type") == "mesh":
            bits = "2" if "foot" in geom.get("name") else "3"
            geom.set("contype", bits)
            geom.set("conaffinity", bits)

    # These hulls overlap in the zero pose (they fill in the U-brackets), so skip these pairs.
    contact = ET.SubElement(mjcf, "contact")
    for side in "lr":
        for a, b in (("hip_yaw", "thigh"), ("shin", "foot")):
            ET.SubElement(contact, "exclude", body1=f"{side}_{a}_link_1", body2=f"{side}_{b}_link_1")

    # IMU.
    sensors = ET.SubElement(mjcf, "sensor")
    ET.SubElement(sensors, "accelerometer", name="imu_accel", site="imu")
    ET.SubElement(sensors, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sensors, "framequat", name="imu_quat", objtype="site", objname="imu")

    ET.indent(tree, space="    ")
    tree.write(OUT)


def marker(model, name):
    """mocap_pos index of a marker body."""
    return model.body_mocapid[model.body(name).id]


def measured_zmp(model, data):
    """ZMP (= center of pressure on flat ground) from the total contact wrench, or None if airborne.

    cfrc_ext is the contact wrench about the robot's CoM; MuJoCo fills it every step because
    of the accelerometer. Inside a control callback it is one physics step (2 ms) old.
    """
    c = data.subtree_com[model.body("root").id]
    torque, force = np.split(data.cfrc_ext.sum(axis=0), 2)
    if force[2] < 1.0:  # N
        return None
    return np.array([c[0] - (torque[1] + c[2] * force[0]) / force[2],
                     c[1] + (torque[0] - c[2] * force[1]) / force[2]])


def update_markers(model, data):
    """Control callback: red dot on the CoM, green disc on the measured ZMP."""
    data.mocap_pos[marker(model, "com")] = data.subtree_com[model.body("root").id]
    zmp = measured_zmp(model, data)
    data.mocap_pos[marker(model, "zmp")] = (0, 0, -1) if zmp is None else (*zmp, 0.002)


def check(model):
    """Done criteria from CLAUDE.md: right mass, stands 5 s in the zero pose under position hold."""
    data = mujoco.MjData(model)
    mass = model.body_mass.sum()
    assert abs(mass - EXPECTED_MASS) < 0.001, f"mass {mass:.4f} kg"
    assert model.nu == 12, f"{model.nu} actuators"

    root = model.body("root").id
    mujoco.mj_forward(model, data)
    touching = [(model.geom(c.geom[0]).name, model.geom(c.geom[1]).name) for c in data.contact[:data.ncon]]
    assert not touching, f"contacts at spawn, exclude or fix these: {touching}"
    start = data.xpos[root].copy()
    peak_torque = 0.0
    while data.time < 5.0:
        mujoco.mj_step(model, data)
        peak_torque = max(peak_torque, np.abs(data.actuator_force).max())

    moved = data.xpos[root] - start
    upright = data.site("imu").xmat[8]  # IMU z axis . world z
    assert upright > 0.99, f"fell over: IMU z axis . world z = {upright:.3f}"
    assert np.linalg.norm(moved) < 0.02, f"root moved {moved} m"
    assert data.ncon >= 2, f"{data.ncon} contacts"
    dot_error = np.linalg.norm(data.site("com").xpos - data.subtree_com[root])
    assert dot_error < 0.001, f"CoM dot is {dot_error * 1e3:.1f} mm off"

    # The ZMP from cfrc_ext must match the center of pressure of the individual contacts.
    force, weight, cop = np.zeros(6), 0.0, np.zeros(2)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, force)
        weight += force[0]
        cop += force[0] * data.contact[i].pos[:2]
    zmp_error = np.linalg.norm(measured_zmp(model, data) - cop / weight)
    assert zmp_error < 0.001, f"measured ZMP is {zmp_error * 1e3:.1f} mm off the contact CoP"
    print(f"OK  mass {mass:.3f} kg | root moved {np.linalg.norm(moved) * 1e3:.1f} mm in 5 s | "
          f"upright {upright:.4f} | contacts {data.ncon} | peak torque {peak_torque:.2f} N·m")


if __name__ == "__main__":
    build()
    op3 = mujoco.MjModel.from_xml_path(str(OUT))
    mujoco.set_mjcb_control(update_markers)  # after loading: the compiler would trigger it too
    try:
        check(op3)
    except AssertionError as failure:  # still open the viewer, to see what went wrong
        print(f"CHECK FAILED: {failure}")
    mujoco.viewer.launch(op3)
