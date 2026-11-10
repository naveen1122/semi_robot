import os, sys, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
from PIL import Image
PKG, OUT = sys.argv[1], sys.argv[2]
m = mujoco.MjModel.from_xml_path(os.path.join(PKG, "mjcf", "scene_empty.xml")); d = mujoco.MjData(m)
A = {m.actuator(i).name: i for i in range(m.nu)}
def run(sec, c):
    for k, v in c.items(): d.ctrl[A[k]] = v
    for _ in range(int(sec / m.opt.timestep)): mujoco.mj_step(m, d)
base = m.body("base_link").id
r = mujoco.Renderer(m, 900, 1200)
cam = mujoco.MjvCamera(); opt = mujoco.MjvOptiondef if False else mujoco.MjvOption()
def shot(fn, az, el, dist, z=0.75, groups=(2,)):
    o = mujoco.MjvOption()
    for g in range(6): o.geomgroup[g] = 1 if g in groups else 0
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    cam.lookat[:] = d.xpos[base] + np.array([0, 0, z - 0.1])
    r.update_scene(d, cam, o); Image.fromarray(r.render()).save(os.path.join(OUT, fn))
run(1.0, {})
shot("pose_zero_front.png", 150, -15, 2.4); shot("pose_zero_side.png", 90, -3, 2.6); shot("pose_zero_rear.png", -35, -20, 2.4)
shot("pose_zero_collision.png", 150, -15, 2.4, groups=(3,))
c = {"lift": 0.45, "head_pan": 0.4, "head_tilt": 0.6}
for side, sg in [("left", 1), ("right", -1)]:
    for k, v in {"j1": 0.25, "j2": 1.0, "j3": 1.2, "j4": -0.6, "j5": 0.0, "j6": 0.0}.items():
        c["%s_arm_%s" % (side, k)] = v * (sg if k in ("j1", "j5", "j6") else 1)
    c[side + "_gripper"] = -0.045
run(18.0, c)
shot("pose_work_front.png", 150, -15, 2.6, z=0.95); shot("pose_work_side.png", 90, -3, 2.8, z=0.95)
print("lift", round(d.qpos[m.joint("lift_joint").qposadr[0]], 3))
