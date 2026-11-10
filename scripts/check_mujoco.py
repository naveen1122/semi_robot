"""Physics check of robot_v1 in MuJoCo: settle, drive, turn, lift, arms, pan-tilt, gripper; renders views and depth."""
import os, sys, json, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

PKG = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PKG, "media")
os.makedirs(OUT, exist_ok=True)
m = mujoco.MjModel.from_xml_path(os.path.join(PKG, "mjcf", "scene.xml"))
d = mujoco.MjData(m)
A = {m.actuator(i).name: i for i in range(m.nu)}
J = lambda n: m.joint(n)
qpos = lambda n: d.qpos[J(n).qposadr[0]]
base = m.body("base_link").id

def step(sec, ctrl=None):
    if ctrl:
        for k, v in ctrl.items():
            d.ctrl[A[k]] = v
    for _ in range(int(sec / m.opt.timestep)):
        mujoco.mj_step(m, d)

def pose():
    p = d.xpos[base].copy()
    R = d.xmat[base].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    tilt = math.degrees(math.acos(max(-1, min(1, R[2, 2]))))
    return p, yaw, tilt

res = {}
mujoco.mj_resetData(m, d)
d.qpos[0] = -3.0                                    # start 3 m behind the table for the driving tests
d.ctrl[:] = 0
step(2.0)
p, yaw, tilt = pose()
res["settle"] = {"base_link_z": round(p[2], 4), "tilt_deg": round(tilt, 3), "xy_drift_m": round(float(np.hypot(p[0], p[1])), 4)}
# ground contact forces per wheel
def wheel_loads():
    loads = {}
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = m.geom(c.geom1), m.geom(c.geom2)
        if g1.name != "floor" and g2.name != "floor": continue
        other = g2 if g1.name == "floor" else g1
        bname = m.body(int(other.bodyid[0])).name
        f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
        loads[bname] = loads.get(bname, 0) + f[0]
    return {k: round(v / 9.81, 2) for k, v in loads.items()}
res["wheel_loads_kg"] = wheel_loads()

p0, yaw0, _ = pose()
step(3.0, {"left_wheel": 5.0, "right_wheel": 5.0})
p1, yaw1, tilt1 = pose()
step(1.0, {"left_wheel": 0.0, "right_wheel": 0.0})
res["drive_5rad_s_3s"] = {"distance_m": round(float(np.hypot(*(p1 - p0)[:2])), 3), "expected_m": round(5 * 0.1 * 3, 3),
                          "yaw_change_deg": round(math.degrees(yaw1 - yaw0), 2), "tilt_deg": round(tilt1, 2)}
p0, yaw0, _ = pose()
step(3.0, {"left_wheel": -3.0, "right_wheel": 3.0})
p1, yaw1, _ = pose()
step(1.0, {"left_wheel": 0.0, "right_wheel": 0.0})
dyaw = (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi
res["spin_3rad_s_3s"] = {"yaw_change_deg": round(math.degrees(dyaw), 1),
                         "expected_deg": round(math.degrees(3 * 0.1 * 2 / 0.5002 * 3) % 360, 1),
                         "centre_drift_m": round(float(np.hypot(*(p1 - p0)[:2])), 3)}
# back in front of the table for lift, arms and cameras
mujoco.mj_resetData(m, d)
d.ctrl[:] = 0
step(1.0)
# lift
step(20.0, {"lift": 0.5})
res["lift_to_0.5"] = {"lift_joint": round(qpos("lift_joint"), 4), "stage2": round(qpos("lift_stage2_joint"), 4)}
step(20.0, {"lift": 0.0})
res["lift_back_to_0"] = round(qpos("lift_joint"), 4)
# arms, head, gripper
ARM = {"j1": 0.3, "j2": 1.2, "j3": 1.6, "j4": -0.4, "j5": 0.0, "j6": 0.5}
c = {}
for side, sg in [("left", 1), ("right", -1)]:
    for k, v in ARM.items():
        c["%s_arm_%s" % (side, k)] = v * (sg if k in ("j1", "j5", "j6") else 1)
    c[side + "_gripper"] = -0.04
c.update({"head_pan": 0.6, "head_tilt": 0.5, "lift": 0.3})
step(12.0, c)
res["arm_tracking_rad"] = {k: round(qpos("left_arm_joint%s" % k[1]) - v, 4) for k, v in ARM.items()}
res["gripper"] = {"cmd": -0.04, "left": round(qpos("left_arm_gripper_joint"), 4), "left_mimic": round(qpos("left_arm_gripper_mimic_joint"), 4)}
res["head"] = {"pan": round(qpos("head_pan_joint"), 3), "tilt": round(qpos("head_tilt_joint"), 3)}
res["lift_hold_0.3"] = round(qpos("lift_joint"), 4)
p, yaw, tilt = pose()
res["final_tilt_deg"] = round(tilt, 2)
res["unstable"] = bool(np.any(~np.isfinite(d.qpos)))
print(json.dumps(res, indent=1))

# ------------------------------------------------------------------ rendering
try:
    W, H = 1280, 960
    r = mujoco.Renderer(m, H, W)
    import PIL.Image as Image
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = base
    shots = {}
    opt = mujoco.MjvOption()
    for name, (az, el, dist, look) in {"view_front_left": (145, -18, 2.6, 0.75), "view_rear_right": (-40, -15, 2.6, 0.75),
                                         "view_side": (90, -5, 2.8, 0.75)}.items():
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = d.xpos[base] + np.array([0, 0, look - 0.1])
        r.update_scene(d, cam, opt)
        Image.fromarray(r.render()).save(os.path.join(OUT, name + ".png"))
    opt.geomgroup[2] = 0; opt.geomgroup[3] = 1
    cam.azimuth, cam.elevation, cam.distance = 145, -18, 2.6
    r.update_scene(d, cam, opt)
    Image.fromarray(r.render()).save(os.path.join(OUT, "view_collision.png"))
    r.close()
    r = mujoco.Renderer(m, 600, 960)
    for cname in ["head_camera", "pantilt_camera", "nav_camera"]:
        r.update_scene(d, cname)
        Image.fromarray(r.render()).save(os.path.join(OUT, cname + "_rgb.png"))
        r.enable_depth_rendering()
        r.update_scene(d, cname)
        dep = r.render()
        r.disable_depth_rendering()
        dd = np.clip(dep, 0, 4.0) / 4.0
        Image.fromarray((255 * (1 - dd)).astype(np.uint8)).save(os.path.join(OUT, cname + "_depth.png"))
        print(cname, "depth min/median/max m:", round(float(dep.min()), 3), round(float(np.median(dep)), 3), round(float(dep.max()), 2))
    r.close()
except Exception as e:
    print("render failed:", repr(e))
