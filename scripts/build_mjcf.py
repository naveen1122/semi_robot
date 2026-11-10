"""Converts robot_v1.urdf to a MuJoCo model (robot_v1.xml + scene.xml) with actuators, cameras and mimic constraints."""
import os, re, sys
import xml.etree.ElementTree as ET
import mujoco

PKG = sys.argv[1]
src = open(os.path.join(PKG, "urdf", "robot_v1.urdf")).read()
src = re.sub(r"\n  <gazebo reference=.*?</gazebo>", "", src, flags=re.S)
mesh_abs = os.path.join(PKG, "meshes", "dk1")
src = src.replace('<robot name="robot_v1">',
                  '<robot name="robot_v1">\n  <mujoco><compiler meshdir="%s" balanceinertia="true" discardvisual="false" '
                  'fusestatic="false" strippath="true"/></mujoco>' % mesh_abs, 1)
tmp = os.path.join(PKG, "mjcf", "_tmp.urdf")
os.makedirs(os.path.dirname(tmp), exist_ok=True)
open(tmp, "w").write(src)
m = mujoco.MjModel.from_xml_path(tmp)
raw = os.path.join(PKG, "mjcf", "_raw.xml")
mujoco.mj_saveLastXML(raw, m)
os.remove(tmp)

t = ET.parse(raw)
root = t.getroot()
os.remove(raw)
root.set("model", "robot_v1")
comp = root.find("compiler")
comp.set("meshdir", "../meshes/dk1")
comp.set("angle", "radian")
opt = root.find("option")
if opt is None:
    opt = ET.SubElement(root, "option")
opt.set("timestep", "0.002")
opt.set("integrator", "implicitfast")
opt.set("cone", "elliptic")
opt.set("impratio", "10")

wb = root.find("worldbody")
base = wb.find("body")
# floating base
fj = ET.Element("freejoint", {"name": "root"})
base.insert(0, fj)

bodies = {b.get("name"): b for b in root.iter("body")}

# visual geoms (those carrying rgba from URDF materials and no collision role) -> group 2, no contacts
for g in root.iter("geom"):
    if g.get("contype") == "0" and g.get("conaffinity") == "0":
        g.set("group", "2")
    else:
        g.set("group", "3")
        g.set("rgba", "0.5 0.5 0.9 0.3")
        g.set("solref", "0.004 1")
        g.set("solimp", "0.95 0.99 0.001")
        if g.get("name", "").startswith(("left_wheel", "right_wheel")) or True:
            pass

# cameras: MuJoCo cameras look along -z with +y up; the camera frames are x forward, z up
for cam, fovy in [("head_camera", 80), ("pantilt_camera", 80), ("nav_camera", 80)]:
    b = bodies[cam + "_left_camera_frame"]
    ET.SubElement(b, "camera", {"name": cam, "pos": "0 0 0", "xyaxes": "0 -1 0 0 0 1", "fovy": str(fovy)})
for side in ["left", "right"]:
    b = bodies[side + "_arm_wrist_camera"]
    ET.SubElement(b, "camera", {"name": side + "_wrist_camera", "pos": "0 0 0", "xyaxes": "0 -1 0 0 0 1", "fovy": "90"})
ET.SubElement(bodies["imu_link"], "site", {"name": "imu", "size": "0.01"})
ET.SubElement(bodies["lidar_link"], "site", {"name": "lidar", "size": "0.01"})

# joint properties: armature (reflected rotor inertia of geared motors) for numerical stability
JP = {}
for side in ["left", "right"]:
    for i in range(1, 4): JP["%s_arm_joint%d" % (side, i)] = 0.05
    for i in range(4, 7): JP["%s_arm_joint%d" % (side, i)] = 0.01
    JP[side + "_arm_gripper_joint"] = 0.05
    JP[side + "_arm_gripper_mimic_joint"] = 0.05
    JP[side + "_wheel_joint"] = 0.02
JP["lift_joint"] = 2.0
JP["lift_stage2_joint"] = 1.0
JP["head_pan_joint"] = 0.01
JP["head_tilt_joint"] = 0.01
for j in root.iter("joint"):
    if j.get("name") in JP:
        j.set("armature", str(JP[j.get("name")]))

sens = ET.SubElement(root, "sensor")
ET.SubElement(sens, "accelerometer", {"name": "imu_acc", "site": "imu"})
ET.SubElement(sens, "gyro", {"name": "imu_gyro", "site": "imu"})
ET.SubElement(sens, "framequat", {"name": "imu_quat", "objtype": "site", "objname": "imu"})
ET.SubElement(sens, "rangefinder", {"name": "lidar_front", "site": "lidar"})

eq = ET.SubElement(root, "equality")
ET.SubElement(eq, "joint", {"name": "lift_stage2_mimic", "joint1": "lift_stage2_joint", "joint2": "lift_joint", "polycoef": "0 0.5 0 0 0"})
for side in ["left", "right"]:
    ET.SubElement(eq, "joint", {"name": side + "_gripper_mimic", "joint1": side + "_arm_gripper_mimic_joint",
                                "joint2": side + "_arm_gripper_joint", "polycoef": "0 1 0 0 0"})

act = ET.SubElement(root, "actuator")
for side in ["left", "right"]:
    ET.SubElement(act, "velocity", {"name": side + "_wheel", "joint": side + "_wheel_joint", "kv": "50",
                                    "ctrlrange": "-20.94 20.94", "forcerange": "-16 16"})
ET.SubElement(act, "position", {"name": "lift", "joint": "lift_joint", "kp": "100000", "kv": "10000",
                                "ctrlrange": "0 0.65", "forcerange": "-600 600"})
ET.SubElement(act, "position", {"name": "head_pan", "joint": "head_pan_joint", "kp": "20", "kv": "1",
                                "ctrlrange": "-1.5708 1.5708", "forcerange": "-4.1 4.1"})
ET.SubElement(act, "position", {"name": "head_tilt", "joint": "head_tilt_joint", "kp": "20", "kv": "1",
                                "ctrlrange": "-0.5236 0.8727", "forcerange": "-4.1 4.1"})
lims = {}
for j in root.iter("joint"):
    if j.get("range"):
        lims[j.get("name")] = j.get("range")
for side in ["left", "right"]:
    for i in range(1, 7):
        jn = "%s_arm_joint%d" % (side, i)
        kp, fr = ("150", "27") if i <= 3 else ("40", "7")
        ET.SubElement(act, "position", {"name": "%s_arm_j%d" % (side, i), "joint": jn, "kp": kp, "kv": str(float(kp) / 10),
                                        "ctrlrange": lims[jn], "forcerange": "-%s %s" % (fr, fr)})
    ET.SubElement(act, "position", {"name": side + "_gripper", "joint": side + "_arm_gripper_joint", "kp": "2000", "kv": "40",
                                    "ctrlrange": lims[side + "_arm_gripper_joint"], "forcerange": "-114 114"})

# contact excludes for arm links that touch in the folded zero pose
cx = ET.SubElement(root, "contact")
for side in ["left", "right"]:
    for a, b in [("arm_link2", "arm_link4"), ("arm_link2", "arm_link5"), ("arm_link2", "arm_link6"), ("arm_link3", "arm_link5"),
                 ("arm_link1", "arm_link3"), ("arm_base", "arm_link2")]:
        ET.SubElement(cx, "exclude", {"body1": "%s_%s" % (side, a), "body2": "%s_%s" % (side, b)})
ET.indent(t)
t.write(os.path.join(PKG, "mjcf", "robot_v1.xml"))
open(os.path.join(PKG, "mjcf", "scene.xml"), "w").write("""<mujoco model="robot_v1_scene">
  <include file="robot_v1.xml"/>
  <statistic center="0 0 0.8" extent="2.2"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="150" elevation="-20" offwidth="1920" offheight="1200"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.88 0.92" rgb2="0.55 0.6 0.68" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.72 0.74 0.76" rgb2="0.62 0.64 0.66"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="4 4" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.005 0.0001"/>
    <body name="table" pos="0.75 0 0.375">
      <geom type="box" size="0.3 0.5 0.375" rgba="0.55 0.42 0.3 1" mass="25"/>
    </body>
    <body name="box_obj" pos="0.6 0.1 0.78">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.03" rgba="0.9 0.3 0.2 1" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
""")
open(os.path.join(PKG, "mjcf", "scene_empty.xml"), "w").write("""<mujoco model="robot_v1_empty">
  <include file="robot_v1.xml"/>
  <statistic center="0 0 0.8" extent="2.2"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="150" elevation="-20" offwidth="1920" offheight="1200"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.88 0.92" rgb2="0.55 0.6 0.68" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.72 0.74 0.76" rgb2="0.62 0.64 0.66"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="4 4" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.005 0.0001"/>
  </worldbody>
</mujoco>
""")
mm = mujoco.MjModel.from_xml_path(os.path.join(PKG, "mjcf", "scene.xml"))
print("MJCF ok: bodies", mm.nbody, "joints", mm.njnt, "actuators", mm.nu, "cameras", mm.ncam, "mass %.2f" % sum(mm.body_mass))
