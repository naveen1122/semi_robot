"""Builds the Robot V1 URDF package from the Onshape data (cad_data.py) and the TRLC-DK1 follower URDF."""
import math, os, shutil, sys, json
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cad_data import MP, NAMES, VIS

DK1 = sys.argv[1]                      # path to trlc-dk1/urdf/follower (vendor files, read only)
PKG = sys.argv[2]                      # output package folder
RIGID = (sys.argv[3] if len(sys.argv) > 3 else "rigid") == "rigid"
PI = math.pi

# ------------------------------------------------------------------ frames
# CAD: +X robot left, -Y robot front, +Z up, mm.  URDF (REP-103): +x front, +y left, +z up, m.
# base_link sits on the drive-wheel axle line, 0.100 m above the floor; base_footprint on the floor.
R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
def c2r(p_mm):
    x, y, z = p_mm
    return np.array([-y, x, z - 100.0]) / 1000.0
def c2r_m(p):
    return np.array([-p[1], p[0], p[2] - 0.1])
def I_c2r(I9):
    I = np.array(I9, float).reshape(3, 3)
    return R @ I @ R.T

ORIG = {"base": np.zeros(3)}
PIV = {"L": np.array([-0.120, 0.206, 0.025]), "R": np.array([-0.120, -0.206, 0.025])}
AXLE = {"L": np.array([0.0, 0.2501, 0.0]), "R": np.array([0.0, -0.2501, 0.0])}
ORIG["susp_L"], ORIG["susp_R"] = PIV["L"], PIV["R"]
ORIG["wheel_L"], ORIG["wheel_R"] = AXLE["L"], AXLE["R"]
CAST = {}
for tag in ["FL", "FR", "RL", "RR"]:
    cx = 216.0 if tag[1] == "L" else -216.0
    cy = -216.0 if tag[0] == "F" else 216.0
    sy = 1.0 if cy > 0 else -1.0
    sw = c2r((cx, cy, 100.0))
    wh = c2r((cx, cy - sy * 28.0, 37.5))
    CAST[tag] = (sw, wh)
    ORIG["caster_%s_swivel" % tag], ORIG["caster_%s_wheel" % tag] = sw, wh
COL = np.array([-0.030, 0.0, 0.0])               # lift column axis
ORIG["lift_stage2"] = COL.copy()
TORSO = np.array([-0.030, 0.0, 0.660])           # torso frame at lift = 0 (CAD z 760)
ORIG["torso"] = TORSO
ORIG["head_cam_upper"] = c2r((0, -128, 1461))

LINKNAME = {"base": "base_link", "susp_L": "left_suspension_link", "susp_R": "right_suspension_link",
            "wheel_L": "left_wheel_link", "wheel_R": "right_wheel_link", "lift_stage2": "lift_stage2_link",
            "torso": "torso_link", "head_cam_upper": "head_camera_link"}
for t in ["FL", "FR", "RL", "RR"]:
    LINKNAME["caster_%s_swivel" % t] = "caster_%s_swivel_link" % t.lower()
    LINKNAME["caster_%s_wheel" % t] = "caster_%s_wheel_link" % t.lower()

# ------------------------------------------------------------------ xml helpers
def f(v):
    return "%.6g" % (0.0 if abs(v) < 1e-12 else v)
def v3(a):
    return " ".join(f(x) for x in a)

MATS = {}
def mat(rgb):
    key = tuple(int(round(c * 100)) for c in rgb[:3])
    if key not in MATS:
        MATS[key] = "c_%02d%02d%02d" % key
    return MATS[key]

class Link:
    def __init__(self, name):
        self.name, self.vis, self.col, self.inertial = name, [], [], None
    def box(self, c, size, rgb, collide=False, rpy=(0, 0, 0), vis=True):
        g = '<box size="%s"/>' % v3(size)
        if vis: self.vis.append((c, rpy, g, rgb))
        if collide: self.col.append((c, rpy, g))
    def cyl(self, c, r, l, axis, rgb, collide=False, vis=True):
        rpy = {"x": (0, PI / 2, 0), "y": (PI / 2, 0, 0), "z": (0, 0, 0)}[axis]
        g = '<cylinder radius="%s" length="%s"/>' % (f(r), f(l))
        if vis: self.vis.append((c, rpy, g, rgb))
        if collide: self.col.append((c, rpy, g))
    def mesh(self, fn, rgb, rpy=(0, 0, 0), c=(0, 0, 0), collide_fn=None):
        if fn: self.vis.append((c, rpy, '<mesh filename="%s"/>' % fn, rgb))
        if collide_fn: self.col.append((c, rpy, '<mesh filename="%s"/>' % collide_fn))
    def xml(self):
        s = ['  <link name="%s">' % self.name]
        if self.inertial:
            m, com, I = self.inertial
            s.append('    <inertial>\n      <origin xyz="%s" rpy="0 0 0"/>\n      <mass value="%s"/>\n'
                     '      <inertia ixx="%s" ixy="%s" ixz="%s" iyy="%s" iyz="%s" izz="%s"/>\n    </inertial>'
                     % (v3(com), f(m), f(I[0, 0]), f(I[0, 1]), f(I[0, 2]), f(I[1, 1]), f(I[1, 2]), f(I[2, 2])))
        for c, rpy, g, rgb in self.vis:
            s.append('    <visual>\n      <origin xyz="%s" rpy="%s"/>\n      <geometry>%s</geometry>\n'
                     '      <material name="%s"/>\n    </visual>' % (v3(c), v3(rpy), g, mat(rgb)))
        for c, rpy, g in self.col:
            s.append('    <collision>\n      <origin xyz="%s" rpy="%s"/>\n      <geometry>%s</geometry>\n    </collision>'
                     % (v3(c), v3(rpy), g))
        s.append('  </link>')
        return "\n".join(s)

def box_inertia(m, sx, sy, sz):
    return np.diag([m * (sy * sy + sz * sz) / 12, m * (sx * sx + sz * sz) / 12, m * (sx * sx + sy * sy) / 12])

JOINTS = []
def joint(name, typ, parent, child, xyz, rpy=(0, 0, 0), axis=None, limit=None, dyn=None, mimic=None, comment=None):
    s = []
    if comment: s.append('  <!-- %s -->' % comment)
    s.append('  <joint name="%s" type="%s">' % (name, typ))
    s.append('    <origin xyz="%s" rpy="%s"/>' % (v3(xyz), v3(rpy)))
    s.append('    <parent link="%s"/>\n    <child link="%s"/>' % (parent, child))
    if axis is not None: s.append('    <axis xyz="%s"/>' % v3(axis))
    if limit is not None:
        lo, hi, eff, vel = limit
        if typ == "continuous":
            s.append('    <limit effort="%s" velocity="%s"/>' % (f(eff), f(vel)))
        else:
            s.append('    <limit lower="%s" upper="%s" effort="%s" velocity="%s"/>' % (f(lo), f(hi), f(eff), f(vel)))
    if dyn is not None: s.append('    <dynamics damping="%s" friction="%s"/>' % (f(dyn[0]), f(dyn[1])))
    if mimic is not None: s.append('    <mimic joint="%s" multiplier="%s" offset="0"/>' % (mimic[0], f(mimic[1])))
    s.append('  </joint>')
    JOINTS.append("\n".join(s))

# ------------------------------------------------------------------ body links from CAD
links = {k: Link(v) for k, v in LINKNAME.items()}
for k, lk in links.items():
    mp = MP[k]
    com = c2r_m(mp["c"]) - ORIG[k]
    lk.inertial = [mp["m"], com, I_c2r(mp["I"])]

def add_point_mass(lk, m, p, size):
    """Adds a box-shaped mass (m at p, link frame) to a link's inertial (parallel-axis theorem)."""
    m0, c0, I0 = lk.inertial
    M = m0 + m
    C = (m0 * c0 + m * np.asarray(p)) / M
    def shift(I, mm, d):
        d = np.asarray(d)
        return I + mm * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    I = shift(I0, m0, c0 - C) + shift(box_inertia(m, *size), m, np.asarray(p) - C)
    lk.inertial = [M, C, I]

# release-3 electronics not yet placed in CAD (arm 24 V supply ~1.0 kg, USB hub, RS485, fan, fuses, shunt, servo interface)
add_point_mass(links["base"], 1.8, (-0.15, 0.0, 0.17), (0.25, 0.30, 0.08))

# casing envelopes: parts fully inside are not drawn (they cannot be seen from outside)
ENV = [((-292, -292, 99), (292, 292, 298.5)), ((-294.6, -77.5, 268), (294.6, 294.6, 442.5)),
       ((-42.5, -31, 655), (145.5, 91, 776.5))]
CASING = {173, 174, 175, 176, 177, 178, 179, 182, 184, 185, 3}
def hidden(n, a, b):
    if n in CASING: return False
    return any(all(a[i] >= lo[i] and b[i] <= hi[i] for i in range(3)) for lo, hi in ENV)

def cbox(lk, key, a, b, rgb, collide=False):
    a, b = np.array(a, float), np.array(b, float)
    c = c2r((a + b) / 2) - ORIG[key]
    d = b - a
    lk.box(c, (d[1] / 1000, d[0] / 1000, d[2] / 1000), rgb, collide)

GREY = (0.69, 0.71, 0.74); DARK = (0.26, 0.27, 0.29); BLACK = (0.09, 0.09, 0.10); WHITE = (0.93, 0.93, 0.95)
STEEL = (0.77, 0.78, 0.80); RED = (0.80, 0.12, 0.12); YEL = (0.94, 0.77, 0.08)

for line in VIS.strip().splitlines():
    t = line.split()
    key, n = t[0], int(t[1])
    a, b = [float(x) for x in t[2:5]], [float(x) for x in t[5:8]]
    rgb = tuple(int(x) / 100.0 for x in t[8:11])
    if key not in links: continue
    lk = links[key]
    if n == 171 or n in (24, 34):            # cable chain (flexible), gas springs (inside the base)
        continue
    if hidden(n, a, b):
        continue
    if n == 3:                                 # belly pan with the two wheel wells
        for x0, y0, x1, y1 in [(-199, -296, 199, 296), (199, -296, 296, -112), (-296, -296, -199, -112),
                               (199, 156, 296, 296), (-296, 156, -199, 296), (213, 110, 296, 156), (-296, 110, -213, 156)]:
            cbox(lk, key, (x0, y0, 100), (x1, y1, 106), rgb)
        continue
    if n in (4, 26):                           # tyre
        s = 1 if n == 26 else -1
        lk.cyl(c2r((s * 250.1, 0, 100)) - ORIG[key], 0.100, 0.0494, "y", BLACK, collide=True)
        continue
    if n in (5, 27):                           # hub motor: rotor, outer cover, boss and shaft
        s = 1 if n == 27 else -1
        for x0, x1, r in [(226.4, 273.8, 0.078), (273.8, 281.2, 0.055), (212, 226.4, 0.022), (174, 212, 0.007)]:
            lk.cyl(c2r((s * (x0 + x1) / 2, 0, 100)) - ORIG[key], r, (x1 - x0) / 1000, "y", DARK)
        continue
    if n in (9, 29):
        lk.cyl(c2r(((a[0] + b[0]) / 2, 120, 125)) - ORIG[key], 0.006, (b[0] - a[0]) / 1000, "y", STEEL)
        continue
    if n in (37, 41, 45, 49):                  # caster wheel
        lk.cyl(np.zeros(3), 0.0375, 0.025, "y", DARK, collide=True)
        continue
    if n in (38, 42, 46, 50):                  # caster axle bolt
        lk.cyl(np.zeros(3), 0.004, 0.038, "y", STEEL)
        continue
    if n in (36, 40, 44, 48):                  # caster swivel head, crown and fork legs (link frame on the swivel axis)
        tag = NAMES[n][-2:]
        sw, wh = CAST[tag]
        dx = wh[0] - sw[0]
        lk.cyl((0, 0, -0.008), 0.021, 0.012, "z", STEEL)
        lk.box((dx * 0.55, 0, -0.0153), (abs(dx) + 0.03, 0.033, 0.0035), STEEL)
        for sgn in (-1, 1):
            lk.box((dx * 0.9, sgn * 0.01525, -0.043), (0.026, 0.0025, 0.056), STEEL)
        continue
    if n == 156:
        lk.cyl(c2r((215, 105, 483)) - ORIG[key], 0.024, 0.082, "z", BLACK)
        lk.col.append((c2r((215, 105, 483)) - ORIG[key], (0, 0, 0), '<cylinder radius="0.024" length="0.082"/>'))
        continue
    if n == 82:
        lk.cyl(c2r((0, 30, 1561)) - ORIG[key], 0.020, 0.040, "z", RED, collide=True)
        continue
    if n == 168:
        lk.cyl(c2r((0, 226, 764)) - ORIG[key], 0.020, 0.040, "x", RED, collide=True)
        continue
    if n == 88:                                # inner tube and motor housing of the column
        cbox(lk, key, (-28.5, 1.5, 142), (28.5, 58.5, 661.5), DARK)
        cbox(lk, key, (-40, -22.5, 661.5), (143, 82.5, 712), DARK)
        continue
    collide = n in {65, 66, 63, 64, 71, 73, 74, 177, 178, 179, 81, 157, 164, 167, 161, 76}
    cbox(lk, key, a, b, rgb, collide)

# base collision: three bands that keep clear of the wheel wells, the electronics hood, column and rear-post stack
for a, b in [((-292, -292, 100), (292, -112, 298)), ((-292, 112, 100), (292, 292, 298)), ((-199, -112, 100), (199, 112, 298)),
             ((-294, -77, 298), (294, 294, 442)), ((-35, -5, 442), (35, 65, 640))]:
    a, b = np.array(a, float), np.array(b, float)
    c = c2r((a + b) / 2); d = b - a
    links["base"].col.append((c, (0, 0, 0), '<box size="%s"/>' % v3((d[1] / 1000, d[0] / 1000, d[2] / 1000))))

# ------------------------------------------------------------------ extra links not yet in CAD (release 3)
def sensor_link(name, m, size, c=(0, 0, 0), rgb=DARK, collide=True):
    lk = Link(name)
    lk.inertial = [m, np.array(c, float), box_inertia(m, *size)]
    lk.box(c, size, rgb, collide)
    links[name] = lk
    return lk

ZEDX = (0.0367, 0.1634, 0.0318); ZEDXM = (0.0367, 0.0936, 0.0318)
links["head_cam_upper"].vis = []; links["head_cam_upper"].col = []
links["head_cam_upper"].inertial = [0.239, np.zeros(3), box_inertia(0.239, *ZEDX)]
links["head_cam_upper"].box((0, 0, 0), ZEDX, DARK, True)
sensor_link("nav_camera_link", 0.151, ZEDXM)
sensor_link("pantilt_camera_link", 0.239, ZEDX)
sensor_link("imu_link", 0.05, (0.055, 0.037, 0.024), rgb=(0.15, 0.35, 0.75), collide=False)
lid = Link("lidar_link")
lid.inertial = [0.19, np.array((0, 0, -0.0075)), box_inertia(0.19, 0.077, 0.077, 0.039)]
lid.cyl((0, 0, -0.0075), 0.0385, 0.039, "z", BLACK, collide=True)
links["lidar_link"] = lid
br = sensor_link("head_pantilt_bracket_link", 0.165, (0.03, 0.03, 0.03), collide=False)
br.vis = []
br.box((-0.044, 0, -0.028), (0.108, 0.05, 0.003), GREY)          # base plate under the pan servo
br.box((-0.0965, 0, 0.0), (0.003, 0.05, 0.11), GREY)              # plate against the head plate
br.box((0, 0, -0.0233), (0.034, 0.0285, 0.0465), BLACK)           # pan servo XM430 body
br.inertial = [0.165, np.array((-0.02, 0, -0.02)), box_inertia(0.165, 0.11, 0.05, 0.06)]
pan = sensor_link("head_pan_link", 0.11, (0.034, 0.0285, 0.0465), c=(0, 0, 0.026), rgb=BLACK, collide=False)
pan.box((0, 0, 0.0015), (0.03, 0.03, 0.003), GREY)
tilt = sensor_link("head_tilt_link", 0.03, (0.06, 0.07, 0.004), c=(0.01, 0, -0.018), rgb=GREY, collide=False)

# ------------------------------------------------------------------ arms (TRLC-DK1 follower, Apache-2.0)
vt = ET.parse(os.path.join(DK1, "TRLC-DK1-Follower.urdf")).getroot()
ARM_LINK = {"base_link": "arm_base", "link1-2": "arm_link1", "link2-3": "arm_link2", "link3-4": "arm_link3",
            "link4-5": "arm_link4", "link5-6": "arm_link5", "link6-7": "arm_link6", "finger_left": "arm_finger_left",
            "finger_right": "arm_finger_right", "tool0": "arm_tool0", "camera": "arm_wrist_camera"}
MESHFILE = {"base_link": "base", "link1-2": "link1", "link2-3": "link2", "link3-4": "link3", "link4-5": "link4",
            "link5-6": "link5", "link6-7": "link6", "finger_left": "finger_left", "finger_right": "finger_right"}
# realistic actuator data: DAMIAO DM-J4340-2EC (joints 1-3) and DM-J4310-2EC (joints 4-6, gripper) at 24 V
LIM = {"joint1": (27.0, 5.49), "joint2": (27.0, 5.49), "joint3": (27.0, 5.49),
       "joint4": (7.0, 20.94), "joint5": (7.0, 20.94), "joint6": (7.0, 20.94)}
mdir = os.path.join(PKG, "meshes", "dk1")
os.makedirs(mdir, exist_ok=True)
VISMESH = {}
for vname, short in MESHFILE.items():
    sc = trimesh.load(os.path.join(DK1, "meshes", "visual", vname.replace("_link", "_link") + ".glb"))
    groups = {}
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname].copy()
        g.apply_transform(T)
        col = tuple(int(c) for c in g.visual.material.baseColorFactor[:3]) if hasattr(g.visual, "material") and \
            getattr(g.visual.material, "baseColorFactor", None) is not None else (60, 60, 60)
        groups.setdefault(col, []).append(trimesh.Trimesh(g.vertices, g.faces, process=False))
    files = []
    i = 0
    for col, parts in sorted(groups.items()):
        m = trimesh.util.concatenate(parts)
        if len(m.faces) < 50 or abs(m.volume) < 1e-9:
            continue
        i += 1
        fn = "dk1_%s_v%d.stl" % (short, i - 1)
        m.export(os.path.join(mdir, fn))
        lin = [min(1.0, (c / 255.0) ** (1 / 2.2) * 1.0) for c in col]     # glTF factors are linear; convert to sRGB-ish
        files.append((fn, (lin[0], lin[1], lin[2])))
    VISMESH[vname] = files
    shutil.copy(os.path.join(DK1, "meshes", "collision", vname + ".stl"), os.path.join(mdir, "dk1_%s_col.stl" % short))
for fn in ["LICENSE"]:
    shutil.copy(os.path.join(DK1, "..", "..", fn), os.path.join(PKG, "meshes", "dk1", "LICENSE-TRLC-DK1.txt"))

def arm(side, xyz):
    p = side + "_"
    joint(p + "arm_mount_joint", "fixed", "torso_link", p + "arm_base", xyz,
          comment="%s arm (TRLC-DK1 follower) on its 10 mm mount plate, 4x M5 on 100 x 40" % side)
    for l in vt.findall("link"):
        n = l.get("name")
        lk = Link(p + ARM_LINK[n])
        ine = l.find("inertial")
        if ine is not None:
            o = ine.find("origin"); I = ine.find("inertia")
            Im = np.array([[float(I.get("ixx")), float(I.get("ixy")), float(I.get("ixz"))],
                           [float(I.get("ixy")), float(I.get("iyy")), float(I.get("iyz"))],
                           [float(I.get("ixz")), float(I.get("iyz")), float(I.get("izz"))]])
            lk.inertial = [float(ine.find("mass").get("value")), np.array([float(x) for x in o.get("xyz").split()]), Im]
        if n in VISMESH:
            for fn, rgb in VISMESH[n]:
                lk.mesh("../meshes/dk1/" + fn, rgb)
            lk.col.append(((0, 0, 0), (0, 0, 0), '<mesh filename="../meshes/dk1/dk1_%s_col.stl"/>' % MESHFILE[n]))
        links[p + ARM_LINK[n]] = lk
    for j in vt.findall("joint"):
        jn, jt = j.get("name"), j.get("type")
        o = j.find("origin")
        xyz = [float(x) for x in o.get("xyz").split()]
        rpy = [float(x) for x in o.get("rpy").split()]
        par, chi = ARM_LINK[j.find("parent").get("link")], ARM_LINK[j.find("child").get("link")]
        ax = j.find("axis"); axis = [float(x) for x in ax.get("xyz").split()] if ax is not None else None
        lim = j.find("limit")
        if jt == "fixed":
            joint(p + {"gripper_tool0": "arm_tool0_joint", "camera_joint": "arm_wrist_camera_joint"}[jn], "fixed",
                  p + par, p + chi, xyz, rpy)
            continue
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        if jn.startswith("joint"):
            eff, vel = LIM[jn]
            joint(p + "arm_" + jn, jt, p + par, p + chi, xyz, rpy, axis, (lo, hi, eff, vel), (0.05, 0.05))
        elif jn == "gripper_left":
            joint(p + "arm_gripper_joint", jt, p + par, p + chi, xyz, rpy, axis, (lo, hi, 114.0, 0.183), (0.5, 0.5),
                  comment="gripper: one DM4310 on a 8.75 mm pinion drives both fingers (114 N, 0.183 m/s)")
        else:
            joint(p + "arm_gripper_mimic_joint", jt, p + par, p + chi, xyz, rpy, axis, (lo, hi, 114.0, 0.183), (0.5, 0.5),
                  mimic=(p + "arm_gripper_joint", 1.0))

# ------------------------------------------------------------------ joints of the body
joint("base_footprint_joint", "fixed", "base_footprint", "base_link", (0, 0, 0.100))
for s, side in [("L", "left"), ("R", "right")]:
    if RIGID:
        joint("%s_suspension_joint" % side, "fixed", "base_link", "%s_suspension_link" % side, PIV[s],
              comment="trailing arm held at ride height (use robot_v1_suspension.urdf for the sprung version)")
    else:
        joint("%s_suspension_joint" % side, "revolute", "base_link", "%s_suspension_link" % side, PIV[s], axis=(0, 1, 0),
              limit=(-0.11, 0.11, 200.0, 10.0), dyn=(2.0, 0.0),
              comment="trailing arm on a 12 mm shaft in two KP001 bearings; Suspa gas spring 200 N (see README for the spring model)")
    joint("%s_wheel_joint" % side, "continuous", "%s_suspension_link" % side, "%s_wheel_link" % side, AXLE[s] - PIV[s],
          axis=(0, 1, 0), limit=(0, 0, 16.0, 20.94), dyn=(0.02, 0.05),
          comment="ZLTECH ZLLG80ASM250-L hub motor, 200 mm tyre: 5.5 N.m rated / 16 N.m peak, 200 rpm (20.94 rad/s)")
for t in ["FL", "FR", "RL", "RR"]:
    sw, wh = CAST[t]
    tl = t.lower()
    joint("caster_%s_swivel_joint" % tl, "continuous", "base_link", "caster_%s_swivel_link" % tl, sw, axis=(0, 0, 1),
          limit=(0, 0, 0.0, 30.0), dyn=(0.002, 0.01),
          comment="Blickle LPA-PATH 75KF swivel caster: 75 x 25 wheel, 28 mm trail, passive" if t == "FL" else None)
    joint("caster_%s_wheel_joint" % tl, "continuous", "caster_%s_swivel_link" % tl, "caster_%s_wheel_link" % tl,
          wh - sw, axis=(0, 1, 0), limit=(0, 0, 0.0, 60.0), dyn=(0.001, 0.005))
joint("lift_joint", "prismatic", "base_link", "torso_link", TORSO, axis=(0, 0, 1), limit=(0.0, 0.650, 600.0, 0.039),
      dyn=(500.0, 50.0),
      comment="TiMOTION TL4 3-stage lifting column: 650 mm stroke, 600 N push, 25-39 mm/s under load (55 mm/s no load), self-locking")
joint("lift_stage2_joint", "prismatic", "base_link", "lift_stage2_link", COL, axis=(0, 0, 1), limit=(0.0, 0.325, 600.0, 0.0195),
      mimic=("lift_joint", 0.5), comment="middle tube of the column moves half the stroke")
joint("head_camera_joint", "fixed", "torso_link", "head_camera_link", ORIG["head_cam_upper"] - TORSO, (0, math.radians(10), 0),
      comment="top head camera (ZED X, fixed, 10 deg down)")
PAN = c2r((0, -160, 1340)) - TORSO
joint("head_pantilt_bracket_joint", "fixed", "torso_link", "head_pantilt_bracket_link", PAN,
      comment="pan-tilt for the second head camera: placed here provisionally until it is drawn in CAD")
joint("head_pan_joint", "revolute", "head_pantilt_bracket_link", "head_pan_link", (0, 0, 0), axis=(0, 0, 1),
      limit=(-1.5708, 1.5708, 4.1, 4.8), dyn=(0.1, 0.05), comment="DYNAMIXEL XM430-W350-T at 12 V: 4.1 N.m stall, 46 rpm")
joint("head_tilt_joint", "revolute", "head_pan_link", "head_tilt_link", (0, 0, 0.045), axis=(0, 1, 0),
      limit=(-0.5236, 0.8727, 4.1, 4.8), dyn=(0.1, 0.05), comment="positive = look down")
joint("pantilt_camera_joint", "fixed", "head_tilt_link", "pantilt_camera_link", (0.030, 0, 0))
joint("nav_camera_joint", "fixed", "base_link", "nav_camera_link", (0.314, 0, 0.150), (0, math.radians(15), 0),
      comment="navigation camera (ZED X Mini) on the base front panel, 15 deg down: provisional until drawn in CAD")
joint("imu_joint", "fixed", "base_link", "imu_link", (-0.020, -0.100, 0.018),
      comment="WitMotion HWT905 on the belly pan near the drive axle: provisional until drawn in CAD")
joint("lidar_joint", "fixed", "base_link", "lidar_link", (0.230, 0, 0.225),
      comment="RPLIDAR S2 (optional) on the front of the battery-bay lid; frame at the scan plane: provisional")
arm("left", (0.097, 0.280, 0.070))
arm("right", (0.097, -0.280, 0.070))

# stereo camera optical frames (ZED convention: left camera is the depth reference)
CAMS = {"head_camera": ("head_camera_link", 0.12), "pantilt_camera": ("pantilt_camera_link", 0.12),
        "nav_camera": ("nav_camera_link", 0.05)}
for cname, (parent, base) in CAMS.items():
    for side, sgn in [("left", 1), ("right", -1)]:
        fr = "%s_%s_camera_frame" % (cname, side)
        links[fr] = Link(fr)
        joint(fr + "_joint", "fixed", parent, fr, (0.0, sgn * base / 2, 0.0))
        op = "%s_%s_camera_optical_frame" % (cname, side)
        links[op] = Link(op)
        joint(op + "_joint", "fixed", fr, op, (0, 0, 0), (-PI / 2, 0, -PI / 2))
links["base_footprint"] = Link("base_footprint")

# ------------------------------------------------------------------ gazebo extensions
GZ = []
def gz_cam(cname, w, h, hfov, near, far, rate):
    GZ.append("""  <gazebo reference="%s_left_camera_frame">
    <sensor name="%s" type="rgbd_camera">
      <always_on>true</always_on>
      <update_rate>%d</update_rate>
      <topic>%s</topic>
      <camera>
        <horizontal_fov>%s</horizontal_fov>
        <image><width>%d</width><height>%d</height><format>R8G8B8</format></image>
        <clip><near>0.05</near><far>%s</far></clip>
        <depth_camera><clip><near>%s</near><far>%s</far></clip></depth_camera>
        <optical_frame_id>%s_left_camera_optical_frame</optical_frame_id>
      </camera>
    </sensor>
  </gazebo>""" % (cname, cname, rate, cname, f(hfov), w, h, f(far), f(near), f(far), cname))
gz_cam("head_camera", 960, 600, math.radians(110), 0.3, 20.0, 30)
gz_cam("pantilt_camera", 960, 600, math.radians(110), 0.3, 20.0, 30)
gz_cam("nav_camera", 960, 600, math.radians(110), 0.1, 8.0, 30)
GZ.append("""  <gazebo reference="imu_link">
    <sensor name="imu" type="imu">
      <always_on>true</always_on>
      <update_rate>200</update_rate>
      <topic>imu</topic>
    </sensor>
  </gazebo>
  <gazebo reference="lidar_link">
    <sensor name="lidar" type="gpu_lidar">
      <always_on>true</always_on>
      <update_rate>10</update_rate>
      <topic>scan</topic>
      <lidar>
        <scan><horizontal><samples>3200</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal></scan>
        <range><min>0.05</min><max>30.0</max><resolution>0.01</resolution></range>
      </lidar>
    </sensor>
  </gazebo>""")
for s in ["left", "right"]:
    GZ.append('  <gazebo reference="%s_wheel_link"><mu1>1.0</mu1><mu2>1.0</mu2><kp>1e6</kp><kd>100</kd></gazebo>' % s)
for t in ["fl", "fr", "rl", "rr"]:
    GZ.append('  <gazebo reference="caster_%s_wheel_link"><mu1>0.6</mu1><mu2>0.6</mu2></gazebo>' % t)
if not RIGID:
    for s in ["left", "right"]:
        GZ.append('  <gazebo reference="%s_suspension_joint"><springStiffness>45</springStiffness>'
                  '<springReference>0.75</springReference></gazebo>' % s)

# ------------------------------------------------------------------ write
order = ["base_footprint", "base_link", "left_suspension_link", "left_wheel_link", "right_suspension_link", "right_wheel_link"]
order += ["caster_%s_%s_link" % (t, k) for t in ["fl", "fr", "rl", "rr"] for k in ["swivel", "wheel"]]
order += ["lift_stage2_link", "torso_link", "head_camera_link", "head_pantilt_bracket_link", "head_pan_link", "head_tilt_link",
          "pantilt_camera_link", "nav_camera_link", "imu_link", "lidar_link"]
rest = [k for k in links if k not in order and not any(k == LINKNAME.get(x) for x in [])]
names_by_link = {lk.name: lk for lk in links.values()}
seq = order + [n for n in names_by_link if n not in order]

hdr = """<?xml version="1.0" encoding="utf-8"?>
<!--
  Robot V1 (Tnkr) - mobile dual-arm manipulator with a 650 mm lifting torso.
  Generated from the Onshape model "Robot V1 Body" (BOM release 2/3) and the TRLC-DK1 follower URDF.
  Frames: REP-103 (x forward, y left, z up). base_footprint on the floor, base_link on the drive axle (0.100 m up).
  Units: SI. Masses and inertias of the body come from Onshape mass properties of the CAD parts.
  %s
  Actuated joints:
    left_wheel_joint, right_wheel_joint      velocity, 16 N.m peak, 20.94 rad/s (2.09 m/s at the tyre)
    lift_joint                               position, 0 to 0.650 m, 600 N, 0.039 m/s
    head_pan_joint, head_tilt_joint          position, 4.1 N.m, 4.8 rad/s
    left_/right_arm_joint1..3                position, 27 N.m peak, 5.49 rad/s (DM-J4340)
    left_/right_arm_joint4..6                position, 7 N.m peak, 20.94 rad/s (DM-J4310)
    left_/right_arm_gripper_joint            position, -0.045 (open) to 0.001 m, 114 N
  Passive: 4 casters (swivel + wheel)%s; mimic: lift_stage2_joint (0.5 x lift), *_arm_gripper_mimic_joint.
  Depth cameras: head_camera (ZED X, fixed), pantilt_camera (ZED X on the pan-tilt), nav_camera (ZED X Mini);
  depth is referenced to <camera>_left_camera_optical_frame. Gazebo sensor definitions are at the end of the file.
-->
<robot name="robot_v1">
""" % ("Suspension is RIGID in this file (drive wheels held at ride height)." if RIGID else
       "Suspension is SPRUNG in this file: revolute trailing arms, gas spring modelled as a torsional spring.",
       "" if RIGID else ", 2 suspension arms")

def write(path):
    out = [hdr]
    for key, name in sorted(MATS.items(), key=lambda kv: kv[1]):
        out.append('  <material name="%s"><color rgba="%s %s %s 1"/></material>' % (name, f(key[0] / 100), f(key[1] / 100), f(key[2] / 100)))
    for n in seq:
        out.append(names_by_link[n].xml())
    out.extend(JOINTS)
    out.extend(GZ)
    out.append("</robot>\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(out))

name = "robot_v1.urdf" if RIGID else "robot_v1_suspension.urdf"
# materials are collected while writing links, so render links first
_ = [names_by_link[n].xml() for n in seq]
write(os.path.join(PKG, "urdf", name))
tot = sum(lk.inertial[0] for lk in links.values() if lk.inertial)
print("wrote", name, "links", len(seq), "joints", len(JOINTS), "mass %.2f kg" % tot)
