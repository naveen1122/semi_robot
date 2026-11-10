# Robot V1 — simulation model (URDF + MuJoCo)

This is a mobile robot with two arms and a 650 mm lifting torso. The model is built from the Onshape model "Robot V1 Body". The body masses and inertias come from Onshape's mass properties of the CAD parts. The two arms are the TRLC-DK1 follower URDF and meshes, which are Apache-2.0 licensed (see `meshes/dk1/LICENSE-TRLC-DK1.txt`). Joint limits use the motor and actuator data sheets.

## Files

| File | Use |
|---|---|
| `urdf/robot_v1.urdf` | Main model. The suspension is rigid: the drive wheels are held at ride height. Use this one for policy training. |
| `urdf/robot_v1_suspension.urdf` | Same model with the two trailing arms free to rotate. You must give these joints a spring (values below). |
| `mjcf/robot_v1.xml` | MuJoCo version of the rigid URDF. It adds actuators, the 3 depth cameras, mimic constraints, an IMU and a range sensor. |
| `mjcf/scene_empty.xml`, `mjcf/scene.xml` | MuJoCo scenes: floor only, or floor plus a table with a small box on it. |
| `config/joints.yaml` | Every actuated joint with its range, effort, speed and source. |
| `config/sensors.yaml` | Camera intrinsics, frames and depth ranges; IMU and lidar. |
| `scripts/check_mujoco.py` | Physics check: settle, drive, turn on the spot, lift, arms, gripper, head. Also writes RGB and depth images. |
| `scripts/build_urdf.py`, `build_mjcf.py`, `cad_data.py` | Generator scripts, for when the CAD changes. |

Mesh paths in the URDF are relative (`../meshes/dk1/...`). To use the model in a ROS package, replace them with `package://<your_pkg>/meshes/dk1/...`.

## Frames

Frames follow REP-103: x forward, y left, z up, SI units.

- `base_footprint` is on the floor.
- `base_link` is on the drive-wheel axle line, 0.100 m above the floor, centred between the wheels.
- Total mass is 88.5 kg. That breaks down as:
  - 77.5 kg body from CAD;
  - 2 × 4.1 kg arms;
  - 1.8 kg allowance for the release-3 electronics that are not placed yet, lumped in `base_link`;
  - about 1 kg of cameras, pan-tilt and sensors.

## Joints

| Joint | Type | Range | Effort | Speed | From |
|---|---|---|---|---|---|
| `left_wheel_joint`, `right_wheel_joint` | continuous, velocity | – | 16 N·m peak (5.5 rated) | 20.94 rad/s = 2.09 m/s | ZLTECH ZLLG80ASM250-L, 200 mm tyre, 0.5002 m track |
| `caster_{fl,fr,rl,rr}_swivel_joint`, `caster_*_wheel_joint` | continuous, passive | – | – | – | Blickle LPA-PATH 75KF, 75 mm wheel, 28 mm trail |
| `lift_joint` | prismatic, position | 0 – 0.650 m | 600 N | 0.039 m/s | TiMOTION TL4; `lift_stage2_joint` mimics at 0.5× |
| `head_pan_joint` / `head_tilt_joint` | revolute, position | ±90° / −30° (up) to +50° (down) | 4.1 N·m | 4.8 rad/s | DYNAMIXEL XM430-W350-T |
| `{left,right}_arm_joint1..3` | revolute, position | from vendor URDF | 27 N·m | 5.49 rad/s | DAMIAO DM-J4340-2EC |
| `{left,right}_arm_joint4..6` | revolute, position | from vendor URDF | 7 N·m | 20.94 rad/s | DAMIAO DM-J4310-2EC |
| `{left,right}_arm_gripper_joint` | prismatic, position | −0.045 (open) – 0.001 m | 114 N | 0.183 m/s | DK1 gripper; the second finger mimics it |

In total there are 19 actuators:

- 2 wheels
- 1 lift
- 2 head joints
- 12 arm joints
- 2 grippers

The 8 caster joints are passive.

## Depth cameras

| Camera | Device | Where | Depth reference frame |
|---|---|---|---|
| `head_camera` | ZED X, 120 mm baseline | Top of the mast, fixed, 10° down; moves with the lift | `head_camera_left_camera_optical_frame` |
| `pantilt_camera` | ZED X | Below it, on the pan-tilt | `pantilt_camera_left_camera_optical_frame` |
| `nav_camera` | ZED X Mini, 50 mm baseline | Base front, 15° down | `nav_camera_left_camera_optical_frame` |

All three use 110° × 80° field of view. Depth ranges are 0.3–20 m for the ZED X and 0.1–8 m for the ZED X Mini. `config/sensors.yaml` gives the intrinsics at 960 × 600.

The URDF carries Gazebo (gz-sim) `rgbd_camera`, `imu` and `gpu_lidar` sensor blocks. The MuJoCo model has `<camera>` elements that render RGB and depth.

## Using it in each simulator

### Isaac Lab / Isaac Sim

1. Import `urdf/robot_v1.urdf` with `fix_base=False` and keep `merge_fixed_joints=False` so the camera frames remain.
2. Set the drives:
   - wheels: velocity drives (stiffness 0, damping about 50);
   - casters: zero-stiffness drives;
   - lift, head, arms and grippers: position drives.
3. Attach a `Camera`/`TiledCamera` (with `distance_to_image_plane`) to each `*_left_camera_frame` prim. Use the intrinsics from `sensors.yaml`.

### MuJoCo / MJX

Load `mjcf/scene_empty.xml` or `mjcf/scene.xml`.

- The mimic joints are already equality constraints.
- Actuator gains are starting points. Arm `kp` is 150 for joints 1–3 and 40 for joints 4–6, which leaves 0.02–0.035 rad gravity sag.
- Raw MuJoCo ignores URDF `<mimic>`, so use the MJCF and not the URDF there.

### Gazebo / ROS 2

1. Use `robot_v1.urdf` with `robot_state_publisher`.
2. Add `gz_ros2_control` or the gz `DiffDrive` plugin for the wheels.

### PyBullet

The URDF loads. Mimic joints have to be added as gear constraints.

## Suspension variant

Each trailing arm pivots 0.123 m behind the axle. The Suspa gas spring (200 N) presses about 28.5 kg onto each drive wheel and changes little over its stroke.

Model it as a torsional spring: τ = 45 · (0.75 − θ) N·m on each `*_suspension_joint`, with damping 2 N·m·s/rad. This gives 33.8 N·m at ride height. Positive θ lowers the wheel. The joint range is ±0.11 rad, about ±14 mm at the wheel.

The Gazebo spring tags for this are already in the file. In Isaac, set the joint drive stiffness to 45 with a target of 0.75 rad. Without a spring, the arms swing up to their stops and the drive wheels lose grip.

## Checked in MuJoCo 4.0

| Check | Result |
|---|---|
| Settle | `base_link` at 0.0999 m, tilt 0.02°, stable |
| Drive at 5 rad/s for 3 s | 1.46 m (1.50 m ideal; the difference is acceleration) |
| Turn on the spot at ±3 rad/s for 3 s | 149° (206° ideal) |
| Lift to 0.5 m | 0.498 m; middle stage at 0.249 m |
| Gripper | Second finger follows within 0.1 mm |
| Cameras | All three render RGB and depth (examples in `media/`) |

The turn falls short because the casters have to swivel and scrub; a real robot loses some turn rate the same way.

## Known limits — read before training

1. **Body visuals are simple shapes.** Body visuals and collisions are boxes and cylinders, sized from the CAD bounding box of each part. Parts hidden inside the casing are left out of the visuals but kept in the mass. Only the arms use real meshes.
2. **Provisional placements.** The IMU, lidar, pan-tilt camera and base navigation camera were added in BOM release 3 and are not drawn in CAD yet. Their positions are provisional and may move when the CAD is updated.
3. **The pan-tilt camera blocks part of the top camera's view.** In its current position it hides the lower front part of the top head camera's image. Fix this in the CAD step by moving the pan-tilt or the fixed camera.
4. **The lidar sees about 180°.** It sits on the battery-bay lid, so the hood blocks its rear half. Delete `lidar_link` and its joint if you do not fit one.
5. **The cable chain is not modelled** because it bends. Its 1.6 kg is included in `base_link`.
6. **There are 4 casters, not 3.** The CAD has 4 free swivel casters. If casters jitter in a GPU simulator, you can replace each caster wheel collision with a sphere of radius 0.0375 m.
7. **Arm motor data.** Arm motor torques are the DAMIAO catalogue values (DM-J4340 27 N·m peak, DM-J4310 7 N·m peak). One reseller lists a newer DM-J4340 at 40 N·m peak, so check against your motors. The vendor URDF limited all arm joints to 5.4 rad/s; this model uses the motor speeds.
8. **Arm zero pose.** The DK1 zero pose is the folded rest pose: upper arm pointing back, gripper forward. That matches the arm orientation in the Onshape assembly.
