r"""Task 01 pipeline proof: the 911 drives a gently curving road; 150 WebP frames for the web page.

Run headless from the repo root (Blender isn't on PATH, so call it by full path):
    & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender\00_proof.py

Safe to re-run: it starts from an empty factory scene, deletes the old proof frames and rewrites
the timing log. The car .glb and the HDRI are only read, never changed.
"""

import math
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent.parent
CAR_GLB = ROOT / "assets-source" / "free_1975_porsche_911_930_turbo.glb"
HDRI = ROOT / "assets-source" / "hdri" / "autumn_hill_view_4k.hdr"
FRAMES_DIR = ROOT / "public" / "frames" / "proof"
TIMING_LOG = ROOT / "renders" / "proof-timing.txt"

FRAME_COUNT = 150
RESOLUTION = (2560, 1440)
SAMPLES = 64
WEBP_QUALITY = 80

CAR_LENGTH = 4.29  # metres bumper to bumper, like a real 930 Turbo
SHADOW_PLANE_MATERIAL = "material_0"  # flat baked-shadow decal that ships with the model

ROAD_RADIUS = 150.0  # the road is one gentle arc, curving right
ROAD_WIDTH = 7.0
ROAD_RANGE = (-30.0, 280.0)  # metres along the arc
DASH_LENGTH, DASH_GAP, DASH_WIDTH = 3.0, 6.0, 0.15  # centre line, so motion is easy to see
LANE_OFFSET = 1.75  # the car drives in the right-hand lane
DRIVE_DISTANCE = 60.0  # metres covered over the whole shot

CAM_BEHIND = 6.5  # metres behind the car, along the road
CAM_SIDE = -3.0  # metres sideways from the car (negative = left)
CAM_HEIGHT = 1.0
CAM_AIM_HEIGHT = 0.6  # aim this far above the car's origin (which sits on the road)
CAM_LENS = 40  # mm


def road_point(s, offset=0.0, z=0.0):
    """Point s metres along the road's centre line, `offset` metres to the right of it."""
    a = s / ROAD_RADIUS
    right = Vector((math.cos(a), -math.sin(a), 0.0))
    return Vector((ROAD_RADIUS * (1.0 - math.cos(a)), ROAD_RADIUS * math.sin(a), z)) + offset * right


def make_material(name, color, roughness):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def add_mesh(scene, name, verts, faces, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj


def build_world(scene):
    world = bpy.data.worlds.new("World")
    nodes = world.node_tree.nodes
    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(HDRI))
    world.node_tree.links.new(env.outputs["Color"], nodes["Background"].inputs["Color"])
    scene.world = world


def build_ground_and_road(scene):
    start, end = ROAD_RANGE

    # Road: a 7 m strip along the arc, one segment per metre.
    steps = int(end - start)
    verts = []
    for i in range(steps + 1):
        verts += [road_point(start + i, -ROAD_WIDTH / 2), road_point(start + i, ROAD_WIDTH / 2)]
    faces = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(steps)]
    add_mesh(scene, "Road", verts, faces, make_material("Asphalt", (0.035, 0.035, 0.037), 0.85))

    # Dashed centre line, 3 mm above the asphalt.
    verts, faces = [], []
    s = start
    while s + DASH_LENGTH <= end:
        half = DASH_WIDTH / 2
        verts += [
            road_point(s, -half, 0.003),
            road_point(s, half, 0.003),
            road_point(s + DASH_LENGTH, half, 0.003),
            road_point(s + DASH_LENGTH, -half, 0.003),
        ]
        faces.append(tuple(range(len(verts) - 4, len(verts))))
        s += DASH_LENGTH + DASH_GAP
    add_mesh(scene, "CentreLine", verts, faces, make_material("LinePaint", (0.75, 0.75, 0.7), 0.6))

    # Ground: a 2 km square, 2 cm below the road.
    c, half = road_point((start + end) / 2), 1000.0
    verts = [(c.x - half, c.y - half, -0.02), (c.x + half, c.y - half, -0.02),
             (c.x + half, c.y + half, -0.02), (c.x - half, c.y + half, -0.02)]
    add_mesh(scene, "Ground", verts, [(0, 1, 2, 3)], make_material("Ground", (0.11, 0.09, 0.05), 1.0))


def vertex_bounds(objects):
    """World-space bounding box of the actual vertices (per-object boxes overstate rotated parts)."""
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for obj in objects:
        if obj.type != "MESH":
            continue
        mw = obj.matrix_world
        for v in obj.data.vertices:
            p = mw @ v.co
            for i in range(3):
                lo[i] = min(lo[i], p[i])
                hi[i] = max(hi[i], p[i])
    return lo, hi


def import_car(scene):
    """Import the .glb, drop its baked shadow plane, scale it to real size and stand it on z 0."""
    existing = set(scene.objects)  # road and ground are already built; measure only the car
    bpy.ops.import_scene.gltf(filepath=str(CAR_GLB))
    shadow = [o for o in scene.objects if o not in existing and o.type == "MESH"
              and any(s.material and s.material.name == SHADOW_PLANE_MATERIAL for s in o.material_slots)]
    for obj in shadow:
        bpy.data.objects.remove(obj, do_unlink=True)

    imported = [o for o in scene.objects if o not in existing]
    roots = [o for o in imported if o.parent is None]
    bpy.context.view_layer.update()
    lo, hi = vertex_bounds(imported)
    length = hi.y - lo.y  # the model's long axis is Y, with its front at -Y
    scale = CAR_LENGTH / length
    centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    fix = Matrix.Translation(-scale * centre) @ Matrix.Scale(scale, 4)

    car = bpy.data.objects.new("Car", None)  # empty that carries the whole model
    scene.collection.objects.link(car)
    for root in roots:
        root.matrix_world = fix @ root.matrix_world
        root.parent = car
    print(f"Car: removed {len(shadow)} shadow plane(s); length {length:.3f} m scaled x{scale:.4f} "
          f"to {CAR_LENGTH} m; size was {tuple(round(v, 3) for v in hi - lo)}; lowest point now z 0")
    return car


def add_camera(scene):
    data = bpy.data.cameras.new("Camera")
    data.lens = CAM_LENS
    data.clip_end = 5000.0  # far enough to see the edge of the 2 km ground
    cam = bpy.data.objects.new("Camera", data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def animate(car, cam):
    """One keyframe per frame: the car drives the arc, the camera chases at a low rear 3/4."""
    for frame in range(1, FRAME_COUNT + 1):
        s = DRIVE_DISTANCE * (frame - 1) / (FRAME_COUNT - 1)
        car_pos = road_point(s, LANE_OFFSET)
        car.location = car_pos
        car.rotation_euler = (0.0, 0.0, math.pi - s / ROAD_RADIUS)  # model front is -Y
        cam_pos = road_point(s - CAM_BEHIND, LANE_OFFSET + CAM_SIDE, CAM_HEIGHT)
        cam.location = cam_pos
        aim = car_pos + Vector((0.0, 0.0, CAM_AIM_HEIGHT))
        cam.rotation_euler = (aim - cam_pos).to_track_quat("-Z", "Y").to_euler()
        for obj in (car, cam):
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("rotation_euler", frame=frame)


def setup_render(scene):
    """Cycles on the OptiX GPU(s) only; stop rather than fall back to the CPU."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.refresh_devices()
    gpus = [d for d in prefs.devices if d.type == "OPTIX"]
    if not gpus:
        raise SystemExit("No OptiX GPU found; refusing to render on the CPU.")
    for d in prefs.devices:
        d.use = d.type == "OPTIX"

    scene.render.engine = "CYCLES"
    cycles = scene.cycles
    cycles.device = "GPU"
    cycles.samples = SAMPLES
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    cycles.denoising_use_gpu = True

    render = scene.render
    render.resolution_x, render.resolution_y = RESOLUTION
    render.resolution_percentage = 100
    render.use_persistent_data = True  # keep the scene on the GPU between frames
    image = render.image_settings
    image.media_type = "IMAGE"
    image.file_format = "WEBP"
    image.color_mode = "RGB"
    image.quality = WEBP_QUALITY
    scene.frame_start, scene.frame_end = 1, FRAME_COUNT
    return f"{cycles.device} via {prefs.compute_device_type}: " + ", ".join(d.name for d in gpus)


def render_frames(scene, device):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_DIR.glob("*.webp"):
        old.unlink()
    TIMING_LOG.parent.mkdir(parents=True, exist_ok=True)

    times = []
    with TIMING_LOG.open("w", encoding="utf-8") as log:

        def out(line):
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        cycles = scene.cycles
        out(f"Blender {bpy.app.version_string} | Cycles on {device}")
        out(f"{RESOLUTION[0]}x{RESOLUTION[1]}, {SAMPLES} samples (adaptive), denoiser {cycles.denoiser} "
            f"on GPU: {cycles.denoising_use_gpu}, WebP quality {WEBP_QUALITY}")
        for frame in range(1, FRAME_COUNT + 1):
            scene.frame_set(frame)
            scene.render.filepath = str(FRAMES_DIR / f"{frame:04d}")
            start = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            times.append(time.perf_counter() - start)
            out(f"frame {frame:04d}  {times[-1]:6.2f} s")

        slowest = max(range(len(times)), key=times.__getitem__)
        out(f"total {sum(times) / 60:.1f} min | average {sum(times) / len(times):.2f} s/frame | "
            f"average without frame 1 {sum(times[1:]) / (len(times) - 1):.2f} s | "
            f"slowest {times[slowest]:.2f} s (frame {slowest + 1:04d})")


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0

    build_world(scene)
    build_ground_and_road(scene)
    car = import_car(scene)
    cam = add_camera(scene)
    animate(car, cam)
    device = setup_render(scene)
    render_frames(scene, device)


main()
