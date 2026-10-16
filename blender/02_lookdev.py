r"""Task 02 car look-dev: paint stills in studio and autumn light, three detail stills and an S-curve
motion test, all into renders/review/02/.

Run headless from the repo root (Blender isn't on PATH, so call it by full path):
    & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender\02_lookdev.py

Safe to re-run: it deletes only its own 02-*.webp, 02-*.mp4 and timing.txt, and rebuilds every scene
from an empty factory setup. The car .glb and the HDRI are only read, never changed.
"""

import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

sys.dont_write_bytecode = True  # no __pycache__ in blender/lib
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import car as car_lib

ROOT = Path(__file__).resolve().parent.parent
HDRI = ROOT / "assets-source" / "hdri" / "autumn_hill_view_4k.hdr"
OUT = ROOT / "renders" / "review" / "02"
TIMING_LOG = OUT / "timing.txt"

STILL_RESOLUTION = (2560, 1440)
STILL_SAMPLES = 256
WEBP_QUALITY = 90
DETAIL_PAINT = "copper-brown"  # chat's pick for the three detail stills; Nems picks the real paint

HDRI_ROTATION = math.radians(36)  # turns the low sun (14 degrees up) to light the camera side of the car
DUSK_STRENGTH = 0.07  # autumn HDRI dimmed ~15x
DUSK_TINT = (0.55, 0.7, 1.0)  # and cooled

# Cameras for the stills: (location, aim point, lens mm). The car's body centre is at the origin,
# facing -Y; its left side is +X.
FRONT_34 = ((4.3, -5.6, 1.15), (0.0, -0.25, 0.6), 50)
REAR_34_LOW = ((3.3, 4.6, 0.42), (0.0, 0.5, 0.62), 40)  # sun side
WHEEL_CLOSE = ((3.1, -3.5, 0.55), (0.75, -1.3, 0.4), 85)
DUSK_FRONT = ((3.4, -6.8, 0.8), (0.2, -2.0, 0.5), 35)

MOTION_RESOLUTION = (960, 540)
MOTION_SAMPLES = 32
MOTION_FPS = 24
MOTION_FRAMES = 60  # 2.5 s
MOTION_SPEED = 40 / 3.6  # m/s (40 km/h, a brisk mountain-road pace)
MOTION_START = 2.0  # metres along the curve where the rear axle starts
S_LEAD = 3.0  # straight road before and after the S, metres
S_LENGTH = 30.0  # the S itself: a left bend then a right bend
S_TIGHTEST_RADIUS = 15.0  # metres, at the middle of each bend
ROAD_WIDTH = 7.0
MOTION_SHUTTER = 0.5  # half a frame open, like a film camera's 180-degree shutter
CAM_FOLLOW = Vector((3.0, 7.5, 0.8))  # fixed offset from the rear axle: ahead and to the right, low
CAM_FOLLOW_LENS = 35


# ---------------------------------------------------------------- shared

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"  # motion blur reads keys between frames
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    return scene


def setup_render(scene, resolution, samples):
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
    cycles.samples = samples
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    cycles.denoising_use_gpu = True
    cycles.caustics_reflective = cycles.caustics_refractive = False
    cycles.transparent_max_bounces = 16  # lamp lenses and windows stack several see-through layers

    render = scene.render
    render.resolution_x, render.resolution_y = resolution
    render.resolution_percentage = 100
    render.use_persistent_data = True
    return f"{cycles.device} via {prefs.compute_device_type}: " + ", ".join(d.name for d in gpus)


def material(name, color, roughness, metallic=0.0):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def noisy_material(name, color, roughness, scale, contrast):
    """Matte surface whose colour varies a little, so the eye reads distance and motion."""
    mat = material(name, color, roughness)
    nt = mat.node_tree
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    ramp = nt.nodes.new("ShaderNodeMix")
    ramp.data_type = "RGBA"
    ramp.inputs["A"].default_value = (*(c * (1 - contrast) for c in color), 1.0)
    ramp.inputs["B"].default_value = (*(c * (1 + contrast) for c in color), 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Factor"])
    nt.links.new(ramp.outputs["Result"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def add_mesh(collection, name, verts, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def hdri_world(name, strength=1.0, tint=(1.0, 1.0, 1.0)):
    world = bpy.data.worlds.new(name)
    nt = world.node_tree
    coords = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, HDRI_ROTATION)
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(HDRI), check_existing=True)
    bg = nt.nodes["Background"]
    nt.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    if tint != (1.0, 1.0, 1.0):
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs["B"].default_value = (*tint, 1.0)
        nt.links.new(env.outputs["Color"], mix.inputs["A"])
        nt.links.new(mix.outputs["Result"], bg.inputs["Color"])
    return world


def add_camera(scene):
    data = bpy.data.cameras.new("Camera")
    data.clip_end = 5000.0
    cam = bpy.data.objects.new("Camera", data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def aim_camera(cam, location, target, lens=None):
    cam.location = location
    cam.rotation_euler = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y").to_euler()
    if lens is not None:
        cam.data.lens = lens


class Log:
    def __init__(self, fh):
        self.fh = fh

    def __call__(self, line):
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


# ---------------------------------------------------------------- stills

def build_studio(scene):
    """Grey curved backdrop facing the front 3/4 camera, one big overhead softbox, two strip lights."""
    coll = bpy.data.collections.new("Studio")
    scene.collection.children.link(coll)

    # Cyclorama profile (y, z): floor, a 4 m curve, then the back wall; extruded 40 m wide.
    profile = [(y, 0.0) for y in range(-20, 8)]
    profile += [(7 + 4 * math.sin(a), 4 - 4 * math.cos(a)) for a in (math.radians(d) for d in range(10, 91, 10))]
    profile += [(11.0, 15.0)]
    verts = [(x, y, z) for y, z in profile for x in (-20.0, 20.0)]
    faces = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(len(profile) - 1)]
    cyc = add_mesh(coll, "Cyclorama", verts, faces, material("StudioGrey", (0.2, 0.2, 0.2), 0.45))
    cyc.rotation_euler.z = math.atan2(FRONT_34[0][0], -FRONT_34[0][1])  # wall sits behind the car from the camera
    for poly in cyc.data.polygons:
        poly.use_smooth = True

    def area(name, size, energy, location, target):
        data = bpy.data.lights.new(name, "AREA")
        data.shape = "RECTANGLE"
        data.size, data.size_y = size
        data.energy = energy
        lamp = bpy.data.objects.new(name, data)
        coll.objects.link(lamp)
        aim_camera(lamp, location, target)  # lights shine down their -Z, like a camera looks

    area("Softbox", (6.0, 2.5), 750.0, (0.0, -0.8, 4.5), (0.0, -0.8, 0.0))
    area("StripFront", (0.5, 4.5), 300.0, (5.0, -2.5, 1.8), (0.0, -1.0, 0.6))
    area("StripBack", (0.5, 4.5), 250.0, (-4.5, 3.5, 2.0), (0.0, 0.0, 0.7))

    world = bpy.data.worlds.new("StudioWorld")
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    return coll, world


def build_autumn(scene):
    """Plain dark asphalt to the horizon; the HDRI gives sky, light and reflections."""
    coll = bpy.data.collections.new("Autumn")
    scene.collection.children.link(coll)
    half = 1000.0
    add_mesh(coll, "Asphalt", [(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)],
             [(0, 1, 2, 3)], noisy_material("Asphalt", (0.045, 0.043, 0.042), 0.8, 40.0, 0.25))
    return coll


def render_still(scene, name, log):
    scene.render.filepath = str(OUT / f"{name}.webp")
    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    log(f"{name}.webp  {time.perf_counter() - start:6.2f} s")


def render_stills(log):
    scene = reset()
    studio, studio_world = build_studio(scene)
    autumn = build_autumn(scene)
    autumn_world = hdri_world("AutumnWorld")
    dusk_world = hdri_world("DuskWorld", DUSK_STRENGTH, DUSK_TINT)
    car = car_lib.build(scene)
    car.root.location = (0.0, -car.centre_y, 0.0)  # body centre on the origin
    cam = add_camera(scene)

    device = setup_render(scene, STILL_RESOLUTION, STILL_SAMPLES)
    image = scene.render.image_settings
    image.media_type = "IMAGE"
    image.file_format = "WEBP"
    image.color_mode = "RGB"
    image.quality = WEBP_QUALITY
    log(f"Blender {bpy.app.version_string} | Cycles on {device}")
    log(f"Stills {STILL_RESOLUTION[0]}x{STILL_RESOLUTION[1]}, {STILL_SAMPLES} samples (adaptive), "
        f"OpenImageDenoise on GPU, WebP quality {WEBP_QUALITY}")

    def lighting(which):
        studio.hide_render = which != "studio"
        autumn.hide_render = which == "studio"
        scene.world = {"studio": studio_world, "autumn": autumn_world, "dusk": dusk_world}[which]

    aim_camera(cam, *FRONT_34)
    for which in ("studio", "autumn"):
        lighting(which)
        for key in car_lib.PAINTS:
            car_lib.set_paint(car, key)
            render_still(scene, f"02-paint-{key}-{which}", log)

    car_lib.set_paint(car, DETAIL_PAINT)
    lighting("autumn")
    aim_camera(cam, *REAR_34_LOW)
    render_still(scene, f"02-detail-rear-34-low-{DETAIL_PAINT}", log)
    aim_camera(cam, *WHEEL_CLOSE)
    render_still(scene, f"02-detail-wheel-chrome-{DETAIL_PAINT}", log)
    lighting("dusk")
    car_lib.set_lights(car, head=1.0, tail=1.0)
    aim_camera(cam, *DUSK_FRONT)
    render_still(scene, f"02-detail-headlights-dusk-{DETAIL_PAINT}", log)


# ---------------------------------------------------------------- motion test

def s_curve(scene):
    """A road heading +Y that bends left, then right (an S), as a Bezier curve. The bend grows and
    eases off smoothly (curvature follows a sine wave), like a real road, so steering never snaps."""
    step, pts = 0.25, []
    x, y, heading = 0.0, 0.0, math.pi / 2
    s = 0.0
    while s <= 2 * S_LEAD + S_LENGTH + 1e-6:
        if s % 2.0 < 1e-6 or 2.0 - s % 2.0 < 1e-6:
            pts.append(Vector((x, y, 0.0)))
        t = (s - S_LEAD) / S_LENGTH
        kappa = math.sin(2 * math.pi * t) / S_TIGHTEST_RADIUS if 0.0 <= t <= 1.0 else 0.0
        heading += kappa * step
        x, y, s = x + math.cos(heading) * step, y + math.sin(heading) * step, s + step

    data = bpy.data.curves.new("SCurve", "CURVE")
    data.dimensions = "3D"
    spline = data.splines.new("BEZIER")
    spline.bezier_points.add(len(pts) - 1)
    for i, (bp, p) in enumerate(zip(spline.bezier_points, pts, strict=True)):
        tangent = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]) / (6 if 0 < i < len(pts) - 1 else 3)
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "FREE"
        bp.handle_left, bp.handle_right = p - tangent, p + tangent
    obj = bpy.data.objects.new("SCurve", data)
    scene.collection.objects.link(obj)
    return obj


def build_road(scene, path):
    """Road strip with a dashed centre line along the path, on brown ground (like 01)."""
    coll = scene.collection
    steps = int(path.length / 0.5)
    verts, faces = [], []
    for i in range(steps + 1):
        s = i * path.length / steps
        p, h = path.point(s), path.heading(s)
        side = Vector((math.sin(h), -math.cos(h), 0.0))  # to the right of travel
        verts += [p - side * ROAD_WIDTH / 2, p + side * ROAD_WIDTH / 2]
        if i:
            faces.append((2 * i - 2, 2 * i - 1, 2 * i + 1, 2 * i))
    add_mesh(coll, "Road", verts, faces, noisy_material("Asphalt", (0.045, 0.043, 0.042), 0.8, 40.0, 0.25))

    verts, faces = [], []
    s = 0.0
    while s + 3.0 <= path.length:
        for a in (s, s + 3.0):
            p, h = path.point(a), path.heading(a)
            side = Vector((math.sin(h), -math.cos(h), 0.0)) * 0.075
            verts += [p - side + Vector((0, 0, 0.003)), p + side + Vector((0, 0, 0.003))]
        n = len(verts)
        faces.append((n - 4, n - 3, n - 1, n - 2))
        s += 9.0
    add_mesh(coll, "CentreLine", verts, faces, material("LinePaint", (0.75, 0.75, 0.7), 0.6))

    half = 500.0
    add_mesh(coll, "Ground", [(-half, -half, -0.02), (half, -half, -0.02), (half, half, -0.02), (-half, half, -0.02)],
             [(0, 1, 2, 3)], noisy_material("Ground", (0.11, 0.09, 0.05), 1.0, 2.0, 0.4))


def render_motion(log):
    scene = reset()
    curve = s_curve(scene)
    path = car_lib.CurvePath(curve)
    build_road(scene, path)
    scene.world = hdri_world("AutumnWorld")
    car = car_lib.build(scene, paint=DETAIL_PAINT)
    cam = add_camera(scene)
    cam.data.lens = CAM_FOLLOW_LENS

    def distance(frame):
        return MOTION_START + MOTION_SPEED * (frame - 1) / MOTION_FPS

    car_lib.drive(car, curve, 1, MOTION_FRAMES, distance)
    for frame in range(1, MOTION_FRAMES + 1):  # follow at a fixed offset; the car turns under the camera
        s = distance(frame)
        axle = path.point(s)
        ahead = Vector((math.cos(path.heading(s)), math.sin(path.heading(s)), 0.0))
        aim_camera(cam, axle + CAM_FOLLOW, axle + ahead * 1.2 + Vector((0, 0, 0.45)))
        cam.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("rotation_euler", frame=frame)

    setup_render(scene, MOTION_RESOLUTION, MOTION_SAMPLES)
    scene.frame_start, scene.frame_end = 1, MOTION_FRAMES
    scene.render.fps = MOTION_FPS
    image = scene.render.image_settings
    image.media_type = "VIDEO"
    image.file_format = "FFMPEG"
    image.color_mode = "RGB"
    ffmpeg = scene.render.ffmpeg
    ffmpeg.format, ffmpeg.codec, ffmpeg.constant_rate_factor = "MPEG4", "H264", "HIGH"

    log(f"Motion test {MOTION_RESOLUTION[0]}x{MOTION_RESOLUTION[1]}, {MOTION_SAMPLES} samples, "
        f"{MOTION_FRAMES} frames at {MOTION_FPS} fps, {MOTION_SPEED * 3.6:.0f} km/h, "
        f"{MOTION_SPEED * MOTION_FRAMES / MOTION_FPS:.1f} m")
    for name, blur in (("02-motion-s-curve", True), ("02-motion-s-curve-noblur", False)):
        scene.render.use_motion_blur = blur
        scene.render.motion_blur_shutter = MOTION_SHUTTER
        scene.render.filepath = str(OUT / f"{name}.mp4")
        start = time.perf_counter()
        bpy.ops.render.render(animation=True)
        took = time.perf_counter() - start
        log(f"{name}.mp4  {took:6.1f} s ({took / MOTION_FRAMES:.2f} s/frame)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in [*OUT.glob("02-*.webp"), *OUT.glob("02-*.mp4"), TIMING_LOG]:
        old.unlink(missing_ok=True)
    total = time.perf_counter()
    with TIMING_LOG.open("w", encoding="utf-8") as fh:
        log = Log(fh)
        render_stills(log)
        render_motion(log)
        log(f"total {(time.perf_counter() - total) / 60:.1f} min")


if __name__ == "__main__":
    main()
