r"""Handoff 08: time test, step 1. Five keyframes of the porch view, from golden hour to night.

The scene is the signed-off round 3 still, built with blender/05_porch_still.py's own builders (loaded as a module and
left unchanged). On top of it this adds a warm porch light and a night sky (procedural stars and a faint glow). Each
keyframe then gets its own sun height, Sun lamp, clouds, porch light and camera exposure.

Run headless from the repo root (Blender isn't on PATH, so call it by full path):
    & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender\08_time_keyframes.py
    ... -P blender\08_time_keyframes.py -- --keys k1            # one keyframe: 4 to 7 minutes
    ... -P blender\08_time_keyframes.py -- --preview NAME [--keys k1,k3] [--no-trees] [--samples N]

The full run builds the scene once and renders all five at 2560x1440 (about 15 minutes). It writes into
renders/review/08/:
    k1-golden.webp, k2-sunset.webp, k3-afterglow.webp, k4-blue-hour.webp, k5-night.webp
    08-keyframes.webp     contact sheet of the keyframes on disk, rebuilt after every run
    08-render-log.txt     one section per keyframe, replaced when that keyframe is rendered again
    checks/k5-stars-png-vs-webp.png   k5's stars at 100%, before and after WebP compression
Previews (1280x720, labelled) go to renders/review/08/preview/. Nothing is written outside renders/review/08/.
"""

import argparse
import importlib.util
import math
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "renders" / "review" / "08"
LOG_PATH = OUT / "08-render-log.txt"

# The signed-off scene. Loaded as a module, so its main() doesn't run. Its builders read the module's settings when
# they're called, so each keyframe sets those first.
sys.dont_write_bytecode = True  # no __pycache__ next to the scripts
_spec = importlib.util.spec_from_file_location("porch_still", Path(__file__).with_name("05_porch_still.py"))
porch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(porch)
porch.OUT = OUT  # 05's own output folder (07/) is never written from here


def _offline(url):
    raise SystemExit(f"08 runs offline, but an asset is missing and 05 tried to download it: {url}")


porch._open = _offline  # every asset is on disk already: this run never touches the network


# ---------------------------------------------------------------------------------------------
# The keyframes.

@dataclass(frozen=True)
class Key:
    name: str
    moment: str
    file: str  # output name, without .webp
    sun: float  # sun height in degrees above the horizon; its yaw stays 55 (the signed-off tree layout follows it)
    lamp_from: str  # where the Sun lamp is measured: "porch", "cloud base", or "" for no lamp
    cloud_cover: float | None  # 05's CLOUD_COVER (higher: fewer, smaller clouds); None: no cloud layer
    cloud_density: float
    porch_light: bool
    exposure: float  # camera exposure, EV


# Exposure: like a time-lapse camera, each keyframe brightens by about two thirds of how much darker its frame is than
# k1's (metered as the frame's log-average brightness in previews), then adjusted by eye by up to a quarter stop. Once
# the porch light is on it dominates the meter, so the camera brightens little from blue hour to night.
KEYS = {k.name: k for k in (
    # Sun 2 degrees above the ridge: the light comes in over the rail and the baluster stripes cross the deck.
    Key("k1", "golden hour", "k1-golden", 4.5, "porch", 0.58, 0.0011, False, -2.3),
    # The disc sits on the 2.48-degree ridge: the deck and pines get the last, partly blocked light.
    Key("k2", "sunset", "k2-sunset", 2.6, "porch", 0.58, 0.0011, False, -1.85),
    # Sun just set: only the clouds and the high haze still see it, so they glow from underneath.
    Key("k3", "afterglow", "k3-afterglow", -1.5, "cloud base", 0.55, 0.0012, False, 0.8),
    Key("k4", "blue hour", "k4-blue-hour", -5.0, "", 0.64, 0.0009, True, 2.4),
    # Far enough below the horizon that the sky model's glow is gone: the night glow and stars are the sky.
    Key("k5", "night", "k5-night", -30.0, "", None, 0.0, True, 3.9),
)}

# The Sun lamp carries the sun's light and shadows. Its strength and colour are measured per keyframe, from where it
# matters: the porch while the sun is above the ridge, the cloud base (5.5 km above the porch) once the ridges shade
# everything lower. Below about -2.7 degrees nothing in the scene sees the sun any more.
LAMP_ALTITUDE = {"porch": porch.SKY_ALTITUDE, "cloud base": porch.SKY_ALTITUDE + porch.CLOUD_BASE}
LAMP_MIN = 0.05  # weaker than this, the lamp is left out

# 05's camera sees 60 km (its clip distance), but its cloud layer only fades out between 68 and 84 km, so the camera
# cuts the layer off. In golden hour that edge hides in the glow; lit pink from below at afterglow it shows as a hard
# horizontal line about 6 degrees up. Fading the layer out before 60 km keeps the clouds the signed-off still shows
# and turns the cut into a soft edge (horizontal distance from the camera, metres).
CLOUD_FADE = (40000.0, 59000.0)

# Porch light: a warm lantern on a small table by the rail, just out of frame to the left, never in view. It lights the
# mug and rail from the left and fades across the deck to the right. (A lamp hanging from the ceiling lit the ceiling
# and header far more than the mug; a lamp closer to the mug burned out the left post.)
PORCH_LIGHT_AT = (-0.3, -0.75, 1.15)
PORCH_LIGHT_POWER = 9.0  # W: at blue hour it's the warm light on the mug and rail, at night the main light
PORCH_LIGHT_KELVIN = 2700.0  # incandescent; the camera stays balanced for daylight, so it renders warm
PORCH_LIGHT_RADIUS = 0.025  # a lantern globe about 5 cm across: soft shadow edges

# Night sky. Stars: a Voronoi pattern over the sky directions puts one candidate star in every cell; each gets a
# random brightness on a power law (about 3 times as many stars per step of 2.5 times fainter, like the real sky) and
# a slight colour. Only the camera sees them, so they add no noise to the lighting. A faint glow under them (like the
# airglow of a real dark sky, brighter toward the horizon) lights the scene a little. Both stay constant through the
# arc: in daylight the sky swamps them, and they show as the sky darkens.
STAR_CELLS = 600.0  # cells per radian: one candidate every ~0.1 degree (~3.7 px at 2560 wide)
STAR_SHARE = 0.45  # share of cells that hold a star
STAR_RADIUS = 0.16  # core radius in cell units (~0.6 px); the f/8 lens and pixel filter spread it to ~1.5 px
STAR_FAINTEST = 4e-3  # core radiance of the faintest star
STAR_SLOPE = 0.85  # power-law exponent of the brightness
STAR_RANGE = 3000.0  # brightest / faintest: a few stars stand out
STAR_EXTINCTION = 0.16  # optical depth of the air straight up: stars dim and vanish toward the horizon
STAR_TINTS = ((0.0, (1.0, 0.62, 0.38)), (0.16, (1.0, 0.86, 0.70)), (0.40, (1.0, 0.97, 0.93)),
              (0.85, (0.86, 0.91, 1.0)), (1.0, (0.70, 0.80, 1.0)))  # orange through white to blue-white
# The glow is several stops brighter than a real moonless sky, so the night stays readable next to the porch light
# (blue hour to night is a 4-5 stop drop here instead of 10 or more), and blue so it reads as night, not grey.
AIRGLOW = (9.0e-4, 1.35e-3, 2.25e-3)  # radiance straight up
AIRGLOW_HORIZON = 2.5  # how much brighter it is at the horizon

# The saturation lift on the bright sky (05's grade) was set in absolute brightness at round 3's exposure. It follows
# the exposure, so it acts on the same displayed brightness in every keyframe.
GRADE_TUNED_AT = porch.EXPOSURE
GRADE_MASK_R3 = porch.GRADE_MASK

# Star check: a 100% crop of k5's sky (x, y from the top-left, width, height), before and after WebP compression.
STAR_CROP = (360, 40, 512, 384)
LABEL_H, LABEL_SIZE = 44, 26


# ---------------------------------------------------------------------------------------------
# Sun lamp measurement.

def mean_rgb(scene, path):
    s = scene.render.image_settings
    s.file_format, s.color_depth = "OPEN_EXR", "32"  # linear values, no view transform
    bpy.data.images["Render Result"].save_render(str(path), scene=scene)
    im, px = porch.read_pixels(path)
    bpy.data.images.remove(im)
    return px[..., :3].reshape(-1, 3).mean(0)


def measure_lamp(key, tmp):
    """Strength and colour for the Sun lamp at this keyframe (rounds 1-2's method): a white plane facing the sun, lit by
    the sky with its sun disc, by the sky without it, and by a strength-1 Sun lamp alone. (disc - no disc) / lamp is
    the sun's own light. Each render resets Blender, so this runs before the scene is built."""
    keep = porch.SUN_ELEVATION, porch.SKY_ALTITUDE
    porch.SUN_ELEVATION, porch.SKY_ALTITUDE = key.sun, LAMP_ALTITUDE[key.lamp_from]
    facing = porch.sun_direction().to_track_quat("Z", "Y").to_euler()  # +Z toward the sun
    means = []
    for mode in ("disc", "no disc", "lamp"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        porch.setup_render(scene, False, 1024)
        scene.cycles.use_adaptive_sampling = scene.cycles.use_denoising = False
        scene.render.resolution_x = scene.render.resolution_y = 16
        mesh = bpy.data.meshes.new("plane")
        mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
        mesh.materials.append(porch.plain_material("white", (1.0, 1.0, 1.0), 1.0))
        porch.link_object(bpy.data.objects.new("plane", mesh)).rotation_euler = facing
        cam = bpy.data.cameras.new("camera")
        cam.type, cam.ortho_scale = "ORTHO", 0.5
        scene.camera = porch.link_object(bpy.data.objects.new("camera", cam))
        scene.camera.location = porch.sun_direction() * 5.0
        scene.camera.rotation_euler = facing  # a camera looks down its -Z: back at the plane
        world = bpy.data.worlds.new("world")
        background = world.node_tree.nodes["Background"]
        if mode == "lamp":
            background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            light = bpy.data.lights.new("sun", "SUN")
            light.energy, light.color, light.angle = 1.0, (1.0, 1.0, 1.0), math.radians(porch.SUN_ANGLE)
            porch.link_object(bpy.data.objects.new("sun", light)).rotation_euler = facing
        else:
            sky = porch.sky_node(world.node_tree, mode == "disc")
            world.node_tree.links.new(sky.outputs["Color"], background.inputs["Color"])
        scene.world = world
        bpy.ops.render.render()
        means.append(mean_rgb(scene, tmp / "measure.exr"))
    porch.SUN_ELEVATION, porch.SKY_ALTITUDE = keep
    disc, sky, lamp = means
    sun = np.maximum((disc - sky) / lamp, 0.0)
    strength = float(sun.max())
    if strength < LAMP_MIN:
        return 0.0, (0.0, 0.0, 0.0)
    return strength, tuple(float(c) for c in sun / strength)


# ---------------------------------------------------------------------------------------------
# Scene: 05's build, plus the porch light and the night sky.

def add_porch_light():
    light = bpy.data.lights.new("porch_light", "POINT")
    light.energy = PORCH_LIGHT_POWER
    light.use_temperature = True
    light.temperature = PORCH_LIGHT_KELVIN
    light.shadow_soft_size = PORCH_LIGHT_RADIUS
    obj = porch.link_object(bpy.data.objects.new("porch_light", light))
    obj.location = PORCH_LIGHT_AT
    return obj


def add_night_sky(world):
    """Adds the night glow to both of 05's sky colours (the one light bounces see and the one the camera sees) and the
    stars to the camera's only."""
    nt = world.node_tree
    lit = nt.nodes["Background"]
    seen = next(n for n in nt.nodes if n.type == "BACKGROUND" and n != lit)
    direction = porch.node(nt, "ShaderNodeNewGeometry").outputs["Position"]  # in a world shader: the ray's direction
    split = porch.node(nt, "ShaderNodeSeparateXYZ")
    nt.links.new(direction, split.inputs[0])
    up = split.outputs["Z"]  # sine of the elevation

    toward_horizon = porch.node(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    toward_horizon.inputs["From Min"].default_value, toward_horizon.inputs["From Max"].default_value = 0.35, 0.0
    nt.links.new(up, toward_horizon.inputs["Value"])
    glow = porch.node(nt, "ShaderNodeVectorMath", operation="SCALE")
    glow.inputs[0].default_value = AIRGLOW
    nt.links.new(porch.math_node(nt, "MULTIPLY_ADD", toward_horizon.outputs[0], AIRGLOW_HORIZON - 1.0, 1.0),
                 glow.inputs["Scale"])

    cells = porch.node(nt, "ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1")
    cells.inputs["Scale"].default_value = STAR_CELLS
    cells.inputs["Randomness"].default_value = 1.0
    nt.links.new(direction, cells.inputs["Vector"])
    rnd = porch.node(nt, "ShaderNodeSeparateColor")  # three independent random numbers per cell
    nt.links.new(cells.outputs["Color"], rnd.inputs[0])
    exists = porch.math_node(nt, "LESS_THAN", rnd.outputs[0], STAR_SHARE)
    brightness = porch.math_node(nt, "MINIMUM", porch.math_node(
        nt, "POWER", porch.math_node(nt, "MAXIMUM", rnd.outputs[1], 1e-6), -STAR_SLOPE), STAR_RANGE)
    core = porch.math_node(nt, "EXPONENT", porch.math_node(nt, "MULTIPLY", porch.math_node(
        nt, "POWER", porch.math_node(nt, "DIVIDE", cells.outputs["Distance"], STAR_RADIUS), 2.0), -1.0))
    extinction = porch.math_node(nt, "EXPONENT", porch.math_node(
        nt, "DIVIDE", -STAR_EXTINCTION, porch.math_node(nt, "MAXIMUM", up, 0.01)))
    amount = porch.math_node(nt, "MULTIPLY", porch.math_node(nt, "MULTIPLY", exists, brightness), porch.math_node(
        nt, "MULTIPLY", core, porch.math_node(nt, "MULTIPLY", extinction, STAR_FAINTEST)))
    tint = porch.node(nt, "ShaderNodeValToRGB")
    els = tint.color_ramp.elements
    for i, (pos, color) in enumerate(STAR_TINTS):
        el = els[i] if i < len(els) else els.new(pos)
        el.position, el.color = pos, (*color, 1.0)
    nt.links.new(rnd.outputs[2], tint.inputs["Fac"])
    stars = porch.node(nt, "ShaderNodeVectorMath", operation="SCALE")
    nt.links.new(tint.outputs["Color"], stars.inputs[0])
    nt.links.new(amount, stars.inputs["Scale"])

    for background, extra in ((lit, [glow.outputs[0]]), (seen, [glow.outputs[0], stars.outputs[0]])):
        color = next(lk.from_socket for lk in nt.links if lk.to_socket == background.inputs["Color"])
        for term in extra:
            add = porch.node(nt, "ShaderNodeVectorMath", operation="ADD")
            nt.links.new(color, add.inputs[0])
            nt.links.new(term, add.inputs[1])
            color = add.outputs[0]
        nt.links.new(color, background.inputs["Color"])


def build_scene(trees, preview, samples, log):
    """Everything that stays the same through the arc, built once with 05's builders. The sky, Sun lamp, clouds and
    compositor depend on the keyframe and are built in apply_key()."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    porch.build_camera(scene)
    porch.build_porch(random.Random(porch.SEED))
    porch.build_mug()
    t = time.time()
    terrain, _ = porch.build_terrain()
    porch.build_far_ridges()
    log(f"Terrain built in {time.time() - t:.0f} s")
    if trees:
        porch.build_trees(terrain, log)
    porch.build_atmosphere()
    add_porch_light()
    return scene, porch.setup_render(scene, preview, samples)


def apply_key(scene, key, lamp):
    porch.SUN_ELEVATION = key.sun
    porch.SUN_STRENGTH, porch.SUN_COLOR = lamp
    for name in ("sun", "clouds"):  # the previous keyframe's
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name])
    old_world = scene.world
    porch.build_world(scene)
    add_night_sky(scene.world)
    if old_world:
        bpy.data.worlds.remove(old_world)
    if lamp[0] > 0:
        porch.build_sun()
    if key.cloud_cover is not None:
        porch.CLOUD_COVER, porch.CLOUD_DENSITY, porch.CLOUD_FADE = key.cloud_cover, key.cloud_density, CLOUD_FADE
        porch.build_clouds()
    bpy.data.objects["porch_light"].hide_render = not key.porch_light
    scene.view_settings.exposure = key.exposure
    porch.GRADE_MASK = tuple(g * 2 ** (GRADE_TUNED_AT - key.exposure) for g in GRADE_MASK_R3)
    old_compositor = scene.compositing_node_group
    porch.build_compositor(scene)
    if old_compositor:
        bpy.data.node_groups.remove(old_compositor)


def render(scene):
    """Renders; returns the time and the highest memory Cycles reported."""
    stats = {"peak_mb": 0.0}

    def on_stats(*args):
        for amount, unit in re.findall(r"(?:Mem|Peak):\s*([\d.]+)([KMG])", " ".join(map(str, args))):
            stats["peak_mb"] = max(stats["peak_mb"], float(amount) * {"K": 1 / 1024, "M": 1, "G": 1024}[unit])

    bpy.app.handlers.render_stats.append(on_stats)
    t = time.time()
    bpy.ops.render.render()
    seconds = time.time() - t
    bpy.app.handlers.render_stats.remove(on_stats)
    return seconds, stats["peak_mb"]


# ---------------------------------------------------------------------------------------------
# Outputs: keyframes, labels, the contact sheet, the star check and the log.

def save_render(scene, path, file_format, quality=95):
    s = scene.render.image_settings
    s.file_format, s.color_mode, s.color_depth = file_format, "RGB", "8"
    if file_format == "WEBP":
        s.quality = quality
    bpy.data.images["Render Result"].save_render(str(path), scene=scene)


def load(path):
    im, px = porch.read_pixels(path)
    bpy.data.images.remove(im)
    return px  # rows from the bottom up, as Blender stores them


def write_png(arr, path):
    h, w = arr.shape[:2]
    im = bpy.data.images.new(Path(path).stem, w, h)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = arr[..., :3]
    im.pixels.foreach_set(rgba.ravel())
    im.file_format = "PNG"
    im.save(filepath=str(path))
    bpy.data.images.remove(im)


def label(text, width):
    """A dark strip with light text (rows from the bottom up). Drawn headless: blf into an ImBuf."""
    import blf
    import imbuf
    ib = imbuf.new((width, LABEL_H))
    with blf.bind_imbuf(0, ib):
        blf.size(0, LABEL_SIZE)
        blf.color(0, 0.92, 0.92, 0.92, 1.0)
        blf.position(0, 16, (LABEL_H - LABEL_SIZE) / 2 + 4, 0)
        blf.draw_buffer(0, text)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "label.png"
        ib.file_type = "PNG"
        imbuf.write(ib, filepath=str(path))
        text_px = load(path)
    strip = np.full((LABEL_H, width, 4), 0.06, np.float32)
    alpha = text_px[..., 3:4]
    strip[..., :3] = strip[..., :3] * (1 - alpha) + text_px[..., :3] * alpha
    return strip


def labelled(img, text):
    return np.concatenate([label(text, img.shape[1]), img], axis=0)  # the label goes under the picture


def key_label(key):
    return f"{key.name}  {key.moment}   ·   sun {key.sun:+.1f}°   ·   exposure {key.exposure:+g} EV"


def contact_sheet():
    """The five keyframes on disk at 1280x720, labelled, three on top and two centred below."""
    w, h, gap = 1280, 720, 24
    tiles = []
    for key in KEYS.values():
        path = OUT / f"{key.file}.webp"
        if path.exists():
            tiles.append(labelled(porch.scaled(path, h), key_label(key)))
        else:
            tiles.append(labelled(np.full((h, w, 4), 0.06, np.float32), f"{key.name}  {key.moment}: not rendered yet"))
    th = tiles[0].shape[0]
    sheet = np.full((2 * th + 3 * gap, 3 * w + 4 * gap, 4), 0.06, np.float32)
    for i, tile in enumerate(tiles):
        row, col = divmod(i, 3)
        x = gap + col * (w + gap) + row * (w + gap) // 2
        top = gap + row * (th + gap)
        sheet[sheet.shape[0] - top - th:sheet.shape[0] - top, x:x + w] = tile
    porch.write_pixels(sheet, OUT / "08-keyframes.webp", quality=92)


def star_peaks(lum, threshold):
    """Mask of stars: pixels brighter than their 8 neighbours and at least `threshold` above the ring 3 px around."""
    h, w = lum.shape

    def shifted(dy, dx):
        return lum[3 + dy:h - 3 + dy, 3 + dx:w - 3 + dx]

    near = np.max([shifted(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx], axis=0)
    ring = np.mean([shifted(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if max(abs(dy), abs(dx)) == 3],
                   axis=0)
    core = shifted(0, 0)
    return (core > near) & (core - ring >= threshold), core - ring


def star_check(scene, webp, log):
    """k5's stars at 100%: the frame saved losslessly before compression, beside the same crop from the final WebP."""
    x, y, w, h = STAR_CROP
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "k5.png"
        save_render(scene, png, "PNG")
        full = load(png)
    crops = [img[img.shape[0] - y - h:img.shape[0] - y, x:x + w] for img in (full, load(webp))]
    lum = [c[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32) for c in crops]
    stars, height = star_peaks(lum[0], 3 / 255)  # 3 code values: about the faintest star the eye picks out here
    stars_webp, height_webp = star_peaks(lum[1], 3 / 255)
    result = f"Star check ({w}x{h} px crop at x={x}, y={y}): {stars.sum()} stars before compression, " \
             f"{stars_webp.sum()} in the WebP"
    if stars.any():
        kept = height_webp[stars] / height[stars]  # at the same pixels
        result += (f"; the WebP keeps a median {np.median(kept):.0%} of each star's height above the sky "
                   f"(lowest tenth: {np.percentile(kept, 10):.0%})")
    log(result)
    gap = 16
    pair = np.full((h, 2 * w + gap, 4), 0.06, np.float32)
    pair[:, :w], pair[:, w + gap:] = crops
    checks = OUT / "checks"
    checks.mkdir(exist_ok=True)
    caption = "k5 stars at 100%:  left, before compression (PNG)  |  right, the final WebP (quality 95)"
    write_png(labelled(pair, caption), checks / "k5-stars-png-vs-webp.png")


def save_log_section(key, lines):
    """08-render-log.txt keeps one section per keyframe; rendering a keyframe again replaces its section."""
    sections = {}
    if LOG_PATH.exists():
        for block in LOG_PATH.read_text(encoding="utf-8").split("\n=== ")[1:]:
            sections[block.split(" ", 1)[0]] = "=== " + block.strip("\n")
    sections[key.name] = f"=== {key.name} {key.moment}, rendered {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lines)
    LOG_PATH.write_text("Time test keyframes (handoff 08). One section per keyframe, from its own run.\n" + "".join(
        f"\n{sections[name]}\n" for name in KEYS if name in sections), encoding="utf-8")


# ---------------------------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(prog="08_time_keyframes.py")
    p.add_argument("--keys", default=",".join(KEYS), help="keyframes to render, comma-separated (default: all five)")
    p.add_argument("--preview", metavar="NAME", help="1280x720 labelled tests into preview/NAME-<keyframe>.webp")
    p.add_argument("--samples", type=int, help="override the sample count")
    p.add_argument("--no-trees", action="store_true", help="skip the trees (fast look tests)")
    args = p.parse_args(argv)
    args.keys = args.keys.split(",")
    unknown = [k for k in args.keys if k not in KEYS]
    if unknown:
        p.error(f"unknown keyframe(s) {unknown}; choose from {list(KEYS)}")
    return args


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    started = time.time()
    shared = []

    def log(msg):
        print(msg, flush=True)
        shared.append(msg)

    keys = [KEYS[name] for name in args.keys]
    preview = args.preview is not None
    samples = args.samples or (porch.PREVIEW_SAMPLES if preview else porch.FINAL_SAMPLES)
    OUT.mkdir(parents=True, exist_ok=True)

    lamps = {}
    with tempfile.TemporaryDirectory() as tmp:
        for key in keys:
            lamps[key.name] = measure_lamp(key, Path(tmp)) if key.lamp_from else (0.0, (0.0, 0.0, 0.0))
    log(f"Sun lamp measured in {time.time() - started:.0f} s")
    scene, devices = build_scene(not args.no_trees, preview, samples, log)
    r = scene.render
    log(f"Scene built in {time.time() - started:.0f} s (peak RAM so far {porch.peak_ram_gb():.1f} GB)")
    log(f"Render: {r.resolution_x * r.resolution_percentage // 100}x{r.resolution_y * r.resolution_percentage // 100}, "
        f"{samples} samples, denoiser {scene.cycles.denoiser}, on {devices}; trees {'off' if args.no_trees else 'on'}")

    for key in keys:
        strength, color = lamps[key.name]
        lines = list(shared)

        def note(msg, lines=lines):
            print(msg, flush=True)
            lines.append(msg)

        apply_key(scene, key, lamps[key.name])
        if not key.lamp_from:
            lamp = "no Sun lamp"
        elif strength:
            lamp = (f"Sun lamp measured at {LAMP_ALTITUDE[key.lamp_from]:.0f} m ({key.lamp_from}): {strength:.2f} x "
                    f"({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
        else:
            lamp = f"Sun lamp measured at {LAMP_ALTITUDE[key.lamp_from]:.0f} m: no sunlight left, so no lamp"
        clouds = (f"clouds cover {key.cloud_cover:g}, density {key.cloud_density:g}, fading out "
                  f"{CLOUD_FADE[0] / 1000:g}-{CLOUD_FADE[1] / 1000:g} km" if key.cloud_cover is not None else "no clouds")
        note(f"Sun {key.sun:+.1f} deg up, {porch.SUN_YAW:g} deg yaw; {lamp}")
        note(f"Exposure {key.exposure:+.2f} EV; {clouds}; porch light {'on' if key.porch_light else 'off'} "
             f"({PORCH_LIGHT_POWER:g} W, {PORCH_LIGHT_KELVIN:.0f} K); saturation lift above "
             f"{porch.GRADE_MASK[0]:.3g}-{porch.GRADE_MASK[1]:.3g}")
        seconds, peak_mb = render(scene)
        note(f"Render time: {seconds:.1f} s")
        note(f"Cycles peak memory (highest reported during the render): {peak_mb:,.0f} MB; "
             f"peak RAM so far in this run: {porch.peak_ram_gb():.1f} GB")
        if preview:
            folder = OUT / "preview"
            folder.mkdir(exist_ok=True)
            path = folder / f"{args.preview}-{key.file}.webp"
            save_render(scene, path, "WEBP", 95)
            porch.write_pixels(labelled(load(path), f"{args.preview}   ·   {key_label(key)}"), path, quality=90)
            note(f"Wrote preview/{path.name}")
            with open(folder / "preview-log.txt", "a", encoding="utf-8") as f:
                f.write(f"--- {args.preview} {key.name} {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lines) + "\n")
        else:
            path = OUT / f"{key.file}.webp"
            save_render(scene, path, "WEBP", 95)
            note(f"Wrote {path.name}")
            px = load(path)[..., :3]
            note(f"Clipping: {(px.max(-1) >= 0.995).mean():.3%} of pixels at full white, "
                 f"{(px.max(-1) <= 0.005).mean():.3%} at full black")
            if key.name == "k5":
                star_check(scene, path, note)
            note(f"Run time so far: {time.time() - started:.0f} s")
            save_log_section(key, lines)
    if not preview:
        contact_sheet()
        print("Wrote 08-keyframes.webp", flush=True)


if __name__ == "__main__":
    main()
