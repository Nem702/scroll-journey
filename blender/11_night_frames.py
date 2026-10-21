r"""Handoff 11: the time-test frames again, with Nems's night picks (Oct 19) and a longer night.

09's frames from golden hour to night (blender/09_time_frames.py, loaded as a module), with:
- two of 10's lantern sconces (blender/10_night_options.py, loaded as a module), on the inner side of posts 2 and 4
  counting from the camera, facing the deck and the cabin. Frosted glass, so lit they glow softly with no bulb in view,
  and each lights a wide pool on the deck. They're lit from where 09's lantern came on (that lantern stays off) and
  show unlit in daylight;
- 10's vivid aurora, fading in as in 10. Its phase steps every frame, so the curtains' folds drift and the rays
  reshape. The rays lean toward a point overhead and are less evenly spaced (10's report flagged both);
- 10's night glow at half strength;
- a longer night: from sun -8 degrees (09's frame 52), where the vivid aurora shows clearly, the frames come closer
  together, so that stretch takes a quarter of the scroll. The frames before it keep 09's spacing. (The vivid aurora
  shows faintly from blue hour, frame 45, so counted from there the night takes a third.)
Two options for Nems to judge, off by default: a faint warm glow from a cabin window behind the camera, on from dusk,
and version B: nothing on the rail, two mugs on a porch table by post 2, in its lamp's pool, with a stool each side.
05, 08, 09 and 10 produce what they did before.

Run headless from the repo root (Blender isn't on PATH, so call it by full path):
    & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender\11_night_frames.py -- --stills --frames 1,40,69
    ... -P blender\11_night_frames.py -- --stills --frames 69-window,1-table2,40-table2
    ... -P blender\11_night_frames.py -- --minutes 100   # all frames; start no new frame after 100 minutes
    ... -P blender\11_night_frames.py -- --preview NAME --frames 50,52-window [--no-trees] [--samples N]

--frames takes frame numbers, each optionally with options: -window (the window glow, from dusk) and -table2 (version B).
--stills renders them full size into renders/review/11/ (f0069-window.webp ..., and 11-stills-log.txt with one line per
still). Without it, a run renders the frames missing from renders/time/master-11/ (lossless PNGs, beside 09's masters,
which stay as they are) and keeps the ones already there, so a cut-off run picks up where it stopped. All of them take
about 2.5 hours. renders/time/11-render-log.txt has one line per frame. Once every master exists, the run encodes the web
frames into public/frames/time/1440/ and 1080/ (replacing 09's) and checks the darkest frame's faint stars
(renders/review/11/checks/). Previews (1280x720, labelled) go to renders/review/11/preview/. Nothing in
renders/review/05/ to 10/ is written.
"""

import argparse
import dataclasses
import importlib.util
import math
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent
MASTERS = ROOT / "renders" / "time" / "master-11"
LOG_PATH = ROOT / "renders" / "time" / "11-render-log.txt"
REVIEW = ROOT / "renders" / "review" / "11"
STILLS_LOG = REVIEW / "11-stills-log.txt"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.dont_write_bytecode = True  # no __pycache__ next to the scripts
# 09: the frame table, the cloud drift, the Sun lamp per frame, the web encode and the star check. It loads 08 and 05
# (the scene); 08 switches off 05's downloader, so this run never touches the network. No main() runs.
frames09 = _load("time_frames", "09_time_frames.py")
kf, porch = frames09.kf, frames09.porch
# 10: the aurora and the sconce. It loads its own copies of 08 and 05; pointing it at 09's keeps one scene and one set
# of settings.
night10 = _load("night_options", "10_night_options.py")
night10.kf, night10.porch = kf, porch
kf.AIRGLOW = tuple(c * night10.NIGHT_GLOW for c in kf.AIRGLOW)  # 10's night glow at half strength (08 reads it)
# 09's encode and star check read the masters from, and write the star crops to, these.
frames09.MASTERS, frames09.REVIEW = MASTERS, REVIEW
kf.OUT = porch.OUT = REVIEW


# ---------------------------------------------------------------------------------------------
# The frames.

# Spacing in sun height: 09's pace (sun height, then the step between frames, in degrees) down to NIGHT_FROM, one of its
# points, where the vivid aurora shows clearly. Below it the steps are NIGHT_PACE of 09's, easing in toward -12 degrees.
# That makes 69 frames: the stretch from frame 52 (the first at -8) to the last takes 17 of the 68 steps, a quarter of
# the scroll (9 of 60 in 09), and frames 1-52 sit within 0.09 degrees of 09's. The page keeps 09's scroll distance per
# frame, so the evening before the aurora feels the same.
NIGHT_FROM = -8.0
NIGHT_PACE = 0.43
frames09.PACE = tuple((sun, step * (NIGHT_PACE if sun < NIGHT_FROM else 1.0)) for sun, step in frames09.PACE)

# The aurora: 10's vivid look and fade-in (from frame 40 to full strength at blue hour). 10's add_aurora options: the
# rays lean toward a point straight ahead, 13 degrees from overhead (as real rays point to the magnetic zenith), and
# crowd together in places; the folds and bright patches drift along the curtains as the phase steps.
AURORA_LOOK = "vivid"
AURORA_RAYS = {"lean_toward": (45.0, 77.0), "bend": 3.0, "drift": 1.5}
# The phase steps this much per frame from the frame the aurora starts fading in. Neighbouring frames then share about
# 95% of their rays' pattern, so a crossfade between them keeps about 99% of the rays' contrast and never shows a ray
# twice; the folds slide 0.12 degrees (about 4 px) a frame.
AURORA_STEP = 0.08

# Two of 10's sconces (its fixture a), on the inner side of posts 2 and 4 counting from the camera, facing the deck and
# the cabin as 10's did on post 1. Both are lit when 09's lantern would be (sun below -3.5 degrees, frame 40). Changed
# from 10's (Nems, round 2):
# - milky frosted glass: lit, the panes glow softly and evenly and the bulb isn't seen. The glow is set for looks, apart
#   from the lights, so the lantern reads as a warm glow rather than a white blob. A quarter of the light gets through
#   the glass: clearer frosted glass glowed with the bright sky behind it at golden hour, as if lit;
# - a wider pool of warm light on the deck: a spot light in each lantern aims down, tipped away from the post, beside
#   10's 1.6 W point light, which still lights the post, the rail and the ceiling around the lantern.
SCONCE_POSTS = (2, 4)
POST_HALF = 0.07  # 05's posts are 14 cm square
FROSTED_ROUGHNESS = 0.5
FROSTED_TRANSMISSION = 0.25
FROSTED_GLOW = 0.1  # radiance of a lit pane
POOL_POWER = 4.5  # W
POOL_CONE = 110.0  # degrees across, softened all the way to the middle
POOL_TILT = 15.0  # degrees from straight down, away from the post
SOFT_LIGHT = 0.05  # m: the lights' size. The frosted lantern, not a bare bulb, casts the shadows, so they're softer

# Option: a warm, very faint glow from a window in 05's cabin wall behind the camera (never in view), facing the porch,
# on from dusk with the sconces, so the mug and the near rail don't sink into black.
WINDOW_GLOW = False  # for the full run; a still asks for it with -window
WINDOW_AT = (0.6, porch.DECK_Y[0] + 0.01, 1.5)  # centre, just in front of the wall
WINDOW_SIZE = (1.2, 1.3)  # m
WINDOW_POWER = 1.0  # W
WINDOW_KELVIN = 2500.0

# Option, version B (round 2 of it, "table2"): nothing on the rail; a small rustic porch table by the rail just right of
# post 2, in its lamp's pool, with two mugs (05's mug twice, at different angles) and a plain round stool each side,
# facing the view. All of it is in the porch's rail wood, below the rail top, and inside x 2432 of the frame (the page
# crops the sides on screens narrower than 16:9).
MUG_ON_TABLE = True  # Nems, Oct 20: version B (table2) for the full run; window glow off
TABLE_AT = (5.05, -0.42)  # m: the table's centre on the deck
TABLE_SIZE = (0.80, 0.50, 0.72)  # length along the rail, depth and the top's height, m
# Each mug: x, y from the top's middle, and its turn, degrees. The camera looks along the table, so they're spread across
# it (one toward the rail, one toward the deck) to stand apart in the frame.
MUGS_ON_TOP = ((-0.10, 0.10, -35.0), (0.12, -0.10, 20.0))
# The stools, x along the rail and y toward the rail from the table's centre (m), one each side. The far one sits a
# little toward the deck: straight beyond the table's end, the table hid it from the camera.
STOOLS_AT = ((-0.64, 0.0), (0.55, -0.35))
STOOL_SIZE = (0.32, 0.46)  # seat diameter and height, m


@dataclasses.dataclass(frozen=True)
class Frame:
    key: object  # 09's settings for the frame (an 08 Key); its porch_light switches the sconces
    aurora: float  # strength, 0 to 1
    phase: float
    window: bool  # the window glow on
    table: bool  # version B: two mugs on the table, stools, nothing on the rail

    @property
    def name(self):
        return self.key.name


def frame_table():
    """Every frame: 09's settings with 11's pace, the aurora's strength and phase, and the options."""
    keys = frames09.frame_settings()
    first = next(int(k.name) for k in keys if night10.aurora_strength(k.sun) > 0)
    return [Frame(k, night10.aurora_strength(k.sun), round(AURORA_STEP * max(0, int(k.name) - first), 6),
                  WINDOW_GLOW and k.porch_light, MUG_ON_TABLE) for k in keys]


def with_options(frame, options):
    """The frame with options asked for by name: window (from dusk, with the sconces) and table2."""
    return dataclasses.replace(frame, window=frame.window or ("window" in options and frame.key.porch_light),
                               table=frame.table or "table2" in options)


def settings_text(frame):
    """The computed settings as the logs show them. A master on disk must match its frame's text."""
    key = frame.key
    clouds = (f"clouds {key.cloud_cover:.3f} / {key.cloud_density:.5f}, drift {frames09.drift(key):.0f} m"
              if key.cloud_cover is not None else "no clouds")
    aurora = f"aurora {frame.aurora:.3f} at phase {frame.phase:.2f}" if frame.aurora else "no aurora"
    return (f"sun {key.sun:+.3f} | exposure {key.exposure:+.3f} EV | {clouds} | "
            f"sconces {'on' if key.porch_light else 'off'} | {aurora} | window glow {'on' if frame.window else 'off'} | "
            f"mug on the {'table' if frame.table else 'rail'}")


def lamp_text(frame, lamp):
    strength, color = lamp
    if not frame.key.lamp_from:
        return "no Sun lamp"
    if not strength:
        return "Sun lamp: no sunlight left"
    return (f"Sun lamp {strength:.2f} x ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}) from the "
            f"{frames09.lamp_where(frame.key)}")


def frame_line(frame, lamp, seconds, peak_mb):
    return (f"{frame.name} {frame.key.moment:2} | {settings_text(frame)} | {lamp_text(frame, lamp)} | "
            f"render {seconds:.1f} s | Cycles {peak_mb:,.0f} MB | RAM peak {porch.peak_ram_gb():.1f} GB | "
            f"{time.strftime('%Y-%m-%d %H:%M')}")


def write_log(path, header, lines, summary=()):
    text = header + "\n" + "".join(lines[n] + "\n" for n in sorted(lines))
    if summary:
        text += "\n" + "".join(s + "\n" for s in summary)
    path.write_text(text, encoding="utf-8")


def check_masters(frames, lines):
    """Stops if a master on disk was rendered with other settings than the frame table now gives, so a run never mixes
    two versions of the table."""
    table = {int(f.name): f for f in frames}
    for path in sorted(MASTERS.glob("[0-9][0-9][0-9][0-9].png")):
        frame, line = table.get(int(path.stem)), lines.get(int(path.stem), "")
        lamp_ok = frame is not None and (
            "no Sun lamp" in line if not frame.key.lamp_from
            else f"from the {frames09.lamp_where(frame.key)} |" in line or "no sunlight left" in line)
        if not lamp_ok or f"| {settings_text(frame)} |" not in line:
            raise SystemExit(f"{path} doesn't match the frame table (its log line: {line or 'missing'!r}). Delete the "
                             "masters that differ, or the whole folder, to render them again.")


# ---------------------------------------------------------------------------------------------
# The scene.

@dataclasses.dataclass
class Extras:
    """What 11 adds to 08's scene, switched per frame by show_frame()."""
    sconces: list  # each sconce's objects, for screen_box()
    lights: list  # the sconces' lights
    pane_glow: object  # the frosted panes' emission strength
    window: object  # the window glow's light
    table: list  # everything only version B shows: the table, the stools and the second mug
    mugs: list  # 05's mug, then the second one
    mug_on: dict  # where the mugs stand, (location, turn) each: "rail" (05's mug only) and "table" (both)


def build_sconces():
    """10's sconce on the inner side of each of SCONCE_POSTS, with frosted panes and a pool light. Returns each sconce's
    objects, all their lights and the panes' emission strength."""
    mats = night10.fixture_materials()
    frosted, glow = night10.glowing_material("sconce_frosted", (0.92, 0.90, 0.86), FROSTED_ROUGHNESS,
                                             night10.LIGHTS["a"].kelvin,
                                             **{"Transmission Weight": FROSTED_TRANSMISSION})
    mats["glass"] = frosted  # the four panes and the floor
    # The lanterns' own lights skip the frosted glass: lit from 6 cm away it would glow white, so its glow is set here.
    glass = bpy.data.collections.new("sconce_glass")  # not in the scene: only the lights' light linking uses it
    sconces, lights = [], []
    for post in SCONCE_POSTS:
        sconce = night10.build_sconce(mats)
        # The sconce's origin, where its back plate meets the post at the lantern's base: on the post's inner face.
        sconce.parts[0].parent.location = (porch.POSTS_X[post - 1], -POST_HALF, night10.SCONCE_AT[2])
        for part in sconce.parts:
            if part.name.startswith("sconce_bulb"):
                part.hide_render = True  # the frosted panes glow instead
            elif frosted in part.data.materials[:]:
                night10.glow_only(part)  # seen only by the camera and through glass: the lights do the lighting
                glass.objects.link(part)
        point = sconce.light
        point.data.shadow_soft_size = SOFT_LIGHT
        spot = bpy.data.lights.new("sconce_pool", "SPOT")
        spot.energy = POOL_POWER
        spot.use_temperature, spot.temperature = True, night10.LIGHTS["a"].kelvin
        spot.spot_size, spot.spot_blend = math.radians(POOL_CONE), 1.0
        spot.shadow_soft_size = SOFT_LIGHT
        pool = porch.link_object(bpy.data.objects.new("sconce_pool", spot), point.users_collection[0])
        pool.parent, pool.location = point.parent, point.location  # where the bulb was, inside the lantern
        pool.rotation_euler.x = -math.radians(POOL_TILT)  # a spot shines down its local -Z: tipped toward -Y, the deck
        pool.visible_transmission = False
        sconces.append(sconce.parts)
        lights += [point, pool]
    for item in glass.collection_objects:
        item.light_linking.link_state = "EXCLUDE"
    for light in lights:
        light.light_linking.receiver_collection = glass
    return sconces, lights, glow


def build_window():
    light = bpy.data.lights.new("window_glow", "AREA")
    light.shape = "RECTANGLE"
    light.size, light.size_y = WINDOW_SIZE
    light.energy = WINDOW_POWER
    light.use_temperature, light.temperature = True, WINDOW_KELVIN
    obj = porch.link_object(bpy.data.objects.new("window_glow", light))
    obj.location = WINDOW_AT
    obj.rotation_euler.x = math.radians(90)  # an area light shines down its local -Z: turned to face +Y, the porch
    return obj


def build_table():
    """Version B: the table (a top of four planks with small gaps, an apron and four square legs), a plain round stool
    each side, and a second copy of 05's mug, each board with the grain along it as on the porch. Returns every object
    shown only in version B, and where each mug stands (location, turn in radians)."""
    wood = bpy.data.materials["rail"]  # 05's porch wood
    coll = night10.new_collection("table")
    root = night10.empty("table", (*TABLE_AT, 0.0), coll)
    rng = random.Random(porch.SEED)
    up = math.radians(90)
    length, depth, height = TABLE_SIZE
    plank, gap, leg, inset, apron = 0.035, 0.008, 0.06, 0.04, 0.08
    plank_w = (depth - 3 * gap) / 4
    parts = [porch.box(f"table_top_{i}", (length, plank_w, plank),
                       (0.0, -depth / 2 + plank_w / 2 + i * (plank_w + gap), height - plank / 2),
                       (0.0, 0.0, rng.uniform(-0.006, 0.006)), wood, 0.004, coll) for i in range(4)]
    lx, ly = length / 2 - inset - leg / 2, depth / 2 - inset - leg / 2
    parts += [porch.box(f"table_leg_{i}", (height - plank, leg, leg), (sx * lx, sy * ly, (height - plank) / 2),
                        (0.0, up, 0.0), wood, 0.004, coll)
              for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1)))]
    z = height - plank - apron / 2
    parts += [porch.box(f"table_apron_{i}", (2 * (lx if along_x else ly) - leg, 0.022, apron),
                        (0.0, s * ly, z) if along_x else (s * lx, 0.0, z), (0.0, 0.0, 0.0 if along_x else up), wood,
                        0.003, coll)
              for i, (along_x, s) in enumerate(((True, -1), (True, 1), (False, -1), (False, 1)))]
    # The stools: a round seat with a softened edge on four straight square legs, turned a little each.
    radius, seat_h = STOOL_SIZE[0] / 2, STOOL_SIZE[1]
    seat = ((0.0, seat_h), (radius - 0.008, seat_h), (radius, seat_h - 0.007), (radius, seat_h - 0.03),
            (radius - 0.006, seat_h - 0.036), (0.0, seat_h - 0.036))
    for i, (x, y) in enumerate(STOOLS_AT):
        stool = night10.empty(f"stool_{i}", (x, y, 0.0), coll, root)
        stool.rotation_euler.z = rng.uniform(0.0, math.tau)
        legs = [porch.box(f"stool_{i}_leg_{j}", (seat_h - 0.036, 0.034, 0.034),
                          (0.095 * math.cos(a), 0.095 * math.sin(a), (seat_h - 0.036) / 2), (0.0, up, a), wood, 0.003,
                          coll) for j, a in enumerate(math.radians(45 + 90 * k) for k in range(4))]
        for obj in [night10.lathe(f"stool_{i}_seat", seat, wood, coll, 48), *legs]:
            obj.parent = stool
            parts.append(obj)
    for obj in parts:
        if obj.parent is None:
            obj.parent = root
    mug2 = porch.build_mug()  # a second mug, as 05 builds it, with its handle and coffee
    parts += [mug2, *mug2.children]
    bpy.context.view_layer.update()
    spots = [((root.matrix_world @ Vector((x, y, height))).to_tuple(), math.radians(turn)) for x, y, turn in MUGS_ON_TOP]
    return parts, mug2, spots


def measure_lamps(frames):
    """Each frame's Sun lamp, as 09 measures it. Each measurement resets Blender, so this runs before the build."""
    with tempfile.TemporaryDirectory() as tmp:
        return {f.name: frames09.measure_lamp(f.key, Path(tmp)) for f in frames}


def build(trees, preview, samples, log):
    scene, devices = kf.build_scene(trees, preview, samples, log)
    sconces, lights, pane_glow = build_sconces()
    mug = bpy.data.objects["mug"]
    rail = [(mug.location.to_tuple(), mug.rotation_euler.z)]
    table, mug2, on_table = build_table()
    extras = Extras(sconces, lights, pane_glow, build_window(), table, [mug, mug2], {"rail": rail, "table": on_table})
    bpy.context.view_layer.update()  # world positions of the parented parts, for screen_box()
    r = scene.render
    log(f"Render: {r.resolution_x * r.resolution_percentage // 100}x{r.resolution_y * r.resolution_percentage // 100}, "
        f"{samples} samples, denoiser {scene.cycles.denoiser}, on {devices}; trees {'on' if trees else 'off'}; peak RAM "
        f"so far {porch.peak_ram_gb():.1f} GB")
    log("In the frame (px of 2560x1440): " + "; ".join(
        f"sconce on post {post} x {x0}-{x1}, y {y0}-{y1}" for post, (x0, x1, y0, y1)
        in zip(SCONCE_POSTS, (night10.screen_box(scene, parts) for parts in sconces), strict=True))
    )
    for obj, (at, turn) in zip(extras.mugs, on_table, strict=True):
        obj.location, obj.rotation_euler.z = at, turn
    bpy.context.view_layer.update()
    log("Version B in the frame: table, stools and mugs x {}-{}, y {}-{}; ".format(*night10.screen_box(scene, table))
        + "; ".join("mug x {}-{}, y {}-{}".format(*night10.screen_box(scene, [m])) for m in extras.mugs))
    return scene, extras


def show_frame(scene, extras, frame, lamp):
    """09's settings for the frame (with 08's lantern out of frame left off), the sconces lit or not, the aurora, and
    the options."""
    frames09.apply_frame(scene, dataclasses.replace(frame.key, porch_light=False), lamp)
    lit = frame.key.porch_light
    for light in extras.lights:
        light.hide_render = not lit
    extras.pane_glow.default_value = FROSTED_GLOW if lit else 0.0
    extras.window.hide_render = not frame.window
    for obj in extras.table:
        obj.hide_render = not frame.table
    for obj, (at, turn) in zip(extras.mugs, extras.mug_on["table" if frame.table else "rail"], strict=False):
        obj.location, obj.rotation_euler.z = at, turn
    if frame.aurora:
        night10.add_aurora(scene.world, night10.LOOKS[AURORA_LOOK], frame.aurora, frame.phase, **AURORA_RAYS)


# ---------------------------------------------------------------------------------------------
# Runs.

def quality(frame):
    """09's WebP quality: 98 from k3 on keeps the faint stars."""
    return frames09.NIGHT_QUALITY if frame.key.sun <= frames09.KEYS[2].sun else frames09.DAY_QUALITY


def still_name(frame, options):
    return f"f{frame.name}" + "".join(f"-{o}" for o in options)


def render_review(wanted, args, started):
    """Full-size stills into renders/review/11/, or labelled 1280x720 previews into its preview/ folder. wanted: (frame,
    options) pairs."""
    preview = args.preview is not None
    samples = args.samples or (porch.PREVIEW_SAMPLES if preview else porch.FINAL_SAMPLES)
    folder = REVIEW / "preview" if preview else REVIEW
    folder.mkdir(parents=True, exist_ok=True)
    setup = []

    def log(msg):
        print(msg, flush=True)
        setup.append(msg)

    lamps = measure_lamps(list({f.name: f for f, _ in wanted}.values()))
    scene, extras = build(not args.no_trees, preview, samples, log)
    log(f"Scene built in {time.time() - started:.0f} s")
    lines = {}
    if not preview and STILLS_LOG.exists():
        lines = {m[1]: line for line in STILLS_LOG.read_text(encoding="utf-8").splitlines()
                 if (m := re.match(r"(f\d{4}[a-z0-9-]*) \| ", line))}
    for frame, options in wanted:
        frame = with_options(frame, options)
        show_frame(scene, extras, frame, lamps[frame.name])
        seconds, peak_mb = kf.render(scene)
        name = still_name(frame, options)
        line = f"{name} | {frame_line(frame, lamps[frame.name], seconds, peak_mb)}"
        print(line, flush=True)
        if preview:
            path = folder / f"{args.preview}-{name}.webp"
            kf.save_render(scene, path, "WEBP", 95)
            label = f"{args.preview}   ·   {name} {frame.key.moment}   ·   {settings_text(frame)}"
            porch.write_pixels(kf.labelled(kf.load(path), label), path, quality=90)
            with open(folder / "preview-log.txt", "a", encoding="utf-8") as f:
                f.write(f"--- {args.preview} {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(setup) + f"\n{line}\n")
            continue
        kf.save_render(scene, folder / f"{name}.webp", "WEBP", quality(frame))
        lines[name] = line
        write_log(STILLS_LOG, "Review stills (handoff 11), 2560x1440: one line per still, from the run that rendered "
                  "it. Last run's setup:\n" + "\n".join(setup) + "\n", lines)
    print(f"Run time: {time.time() - started:.0f} s", flush=True)


def render_masters(frames, args, started):
    """The lossless masters missing from renders/time/master-11/, then (once all exist) the web frames and star check."""
    MASTERS.mkdir(parents=True, exist_ok=True)
    lines = frames09.read_log(LOG_PATH)
    check_masters(frames, lines)
    todo = [f for f in frames if not (MASTERS / f"{f.name}.png").exists()]
    header = (f"Time test frames (handoff 11), 2560x1440, trees on, {porch.FINAL_SAMPLES} samples: one line per frame, "
              "rewritten after every frame.")
    keyframes = ", ".join(f"{f.key.moment} {f.name}" for f in frames if f.key.moment)
    print(f"{len(frames)} frames (keyframes: {keyframes}); {len(frames) - len(todo)} already rendered, {len(todo)} to go",
          flush=True)
    if todo:
        lamps = measure_lamps(todo)
        print(f"Sun lamp measured for {len(todo)} frames in {time.time() - started:.0f} s", flush=True)
        scene, extras = build(True, False, porch.FINAL_SAMPLES, lambda msg: print(msg, flush=True))
        print(f"Scene built in {time.time() - started:.0f} s", flush=True)
        for i, frame in enumerate(todo):
            if args.minutes is not None and time.time() - started > args.minutes * 60:
                print(f"Stopped after {args.minutes:g} minutes with {len(todo) - i} frames left: run again to "
                      "continue.", flush=True)
                return
            if args.max_frames is not None and i >= args.max_frames:  # RAM creeps up ~0.1 GB a frame; a fresh run resets it
                print(f"Stopped after {i} frames with {len(todo) - i} left: run again to continue.", flush=True)
                return
            show_frame(scene, extras, frame, lamps[frame.name])
            seconds, peak_mb = kf.render(scene)
            part = MASTERS / f"{frame.name}.part.png"  # renamed once complete: a cut-off run never leaves a broken one
            kf.save_render(scene, part, "PNG")
            lines[int(frame.name)] = frame_line(frame, lamps[frame.name], seconds, peak_mb)
            write_log(LOG_PATH, header, lines)
            part.replace(MASTERS / f"{frame.name}.png")
            print(lines[int(frame.name)], flush=True)
    summary = frames09.encode_web([f.key for f in frames])
    write_log(LOG_PATH, header, lines, summary)
    for line in summary:
        print(line, flush=True)
    print(f"Run time: {time.time() - started:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------

OPTIONS = ("window", "table2")


def parse_args(argv):
    p = argparse.ArgumentParser(prog="11_night_frames.py")
    p.add_argument("--stills", action="store_true", help="full-size review stills of --frames into renders/review/11/")
    p.add_argument("--preview", metavar="NAME", help="1280x720 labelled tests of --frames into renders/review/11/preview/")
    p.add_argument("--frames", help="frame numbers, comma-separated, each optionally with -window and/or -table2 (for "
                                    "--stills and --preview)")
    p.add_argument("--no-trees", action="store_true", help="skip the trees (previews only)")
    p.add_argument("--samples", type=int, help="override the sample count (previews only)")
    p.add_argument("--minutes", type=float, help="start no new frame after this many minutes (default: render all)")
    p.add_argument("--max-frames", type=int, help="render at most this many masters, then stop (default: all)")
    args = p.parse_args(argv)
    review = args.stills or args.preview is not None
    if args.stills and args.preview is not None:
        p.error("--stills and --preview are separate runs")
    if review != (args.frames is not None):
        p.error("--frames goes with --stills or --preview, and they need it")
    if (args.no_trees or args.samples) and args.preview is None:
        p.error("--no-trees and --samples are for previews")
    if args.frames is not None:
        args.frames = [item.split("-") for item in args.frames.split(",")]
        for number, *options in args.frames:
            if not number.isdigit() or any(o not in OPTIONS for o in options):
                p.error(f"--frames items are numbers with -window or -table2: {'-'.join([number, *options])!r}")
    return args


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    started = time.time()
    frames = frame_table()
    if args.frames is None:
        render_masters(frames, args, started)
        return
    by_number = {int(f.name): f for f in frames}
    missing = [int(n) for n, *_ in args.frames if int(n) not in by_number]
    if missing:
        raise SystemExit(f"No frame(s) {missing}: the frames are 1 to {len(frames)}")
    render_review([(by_number[int(n)], options) for n, *options in args.frames], args, started)


if __name__ == "__main__":
    main()
