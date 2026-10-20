r"""Handoff 09: time test, step 2. The porch view in 61 frames from golden hour to night, for the scrub page.

The scene and the five approved keyframes come from blender/08_time_keyframes.py, loaded as a module and left
unchanged (it loads 05 the same way). Between the keyframes the exposure and the clouds move in step with how much the
sky has darkened, the Sun lamp is measured for every frame, and the clouds drift slowly toward the camera.

Run headless from the repo root (Blender isn't on PATH, so call it by full path):
    & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender\09_time_frames.py
    ... -P blender\09_time_frames.py -- --minutes 100   # start no new frame after 100 minutes (ends by about 104)
    ... -P blender\09_time_frames.py -- --preview       # 1280x720 without trees, into renders/review/09/preview/

A run renders the frames missing from renders/time/master/ (lossless 8-bit PNGs, 0001.png to 0061.png) and keeps the
ones already there, so a cut-off run picks up where it stopped. All 61 take about 2.2 hours.
renders/time/09-render-log.txt has one line per frame. Once every master exists, the run encodes the web frames
(public/frames/time/1440/0001.webp ... and public/frames/time/1080/...) and checks how many of the darkest frame's faint
stars the compression keeps (renders/review/09/checks/). Nothing in renders/review/05/ to 08/ is written.
"""

import argparse
import dataclasses
import importlib.util
import math
import re
import sys
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MASTERS = ROOT / "renders" / "time" / "master"
LOG_PATH = ROOT / "renders" / "time" / "09-render-log.txt"
REVIEW = ROOT / "renders" / "review" / "09"
WEB = ROOT / "public" / "frames" / "time"
WEB_SETS = {"1440": (2560, 1440), "1080": (1920, 1080)}

# The approved keyframes and the scene. 08's main() doesn't run. Loading it also loads 05 and switches off 05's
# downloader, so this run never touches the network.
sys.dont_write_bytecode = True  # no __pycache__ next to the scripts
_spec = importlib.util.spec_from_file_location("time_keyframes", Path(__file__).with_name("08_time_keyframes.py"))
kf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kf)
porch = kf.porch
kf.OUT = porch.OUT = REVIEW  # 07's and 08's folders are never written from here
KEYS = list(kf.KEYS.values())  # k1 golden hour ... k5 night


# ---------------------------------------------------------------------------------------------
# The frames.

# Frame spacing: the step in sun height between neighbouring frames, by sun height (degrees, linear in between).
# - Small while the sun's disc is in view. It moves 36.5 px per degree at 1440p and is about 19 px across, so a
#   crossfade between neighbours reads as one slightly longer sun. Smallest (0.1) while the disc slips behind the ridge.
# - 0.24-0.36 from sunset to blue hour, where the sky darkens fastest (about a stop per degree).
# - Long in deep night, where the constant night glow is most of the sky and little changes.
# Each keyframe-to-keyframe stretch gets a whole number of steps (13, 18, 13 and 16), so the keyframes are frames 1,
# 14, 32, 45 and 61.
PACE = ((4.5, 0.20), (2.75, 0.11), (2.15, 0.10), (1.4, 0.30), (0.5, 0.36), (-1.5, 0.24), (-5.0, 0.30), (-8.0, 0.62),
        (-12.0, 1.3), (-18.0, 3.6), (-30.0, 12.0))

# How bright the sky is at each sun height, night glow and stars included: the log-average of the camera's view of the
# sky alone (no scene), as 08's planning probe measured it (renders/review/08/plan-probes/), at more heights. Below
# about -15 degrees the night glow is most of it (0.00303 on its own), so the sky stops darkening there.
SKY_KEY = ((4.5, 5.137), (4.0, 4.671), (3.5, 4.174), (3.0, 3.652), (2.7, 3.331), (2.6, 3.222), (2.2, 2.787),
           (1.8, 2.354), (1.4, 1.933), (1.0, 1.535), (0.6, 1.172), (0.3, 0.9351), (0.0, 0.72), (-0.5, 0.4212),
           (-1.0, 0.2413), (-1.5, 0.1752), (-2.0, 0.1422), (-2.5, 0.1161), (-3.0, 0.09456), (-3.5, 0.07671),
           (-4.0, 0.06206), (-4.5, 0.05022), (-5.0, 0.04079), (-6.0, 0.0276), (-7.0, 0.01958), (-8.0, 0.01461),
           (-9.0, 0.01136), (-10.0, 0.00914), (-12.0, 0.006398), (-15.0, 0.004434), (-18.0, 0.003618),
           (-21.0, 0.003278), (-24.0, 0.003137), (-27.0, 0.003078), (-30.0, 0.003053))

# The Sun lamp is measured where 08 measured it: from the porch while the sun's disc can still reach it, at the cloud
# base once only the clouds and the high haze see the sun. Between these heights (the disc is fully behind the ridge at
# 2.2) the two measured lights are mixed in linear light, on a smoothstep. The porch is in the ridge's shadow by then,
# so only the clouds see the change. From k4 on there's no lamp, as in 08; it has faded out by about -3 degrees.
LAMP_BLEND = (2.2, 0.6)
LANTERN_BELOW = -3.5  # the porch lantern switches on here (frame 40), two frames after the clouds lose their last pink
CLOUDS_GONE = (-10.0, 0.72)  # after k4 the clouds thin out (cover up to 0.72, density down to 0) and are gone here
CLOUD_DRIFT = 50.0  # metres per frame from the sunset side toward the camera: about 2 px a frame at the top at 1440p

# WebP quality: 95 like the signed-off stills, then 98 from k3 on, as the sky darkens toward the first stars. 98 keeps
# the most faint stars of Blender's lossy settings (95 drops about half of them; 99 keeps fewer than 98).
DAY_QUALITY, NIGHT_QUALITY = 95, 98
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def sun_heights():
    """Sun height per frame (frame 1 first, rounded to 0.001 degrees) and the frame number of each keyframe. Each
    keyframe-to-keyframe stretch gets as many steps as PACE fits into it, rounded, spread to follow PACE's shape."""
    pace_at, pace = np.array(PACE).T[:, ::-1]  # rising sun height, for np.interp
    heights, key_frames = [KEYS[0].sun], [1]
    for a, b in pairwise(KEYS):
        grid = np.linspace(a.sun, b.sun, 4001)
        step = np.interp(grid, pace_at, pace)
        steps = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(grid)) / ((step[:-1] + step[1:]) / 2))])
        n = max(1, round(steps[-1]))
        part = np.interp(np.arange(1, n + 1) * steps[-1] / n, steps, grid)
        part[-1] = b.sun
        heights += [round(float(s), 3) for s in part]
        key_frames.append(len(heights))
    return heights, key_frames


def sky_stops(sun):
    """The sky's brightness at this sun height, in stops (log2 of SKY_KEY, linear in between)."""
    at, key = np.array(SKY_KEY).T[:, ::-1]
    return float(np.interp(sun, at, np.log2(key)))


def darkened(sun, a, b):
    """How far the sky has darkened at this sun height, from sun height a (0) to sun height b (1)."""
    return (sky_stops(a) - sky_stops(sun)) / (sky_stops(a) - sky_stops(b))


def lerp(a, b, t):
    return a * (1.0 - t) + b * t  # exactly a at t = 0 and exactly b at t = 1


def cloud_share(sun):
    """The cloud base's share of the Sun lamp (0: measured from the porch only)."""
    return float(porch.smoothstep(*LAMP_BLEND, sun))


def frame_settings():
    """Every frame's settings as an 08 Key; name is the frame number, moment its keyframe if it is one. Keyframe frames
    get exactly the approved values. In between, the exposure (in stops) and the clouds move in step with how much the
    sky has darkened, so they barely change where the sky barely does."""
    heights, key_frames = sun_heights()
    frames = []
    for n, sun in enumerate(heights, 1):
        i = max(j for j in range(len(KEYS) - 1) if key_frames[j] <= n)
        a, b = KEYS[i], KEYS[i + 1]
        t = darkened(sun, a.sun, b.sun)
        if b.cloud_cover is not None:
            cover, density = lerp(a.cloud_cover, b.cloud_cover, t), lerp(a.cloud_density, b.cloud_density, t)
        elif sun > CLOUDS_GONE[0]:  # k4 to night: the clouds thin out instead of jumping to k5's clear sky
            gone = darkened(sun, a.sun, CLOUDS_GONE[0])
            cover, density = lerp(a.cloud_cover, CLOUDS_GONE[1], gone), lerp(a.cloud_density, 0.0, gone)
        else:
            cover, density = None, 0.0
        share = cloud_share(sun)
        lamp_from = "" if not a.lamp_from else "porch" if share == 0 else "cloud base" if share == 1 else "both"
        moment = next((k.name for k, at in zip(KEYS, key_frames, strict=True) if at == n), "")
        frames.append(kf.Key(f"{n:04d}", moment, f"{n:04d}", sun, lamp_from, cover, density, sun < LANTERN_BELOW,
                             lerp(a.exposure, b.exposure, t)))
    return frames


def drift(frame):
    return CLOUD_DRIFT * (int(frame.name) - 1)


def settings_text(frame):
    """The computed settings as the log shows them. A master on disk must match its frame's text."""
    clouds = (f"clouds {frame.cloud_cover:.3f} / {frame.cloud_density:.5f}, drift {drift(frame):.0f} m"
              if frame.cloud_cover is not None else "no clouds")
    return (f"sun {frame.sun:+.3f} | exposure {frame.exposure:+.3f} EV | {clouds} | "
            f"lantern {'on' if frame.porch_light else 'off'}")


def lamp_where(frame):
    """Where this frame's Sun lamp is measured, as the log shows it."""
    if frame.lamp_from != "both":
        return frame.lamp_from
    share = cloud_share(frame.sun)
    return f"porch {1 - share:.0%} + cloud base {share:.0%}"


# ---------------------------------------------------------------------------------------------
# Rendering.

def measure_lamp(frame, tmp):
    """This frame's Sun lamp (strength, colour), with 08's measurement. Each measurement resets Blender, so this runs
    before the scene is built."""
    if not frame.lamp_from:
        return 0.0, (0.0, 0.0, 0.0)
    share = cloud_share(frame.sun)
    light = np.zeros(3)
    for where, weight in (("porch", 1.0 - share), ("cloud base", share)):
        if weight > 0:
            strength, color = kf.measure_lamp(dataclasses.replace(frame, lamp_from=where), tmp)
            light += weight * strength * np.array(color)
    strength = float(light.max())
    if strength < kf.LAMP_MIN:
        return 0.0, (0.0, 0.0, 0.0)
    return strength, tuple(float(c) for c in light / strength)


def apply_frame(scene, frame, lamp):
    """08's apply_key, with the clouds drifted. 05's cloud builder lays its pattern out around CAM_LOC, reading it when
    called, so building the clouds around a point moved away from the sun shifts the whole layer toward the camera.
    The camera itself was placed when the scene was built, and nothing else apply_key builds reads CAM_LOC."""
    keep = porch.CAM_LOC
    yaw = math.radians(porch.SUN_YAW)
    porch.CAM_LOC = (keep[0] - drift(frame) * math.sin(yaw), keep[1] - drift(frame) * math.cos(yaw), keep[2])
    try:
        kf.apply_key(scene, frame, lamp)
    finally:
        porch.CAM_LOC = keep


def frame_line(frame, lamp, seconds, peak_mb):
    strength, color = lamp
    if not frame.lamp_from:
        light = "no Sun lamp"
    elif not strength:
        light = "Sun lamp: no sunlight left"
    else:
        light = (f"Sun lamp {strength:.2f} x ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}) from the "
                 f"{lamp_where(frame)}")
    return (f"{frame.name} {frame.moment:2} | {settings_text(frame)} | {light} | render {seconds:.1f} s | "
            f"Cycles {peak_mb:,.0f} MB | RAM peak {porch.peak_ram_gb():.1f} GB | {time.strftime('%Y-%m-%d %H:%M')}")


def read_log(path):
    """{frame number: line} from the log an earlier run left."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    return {int(line[:4]): line for line in text.splitlines() if re.match(r"\d{4} .. \| sun ", line)}


def write_log(path, lines, preview, summary=()):
    r = "1280x720, no trees" if preview else "2560x1440, trees on"
    samples = porch.PREVIEW_SAMPLES if preview else porch.FINAL_SAMPLES
    text = (f"Time test frames (handoff 09), {r}, {samples} samples: one line per frame, rewritten after every frame.\n"
            + "".join(lines[n] + "\n" for n in sorted(lines)))
    if summary:
        text += "\n" + "".join(s + "\n" for s in summary)
    path.write_text(text, encoding="utf-8")


def check_masters(folder, frames, lines):
    """Stops if a master on disk was rendered with other settings than the frame table now gives (sun, exposure,
    clouds, lantern, and where the Sun lamp is measured), so a run never mixes two versions of the table."""
    table = {int(f.name): f for f in frames}
    for path in sorted(folder.glob("[0-9][0-9][0-9][0-9].png")):
        n = int(path.stem)
        line = lines.get(n, "")
        frame = table.get(n)
        lamp_ok = frame is not None and (
            "no Sun lamp" in line if not frame.lamp_from
            else f"from the {lamp_where(frame)} |" in line or "no sunlight left" in line)
        if not lamp_ok or settings_text(frame) not in line:
            raise SystemExit(f"{path} doesn't match the frame table (its log line: {line or 'missing'!r}). Delete the "
                             "masters that differ, or the whole folder, to render them again.")


def render_frames(frames, todo, folder, log_path, lines, args, started):
    samples = porch.PREVIEW_SAMPLES if args.preview else porch.FINAL_SAMPLES
    with tempfile.TemporaryDirectory() as tmp:
        lamps = {f.name: measure_lamp(f, Path(tmp)) for f in todo}
    print(f"Sun lamp measured for {len(todo)} frames in {time.time() - started:.0f} s", flush=True)
    scene, devices = kf.build_scene(not args.preview, args.preview, samples, lambda msg: print(msg, flush=True))
    r = scene.render
    print(f"Scene built in {time.time() - started:.0f} s (peak RAM so far {porch.peak_ram_gb():.1f} GB)", flush=True)
    print(f"Render: {r.resolution_x * r.resolution_percentage // 100}x{r.resolution_y * r.resolution_percentage // 100}"
          f", {samples} samples, denoiser {scene.cycles.denoiser}, on {devices}; trees "
          f"{'off' if args.preview else 'on'}", flush=True)
    for i, frame in enumerate(todo):
        if args.minutes is not None and time.time() - started > args.minutes * 60:
            print(f"Stopped after {args.minutes:g} minutes with {len(todo) - i} frames left: run again to continue.",
                  flush=True)
            return
        lamp = lamps[frame.name]
        apply_frame(scene, frame, lamp)
        seconds, peak_mb = kf.render(scene)
        path = folder / f"{frame.name}.png"
        part = folder / f"{frame.name}.part.png"  # renamed once complete: a cut-off run never leaves a broken master
        kf.save_render(scene, part, "PNG")
        lines[int(frame.name)] = frame_line(frame, lamp, seconds, peak_mb)
        write_log(log_path, lines, args.preview)
        part.replace(path)
        print(lines[int(frame.name)], flush=True)


# ---------------------------------------------------------------------------------------------
# Web frames and the star check.

def image_from(px, size):
    """A new in-memory image from pixels (rows from the bottom up), scaled to size by Blender. Blender ignores the WebP
    quality when it saves an image it loaded from a PNG (it writes it losslessly), but honours it for one like this."""
    h, w = px.shape[:2]
    im = bpy.data.images.new("frame", w, h)
    im.pixels.foreach_set(px.ravel())
    if (w, h) != size:
        im.scale(*size)
    return im


def encode_web(frames):
    """Both web sets, rebuilt from the masters. Returns the summary for the log."""
    for label in WEB_SETS:
        (WEB / label).mkdir(parents=True, exist_ok=True)
        for old in (WEB / label).glob("[0-9][0-9][0-9][0-9].webp"):
            old.unlink()
    sizes = {label: [] for label in WEB_SETS}
    darkest = None
    for frame in frames:
        px = kf.load(MASTERS / f"{frame.name}.png")
        quality = NIGHT_QUALITY if frame.sun <= KEYS[2].sun else DAY_QUALITY
        for label, size in WEB_SETS.items():
            path = WEB / label / f"{frame.name}.webp"
            im = image_from(px, size)
            im.file_format = "WEBP"
            im.save(filepath=str(path), quality=quality)
            bpy.data.images.remove(im)
            sizes[label].append(path.stat().st_size)
        brightness = float(px[..., :3].mean())
        if darkest is None or brightness < darkest[0]:
            darkest = (brightness, frame)
    night_from = next(f.name for f in frames if f.sun <= KEYS[2].sun)
    summary = [(f"Web frames, encoded {time.strftime('%Y-%m-%d %H:%M')}: WebP quality {DAY_QUALITY}, and "
                f"{NIGHT_QUALITY} from frame {night_from} (k3) on")]
    for label, s in sizes.items():
        summary.append(f"{label} set: {len(s)} frames, {sum(s) / 2**20:.1f} MB in all, "
                       f"{min(s) / 1024:.0f}-{max(s) / 1024:.0f} KB per frame")
    return summary + star_check(darkest[1])


def star_check(frame):
    """The darkest frame's faint stars at 100%, before and after compression, in each set: 08's detector and crop
    (scaled for 1080). Writes the crops side by side to renders/review/09/checks/."""
    master = kf.load(MASTERS / f"{frame.name}.png")
    checks = REVIEW / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    lines = []
    for label, size in WEB_SETS.items():
        x, y, w, h = (round(v * size[0] / 2560) for v in kf.STAR_CROP)
        im = image_from(master, size)  # what the encoder was given
        before = np.empty(size[0] * size[1] * 4, np.float32)
        im.pixels.foreach_get(before)
        bpy.data.images.remove(im)
        after = kf.load(WEB / label / f"{frame.name}.webp")
        crops = [img[img.shape[0] - y - h:img.shape[0] - y, x:x + w]
                 for img in (before.reshape(size[1], size[0], 4), after)]
        stars, height = kf.star_peaks(crops[0][..., :3] @ LUMA, 3 / 255)
        found, height_after = kf.star_peaks(crops[1][..., :3] @ LUMA, 3 / 255)
        result = (f"Star check, {label} set, darkest frame {frame.name} ({w}x{h} px crop at x={x}, y={y}): "
                  f"{stars.sum()} stars before compression, {found.sum()} in the WebP")
        if stars.any():
            # Kept: a star the WebP still shows within 1 px of where it was (new peaks from the compression don't count).
            padded = np.pad(found, 1)
            near = np.zeros_like(found)
            for dy in range(3):
                for dx in range(3):
                    near |= padded[dy:dy + found.shape[0], dx:dx + found.shape[1]]
            kept = height_after[stars] / height[stars]
            result += (f"; {(stars & near).sum() / stars.sum():.0%} of the stars are kept (still found within 1 px); "
                       f"at the same pixels the WebP keeps a median {np.median(kept):.0%} of each star's height above "
                       f"the sky")
        lines.append(result)
        gap = 16
        pair = np.full((h, 2 * w + gap, 4), 0.06, np.float32)
        pair[:, :w], pair[:, w + gap:] = crops
        caption = (f"frame {frame.name} stars at 100%, {label} set:  left, before compression  |  right, the WebP "
                   f"(quality {NIGHT_QUALITY})")
        kf.write_png(kf.labelled(pair, caption), checks / f"stars-{label}.png")
    return lines


# ---------------------------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(prog="09_time_frames.py")
    p.add_argument("--minutes", type=float, help="start no new frame after this many minutes (default: render all)")
    p.add_argument("--preview", action="store_true", help="1280x720 without trees, into renders/review/09/preview/")
    return p.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    started = time.time()
    folder = REVIEW / "preview" if args.preview else MASTERS
    log_path = folder / "09-preview-log.txt" if args.preview else LOG_PATH
    folder.mkdir(parents=True, exist_ok=True)
    frames = frame_settings()
    lines = read_log(log_path)
    check_masters(folder, frames, lines)
    todo = [f for f in frames if not (folder / f"{f.name}.png").exists()]
    keyframes = ", ".join(f"{f.moment} {f.name}" for f in frames if f.moment)
    print(f"{len(frames)} frames (keyframes: {keyframes}); {len(frames) - len(todo)} already rendered, {len(todo)} to go",
          flush=True)
    if todo:
        render_frames(frames, todo, folder, log_path, lines, args, started)
    if args.preview or not all((MASTERS / f"{f.name}.png").exists() for f in frames):
        return
    summary = encode_web(frames)
    write_log(LOG_PATH, lines, False, summary)
    for line in summary:
        print(line, flush=True)
    print(f"Run time: {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
