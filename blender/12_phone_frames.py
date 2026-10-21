r"""Handoff 12: the upright phone frames, cut from 11's masters (no re-render).

Nems's crop for phones (B, Oct 20) is master x 1560-2288 at full height: the sun beside post 2 at golden hour, the
lantern, and the table with the two mugs. The phone frames are 9:16 around it, master x 1519-2329. A phone whose browser
shows about 9:17.8 upright (Nems's iPhone in Safari) sees x 1560-2288; wider phones see a little more.

Run headless from the repo root (Blender isn't on PATH, so call it by full path):
    & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender\12_phone_frames.py

It reads renders/time/master-11/0001.png ... 0069.png (2560x1440, never written) and writes, in about two minutes:
- public/frames/time/phone1440/ (810x1440: the masters' own pixels) and phone1080/ (608x1080, scaled as the 1080 set
  is), WebP with 11's quality rule: 95, and 98 from k3 (frame 32) on;
- renders/review/12/checks/: the star check on the darkest phone frame, before and after compression;
- renders/review/12/phone-frames.webp: phone frames 1, 40 and 69 side by side, labelled;
- renders/review/12/12-phone-frames-log.txt: the sets' sizes and the star check.
The desktop sets (public/frames/time/1440/ and 1080/) and renders/review/05/ to 11/ aren't touched.
"""

import importlib.util
import struct
import sys
import time
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "renders" / "review" / "12"
LOG_PATH = REVIEW / "12-phone-frames-log.txt"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.dont_write_bytecode = True  # no __pycache__ next to the scripts
# 11: the frame table, the masters and the WebP quality rule. It loads 09, 10, 08 and 05 as modules: no main() runs,
# and 08 switches off 05's downloader, so nothing here touches the network.
frames11 = _load("night_frames", "11_night_frames.py")
frames09, kf, porch = frames11.frames09, frames11.kf, frames11.porch
kf.OUT = porch.OUT = REVIEW  # where 08's and 05's helpers would write by default: never 11's folder from here
MASTERS, WEB = frames11.MASTERS, frames09.WEB
MASTER_SIZE = (2560, 1440)

# The cut: master x 1519-2329 at full height. 9:16, centred on crop B (x 1560-2288, centre 1924).
CUT = (1519, 2329)
# The phone sets, name: (width, height). 608 is 810 x 3/4 = 607.5 rounded up, so phone1080 is stretched sideways by
# 0.08%, under a pixel across the frame.
PHONE_SETS = {"phone1440": (810, 1440), "phone1080": (608, 1080)}
# Star check: a 100% crop of the sky left of post 2, under the porch beam and above the ridge (x, y from the top-left of
# a phone1440 frame, width, height; scaled for phone1080). The sky right of the post is mostly pine branches, which the
# star detector would count as stars.
STAR_CROP = (8, 168, 220, 264)
SHEET_FRAMES = ((1, "golden hour"), (40, "the sconces come on"), (69, "night"))


def png_size(path):
    """A PNG's width and height, read from its header."""
    with open(path, "rb") as f:
        return struct.unpack(">2I", f.read(24)[16:24])


def cut(name):
    """A master's phone cut (rows from the bottom up, as Blender stores them)."""
    return np.ascontiguousarray(kf.load(MASTERS / f"{name}.png")[:, CUT[0]:CUT[1]])


def encode(frames):
    """Both phone sets, rebuilt from the masters. Returns the darkest frame (the cut's mean) and the summary for the
    log."""
    for label in PHONE_SETS:
        (WEB / label).mkdir(parents=True, exist_ok=True)
        for old in (WEB / label).glob("[0-9][0-9][0-9][0-9].webp"):
            old.unlink()
    sizes = {label: [] for label in PHONE_SETS}
    darkest = None
    for frame in frames:
        px = cut(frame.name)
        quality = frames11.quality(frame)
        for label, size in PHONE_SETS.items():
            path = WEB / label / f"{frame.name}.webp"
            im = frames09.image_from(px, size)  # a new image, so Blender honours the quality (see image_from)
            im.file_format = "WEBP"
            im.save(filepath=str(path), quality=quality)
            bpy.data.images.remove(im)
            sizes[label].append(path.stat().st_size)
        print(f"{frame.name}: quality {quality}, "
              + ", ".join(f"{label} {s[-1] / 1024:.0f} KB" for label, s in sizes.items()), flush=True)
        brightness = float(px[..., :3].mean())
        if darkest is None or brightness < darkest[0]:
            darkest = (brightness, frame)
    night_from = next(f.name for f in frames if frames11.quality(f) == frames09.NIGHT_QUALITY)
    summary = [(f"Phone frames, encoded {time.strftime('%Y-%m-%d %H:%M')} from master x {CUT[0]}-{CUT[1]} at full "
                f"height: WebP quality {frames09.DAY_QUALITY}, and {frames09.NIGHT_QUALITY} from frame {night_from} on")]
    for label, s in sizes.items():
        summary.append(f"{label} set ({'x'.join(map(str, PHONE_SETS[label]))}): {len(s)} frames, "
                       f"{sum(s) / 2**20:.1f} MB in all, {min(s) / 1024:.0f}-{max(s) / 1024:.0f} KB per frame")
    summary.append(f"Darkest phone frame: {darkest[1].name} (mean {darkest[0] * 255:.1f} of 255)")
    return darkest[1], summary


def star_check(frame):
    """The darkest phone frame's faint stars at 100%, before and after compression, in each phone set: 08's detector
    and 09's rule for a kept star, in STAR_CROP. Writes the crops side by side to renders/review/12/checks/."""
    px = cut(frame.name)
    checks = REVIEW / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    lines = []
    for label, size in PHONE_SETS.items():
        x, y, w, h = (round(v * size[1] / 1440) for v in STAR_CROP)
        im = frames09.image_from(px, size)  # what the encoder was given
        before = np.empty(size[0] * size[1] * 4, np.float32)
        im.pixels.foreach_get(before)
        bpy.data.images.remove(im)
        after = kf.load(WEB / label / f"{frame.name}.webp")
        crops = [img[img.shape[0] - y - h:img.shape[0] - y, x:x + w]
                 for img in (before.reshape(size[1], size[0], 4), after)]
        stars, height = kf.star_peaks(crops[0][..., :3] @ frames09.LUMA, 3 / 255)
        found, height_after = kf.star_peaks(crops[1][..., :3] @ frames09.LUMA, 3 / 255)
        result = (f"Star check, {label} set, darkest frame {frame.name} ({w}x{h} px crop at x={x}, y={y} of the phone "
                  f"frame): {stars.sum()} stars before compression, {found.sum()} in the WebP")
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
        caption = f"before  |  WebP {frames11.quality(frame)}"  # short: the pair is only 346-456 px wide
        kf.write_png(kf.labelled(pair, caption), checks / f"stars-{label}-{frame.name}.png")
    return lines


def review_sheet():
    """Phone frames 1, 40 and 69 as the phone1440 set has them, side by side and labelled, to check the cut against
    crop B (renders/review/12/phone-crops.png)."""
    gap = 24
    tiles = [kf.labelled(kf.load(WEB / "phone1440" / f"{n:04d}.webp"), f"phone frame {n}, {moment}")
             for n, moment in SHEET_FRAMES]
    th, tw = tiles[0].shape[:2]
    sheet = np.full((th + 2 * gap, len(tiles) * (tw + gap) + gap, 4), 0.06, np.float32)
    for i, tile in enumerate(tiles):
        x = gap + i * (tw + gap)
        sheet[gap:gap + th, x:x + tw] = tile
    path = REVIEW / "phone-frames.webp"
    porch.write_pixels(sheet, path, quality=92)
    return path


def main():
    started = time.time()
    REVIEW.mkdir(parents=True, exist_ok=True)
    frames = frames11.frame_table()
    # Every master is checked before the phone sets are cleared, so a bad one never leaves a set half written.
    paths = [MASTERS / f"{f.name}.png" for f in frames]
    bad = [p.name for p in paths if not p.exists() or png_size(p) != MASTER_SIZE]
    if bad:
        raise SystemExit(f"Masters missing from {MASTERS} or not {MASTER_SIZE[0]}x{MASTER_SIZE[1]}: {', '.join(bad)}. "
                         "11 renders them. Nothing was written.")
    darkest, summary = encode(frames)
    summary += star_check(darkest)
    summary.append(f"Review sheet: {review_sheet().relative_to(ROOT)}")
    LOG_PATH.write_text("".join(line + "\n" for line in summary), encoding="utf-8")
    for line in summary:
        print(line, flush=True)
    print(f"Run time: {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
