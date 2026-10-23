r"""Handoffs 13 and 15: section 2's look test. The car pulled over at a pull-out under the aurora, as full-size stills.

Section 2 (option C) opens behind the parked car on its two red tail-lights, then circles its right side to the front. This
renders two of its shots for Nems to judge the look before any frames are made: C1, behind the car (the opening shot),
and C3, its right side. Round 2 (handoff 15) renders each under three sky options (--sky), so section 2's sky reads as
the porch's:
- lens: longer lenses, the cameras moved back so the car keeps its size and place. C1 then sees the porch's own slice of
  sky (the same direction, up to the same height), mostly green; the aurora is unchanged.
- fringe: 13's cameras, the aurora's magenta-red top at a third of its strength (10's top_share 0.6 -> 0.2), here only.
- both: the two together.

The night is section 1's last frame (69), from 11's frame table: 05's sky with the sun far below the horizon, 08's stars
and night glow (at half, as 11 has it), 10's vivid aurora with 11's rays at the frame's phase, the frame's exposure and
05's grade. 11, 10, 09, 08 and 05 are loaded as modules and left unchanged; 08 switches off 05's downloader, so nothing
here touches the network. The car is blender/lib/car.py's, Guards Red, headlights and tail-lights on. The only light is
the sky, the aurora and the car's own lamps.

The set is built around the car at the porch's own map position, 200 m lower, so the valley, its mist, the ridge across
it and the far ridges sit where section 1 shows them:
- the ground in the car's frame: a grid, fine near the car and coarse out to 9 km, for the road behind the car, a gravel
  pull-out out to the edge of the drop, the hillside rising behind the road (the road bends uphill 200 m along) and the
  drop; past 300 m beyond the edge, the valley floor and the ridge across it are rings around the car, so their outlines
  have no straight facets. Near the car the ground is a separate object, whose gravel and asphalt are truly displaced;
- a low wooden guardrail 5.5 m ahead of the nose, with the edge of the drop 0.6 m past it;
- handoff 14's light pines and firs on the hillside;
- 05's far ridges where section 1 sees them and again where C3 looks, in narrow wedges (no facets), the nearest one
  (8 km) lowered so the three layer and fade with distance from 200 m below the porch; 05's mist and haze;
- a thin, local haze in wisps over the drop, low and short, so the headlight beams show without a glowing sheet.
The scripts loaded here (05, 08-11, car.py) are unchanged: the ridge, aurora and lamp changes are made only in this run.

Run headless from the repo root (Blender isn't on PATH, so call it by full path), one still per run:
    & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender\13_s2_look.py -- --still C1 --sky lens *> renders\review\15\15-console-C1-lens.log
    ... -P blender\13_s2_look.py -- --still C3 --sky fringe *> renders\review\15\15-console-C3-fringe.log
    ... -P blender\13_s2_look.py -- --still C1 --sky both --preview NAME [--samples N] [--no-trees]   # 1280x720 tests

A full run writes, into renders/review/15/: 15-<still>-<sky>.png (8-bit, like 11's masters), 15-<still>-<sky>.webp
(encoded from the PNG as 11's night frames are, quality 98) and its block in 15-summary.txt (the camera, render time,
memory, and for C1 the tail-lights' pixels). Previews go to renders/review/15/preview/. Nothing else is written; 13's
stills in renders/review/13/ are never touched.
"""

import argparse
import ctypes
import dataclasses
import importlib.util
import itertools
import math
import re
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "renders" / "review" / "15"
PREVIEW = REVIEW / "preview"
SUMMARY = REVIEW / "15-summary.txt"
TEXTURES = ROOT / "assets-source" / "textures"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.dont_write_bytecode = True  # no __pycache__ next to the scripts
HERE = Path(__file__).resolve().parent
# 11: the frame table and the aurora's look and rays. It loads 09, 10, 08 and 05 as modules (no main() runs) and halves
# 08's night glow as section 1 has it; 08 switches off 05's downloader, so nothing here touches the network.
frames11 = _load("night_frames", HERE / "11_night_frames.py")
frames09, kf, porch, night10 = frames11.frames09, frames11.kf, frames11.porch, frames11.night10
kf.OUT = porch.OUT = REVIEW  # where 08's and 05's helpers would write by default: never their own folders from here
car_lib = _load("car", HERE / "lib" / "car.py")
FRAME = 69  # section 1's last frame: its sky, aurora, exposure and grade

smoothstep, fbm = porch.smoothstep, porch.fbm_field


# ---------------------------------------------------------------------------------------------
# The set. Positions are in the car's frame: u forward (where the nose points), w toward the car's left, z up from the
# pull-out's surface; the origin is the centre of the car's footprint. Lengths in metres.

SET_AT = (0.0, 0.0, -200.0)  # that origin in the world: the porch's map position, 200 m below its deck
HEADING = porch.CAM_YAW  # degrees from +Y toward +X: the car faces the porch's view, so the aurora is over the valley
TURN = math.radians(90.0 - HEADING)  # the frame's turn about z

ROAD = (-10.5, -3.5)  # u of the asphalt's uphill and valley edges: a 7 m road behind the car
CROWN = 0.07  # the road's middle stands this much above its edges
APRON_FLAT, APRON_FALL = 3.0, 0.008  # the pull-out is level up to this u (the car stands level), then falls 0.8%
RAIL_U = 7.7  # the guardrail's line (the beam's valley face): 5.55 m ahead of the nose
LIP_U = RAIL_U + 0.6  # the edge of the drop
UNDULATION = 0.025  # gentle unevenness of the pull-out, none under the car
DITCH_DEPTH = 0.25  # a shallow ditch along the road's uphill edge
HILL_TOP, HILL_SLOPE = 260.0, 0.55  # the hillside rises toward this height above the road, starting at this slope (29 deg)
# m: the hillside's large humps (150 m across, from 15 m up the slope, so the bank by the road stays a plain slope), its
# rough ground (6 m across) and small bumps (1.5 m)
HILL_LUMPS = (8.0, 0.6, 0.15)
# The bank between the road and the trees: two finer layers of lumps (height m, across m) from the ditch up, on a fine
# grid up to U_FINE's uphill end (20 m up the slope), and the forest floor's own height map as a bump.
BANK_LUMPS = ((0.12, 1.5), (0.05, 0.5))
BANK_BUMP = 0.03  # m: the height map's relief
DROP_SCALE = 60.0  # the drop's e-folding length past the lip: 72 degrees at the lip, easing to the valley floor
# 200 m along the road (toward C3's view), the road and the hillside start to bend uphill (toward -u), turning on this
# radius until they run 30 degrees off their first line, so the road and the rail don't run straight to the horizon.
BEND = (200.0, 400.0, 30.0)

# The ground grid: every U_STEP / W_STEP metres across the fine part, cells growing by GRID_GROWTH each beyond it.
GROUND_FAR = 9000.0
U_FINE, U_STEP = (-31.0, 11.0), 0.1  # across the shelf: the road's edges, the rail and the lip are grid lines
W_FINE, W_STEP = (-20.0, 56.0), 0.25
GRID_GROWTH = 1.05
# Past the edge, the grid's cells grow to ~300 m and show as straight facets on the valley's outlines. From VALLEY_FROM m
# past the lip the valley is rings around the car instead, VALLEY_STEP degrees apart (about 4 px at 55 mm) and each
# VALLEY_GROWTH times the last outward, so an outline seen from the car is sampled every few pixels. The grid stops
# GRID_TO m past the lip; between the two they overlap, about 180 m below the lip, out of sight from the pull-out.
VALLEY_FROM, GRID_TO = 300.0, 700.0
VALLEY_STEP, VALLEY_GROWTH = 0.055, 1.007
# The part near the car, subdivided at render time for true displacement: its edges are grid lines, and the displacement
# fades out before them, so it meets the rest of the ground without a crack.
PULLOUT_U, PULLOUT_W = (-12.0, 9.3), (-16.0, 50.0)
# Micro-polygons 2 px across: against 1 px, C1's gravel differs by about 1.5 of 255 levels on average (render noise) and
# the run needs about 2 GB less RAM.
DICING_PX = 2.0
# Out of view (beside and behind the camera, seen only in reflections and shadows) the micro-polygons are this many times
# larger. Blender's default, 4, spent about 2 GB of RAM on the ground right under and beside the camera.
OFFSCREEN_DICING = 16.0
GRAVEL_TILE, ASPHALT_TILE, GROUND_TILE = 2.0, 3.0, 2.5  # metres one texture covers
GRAVEL_RELIEF, ASPHALT_RELIEF = 0.04, 0.008  # displacement, highest to lowest point
EDGE_JITTER = 0.15  # the edges between asphalt, gravel and ground wander by up to this

# The guardrail: square posts every 2 m, a beam on their road side in 4 m lengths. 05's worn-wood material.
RAIL_W = (-30.0, 150.0)
POST_STEP = 2.0
POST = (0.14, 0.75, 0.15)  # width, height above the ground, depth in the ground
BEAM = (0.12, 0.20, 0.65)  # depth, height, top above the ground

# Pines and firs on the hillside, uphill from the road's edge (m): none below the first, full from the second, thinning
# out between the last two. 05's density, sizes and tilt for its pines by the porch.
TREES_S = (3.0, 10.0, 320.0, 420.0)

# Beam haze: thin mist low over the drop, where the beams are bright. It fades in from the nose to the lip, and out
# toward the far end, the sides, the top and the bottom (each: full density up to the first value, none past the
# second). Round 2: short and low (13's box reached 35 m out and 8 m up, so from behind it glowed as one sheet up past
# the skyline), and broken into wisps with clear air between them.
HAZE_BOX = ((2.0, -14.0, -8.0), (25.0, 14.0, 3.0))  # (u, w, z) corners
HAZE_FADE = {"far": (12.5, 25.0), "side": (7.0, 14.0), "top": (1.05, 3.0), "bottom": (-4.0, -8.0)}
HAZE_DENSITY = 0.03  # per metre where it's densest (Nems, Oct 21; 13 had 0.06)
HAZE_WISPS = 3.0  # m across, flattened 2.5 times in height
HAZE_CLEAR = 0.4  # share of the box with no haze
HAZE_ANISOTROPY = 0.1  # nearly even: the beams show from behind (C1) and from the side (C3)

# 05's far ridges: the nearest (8 km) crests at RIDGE1_CREST m above the porch's deck instead of 05's 250 m. From 200 m
# below the porch, 05's three ridges all crest about 3.2 degrees up, so the nearest hid the other two. Each span is
# built in RIDGE_SPLIT wedges: 05's 700 steps across a third of 100 degrees come to ~0.05 degrees, no visible facets.
RIDGE1_CREST = 60.0
RIDGE_SPANS = ((0.0, 100.0), (262.0, 360.0))  # yaw: where section 1 (and C1) looks, and where C3 looks
RIDGE_SPLIT = 3

# The aurora's magenta-red top in the "fringe" and "both" options: 10's top_share for the vivid look is 0.6.
FRINGE_TOP_SHARE = 0.2

CAR_SINK = 0.005  # the tyres settle this far into the gravel
# car.py's tail-light lamps are point lights inside the lamp housings, so they light the car's own engine lid and rear
# panel bright pink from a few centimetres away. Here they're wide spots aimed out of the lenses: they light the ground
# behind the car and the bumper, not the bodywork around them. At half car.py's power (Nems, Oct 21): only the light
# they cast; the lit lens panels keep their brightness. car.py itself is unchanged.
TAIL_CONE, TAIL_DOWN = 160.0, 10.0  # degrees across, and below level
TAIL_POWER = 0.5


@dataclasses.dataclass(frozen=True)
class Shot:
    name: str
    what: str
    at: tuple  # (u, w)
    height: float  # above the ground under the camera
    look: float  # degrees from straight ahead (+u) toward the car's left (+w)
    tilt: float  # degrees up
    lens: float  # mm on a 36 mm sensor
    focus: float  # m


# 13's cameras: the task's numbers, except C1's height (at 0.95 m the tail-lights sit 73% down the frame, at 0.70 m 66%).
# They see the sky up to about 20 degrees, against about 15 from the porch, and the aurora's magenta rises fast above 13.
SHOTS = {s.name: s for s in (
    Shot("C1", "behind the car", (-8.5, 0.0), 0.70, 0.0, 4.0, 34.0, 6.42),  # focused on the tail-lights
    Shot("C3", "its right side", (0.0, -8.0), 1.00, 90.0, 5.0, 40.0, 7.11),  # focused on the car's side
)}
# Longer lenses (sky options "lens" and "both"), moved back so the car keeps its size and place. C1's top edge sits at
# 14.5 degrees up, like the porch's, and its tail-lights are 13's: as far apart and 67% down the frame (solved by
# projection). Its camera stands in the ditch by the road, 0.76 m above it: about 0.6 m above the road.
LONG_SHOTS = {s.name: s for s in (
    Shot("C1", "behind the car", (-11.875, 0.0), 0.755, 0.0, 3.48, 52.0, 9.81),  # focused on the tail-lights
    Shot("C3", "its right side", (0.0, -10.67), 1.00, 90.0, 3.64, 55.0, 9.78),
)}
SKIES = {  # label: (the cameras, the aurora fringe's top_share or None for 10's)
    "lens": (LONG_SHOTS, None),
    "fringe": (SHOTS, FRINGE_TOP_SHARE),
    "both": (LONG_SHOTS, FRINGE_TOP_SHARE),
}


def to_world(u, w, z):
    return Vector(SET_AT) + Matrix.Rotation(TURN, 3, "Z") @ Vector((u, w, z))


def world_xy(U, W):
    c, s = math.cos(TURN), math.sin(TURN)
    return SET_AT[0] + U * c - W * s, SET_AT[1] + U * s + W * c


# ---------------------------------------------------------------------------------------------
# The ground's shape.

def bend(W):
    """How far the shelf's line has moved uphill at w, and the slope of its turn."""
    t = np.maximum(W - BEND[0], 0.0)
    radius, turn = BEND[1], math.tan(math.radians(BEND[2]))
    t1 = radius * turn
    off = np.where(t < t1, t * t / (2 * radius), radius * turn * turn / 2 + turn * (t - t1))
    return off, np.minimum(t / radius, turn)


def lateral(U, W):
    """Distance across the shelf, positive toward the valley: u up to the bend, then measured from the bending line."""
    off, slope = bend(W)
    return (U + off) / np.sqrt(1.0 + slope * slope)


def valley_floor(X, Y, t):
    """The valley floor under the mist, from the pull-out's height: 05's numbers at the same world positions, so the ridge
    across the valley (about 6 km out) and the bumps are section 1's. t: distance past the lip (05: past the rail)."""
    z = np.full(X.shape, porch.FLOOR_Z - SET_AT[2])
    ridge_y = 4300.0 + 500.0 * fbm(X, X * 0, 5000.0, 2.1, 3)
    crest = 470.0 + 170.0 * fbm(X, X * 0, 2600.0, 3.3, 4)
    bump = np.exp(-(((Y - ridge_y) / 1300.0) ** 2)) * smoothstep(1500.0, 3000.0, Y)
    z += bump * crest * (1.0 + 0.12 * fbm(X, Y, 800.0, 9.9, 5, ridged=True))
    z += 55.0 * fbm(X, Y, 700.0, 1.7) * smoothstep(*porch.BUMPS_FROM, t)
    z += 8.0 * fbm(X, Y, 110.0, 5.1, 4) * smoothstep(60.0, 300.0, t)
    return z


def ground_height(U, W):
    """Height of the ground (from the pull-out's surface by the car) on arrays of u and w, and the distance across the
    shelf. Across the shelf: the hillside, a ditch, the crowned road, the pull-out, the lip, the drop to the valley."""
    d = lateral(U, W)
    X, Y = world_xy(U, W)
    edge = -CROWN
    z = np.full(U.shape, edge)
    road = (d >= ROAD[0]) & (d <= ROAD[1])
    z[road] = -CROWN * np.abs(d[road] - sum(ROAD) / 2) / ((ROAD[1] - ROAD[0]) / 2)
    z_lip = edge - APRON_FALL * (LIP_U - APRON_FLAT)
    apron = (d > ROAD[1]) & (d <= LIP_U)
    z[apron] = edge - APRON_FALL * np.maximum(d[apron] - APRON_FLAT, 0.0)

    drop = d > LIP_U
    t = d[drop] - LIP_U
    xd, yd = X[drop], Y[drop]
    rough = 1.5 * fbm(xd, yd, 6.0, 21.3, 4) * smoothstep(0.3, 6.0, t)
    z[drop] = z_lip + (valley_floor(xd, yd, t) - z_lip) * (1.0 - np.exp(-t / DROP_SCALE)) + rough

    hill = d < ROAD[0]
    s = ROAD[0] - d[hill]
    xh, yh = X[hill], Y[hill]
    ditch = -DITCH_DEPTH * np.sin(np.pi * np.clip(s - 0.6, 0.0, 1.0))
    rise = HILL_TOP * (1.0 - np.exp(-HILL_SLOPE * np.maximum(s - 1.6, 0.0) / HILL_TOP))
    lumps = (HILL_LUMPS[0] * fbm(xh, yh, 150.0, 22.7, 4) * smoothstep(15.0, 120.0, s)
             + HILL_LUMPS[1] * fbm(xh, yh, 6.0, 23.9, 4) * smoothstep(2.0, 8.0, s)
             + HILL_LUMPS[2] * fbm(xh, yh, 1.5, 24.1, 3) * smoothstep(1.6, 4.0, s))
    bank = s < ROAD[0] - U_FINE[0]  # on the fine grid, fading out before its end
    sb = s[bank]
    lumps[bank] += (sum(a * fbm(xh[bank], yh[bank], size, 26.3 + i, 3) for i, (a, size) in enumerate(BANK_LUMPS))
                    * smoothstep(1.6, 2.6, sb) * (1.0 - smoothstep(16.0, 20.0, sb)))
    z[hill] = edge + ditch + rise + lumps

    # The pull-out's gentle unevenness (carried on over the lip so the edge stays continuous), none under the car.
    side = d > ROAD[1]
    car = (1.0 - smoothstep(2.4, 3.4, np.abs(U))) * (1.0 - smoothstep(1.2, 2.2, np.abs(W)))
    z[side] += (UNDULATION * fbm(X[side], Y[side], 2.5, 25.3, 3) * smoothstep(ROAD[1], ROAD[1] + 1.0, d[side])
                * (1.0 - car[side]))
    return z - (U * U + W * W) / (2 * porch.EARTH_R), d


def height_at(u, w):
    return float(ground_height(np.array([[u]]), np.array([[w]]))[0][0, 0])


def tree_density(d, X, Y, W):
    """0-1 per point: trees on the hillside only, in 05's patches and clearings."""
    s = ROAD[0] - d
    dens = np.zeros(d.shape)
    on = s > TREES_S[0]
    clumps = smoothstep(-1.3, -0.4, fbm(X[on], Y[on], 170.0, 11.1, 4))
    dens[on] = (smoothstep(TREES_S[0], TREES_S[1], s[on]) * (1.0 - smoothstep(TREES_S[2], TREES_S[3], s[on])) * clumps
                * smoothstep(-90.0, -50.0, W[on]) * (1.0 - smoothstep(900.0, 1100.0, W[on])))
    return dens


def axis(fine, step):
    """Grid lines: every `step` across `fine`, then cells growing by GRID_GROWTH each, out to +-GROUND_FAR."""
    a, b = fine
    lines = list(np.linspace(a, b, round((b - a) / step) + 1))
    for sign, start in ((1.0, b), (-1.0, a)):
        size, x, more = step, start, []
        while abs(x) < GROUND_FAR:
            size *= GRID_GROWTH
            x = float(np.clip(x + sign * size, -GROUND_FAR, GROUND_FAR))
            more.append(x)
        lines = lines + more if sign > 0 else more[::-1] + lines
    return np.array(lines)


def grid_part(name, pos, keep, attributes, mat, parent):
    """A mesh object of the grid cells where keep is set, with per-point float attributes."""
    nu, nw = pos.shape[:2]
    idx = np.arange(nu * nw).reshape(nu, nw)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1)[keep]  # counter-clockwise from above
    used, corners = np.unique(quads, return_inverse=True)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(used))
    mesh.vertices.foreach_set("co", pos.reshape(-1, 3)[used].astype(np.float32).ravel())
    mesh.loops.add(quads.size)
    mesh.loops.foreach_set("vertex_index", corners.ravel().astype(np.int32))
    mesh.polygons.add(len(quads))
    mesh.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4, dtype=np.int32))
    mesh.update(calc_edges=True)
    mesh.shade_smooth()
    for key, values in attributes.items():
        mesh.attributes.new(key, "FLOAT", "POINT").data.foreach_set("value", values.ravel()[used].astype(np.float32))
    mesh.materials.append(mat)
    obj = porch.link_object(bpy.data.objects.new(name, mesh))
    obj.parent = parent
    return obj


def corners(a, op):
    """op (np.minimum or np.maximum) over each grid cell's four corners."""
    return op.reduce([a[:-1, :-1], a[1:, :-1], a[1:, 1:], a[:-1, 1:]])


def build_valley(mat, frame):
    """The valley from VALLEY_FROM m past the lip out to GROUND_FAR: rings around the car, as one mesh."""
    r0 = 0.9 * (LIP_U + VALLEY_FROM)  # the valley comes no nearer than this
    rs = r0 * VALLEY_GROWTH ** np.arange(math.ceil(math.log(GROUND_FAR / r0) / math.log(VALLEY_GROWTH)) + 1)
    rs[-1] = GROUND_FAR
    yaws = np.radians(np.linspace(-180.0, 180.0, round(360.0 / VALLEY_STEP) + 1))  # the seam is uphill, behind the car
    R, A = np.meshgrid(rs, yaws, indexing="ij")
    U, W = R * np.cos(A), R * np.sin(A)
    d = lateral(U, W)
    keep = corners(d, np.maximum) > LIP_U + VALLEY_FROM
    used = np.zeros(U.shape, bool)
    for i, j in ((0, 0), (1, 0), (1, 1), (0, 1)):
        used[i:i + keep.shape[0], j:j + keep.shape[1]] |= keep
    Z = np.zeros(U.shape)
    Z[used] = ground_height(U[used], W[used])[0]  # only the points in use: the noise is computed point by point
    return grid_part("valley", np.stack([U, W, Z], -1), keep, {"lateral": d}, mat, frame)


def build_ground(frame, trees, log):
    """The ground in three objects: the pull-out near the car (subdivided and displaced at render time), the valley in
    rings, and the rest, which carries the trees. Returns the rest."""
    t = time.time()
    us, ws = axis(U_FINE, U_STEP), axis(W_FINE, W_STEP)
    U, W = np.meshgrid(us, ws, indexing="ij")
    Z, d = ground_height(U, W)
    X, Y = world_xy(U, W)
    pos = np.stack([U, W, Z], -1)
    uc, wc = (us[:-1] + us[1:]) / 2, (ws[:-1] + ws[1:]) / 2
    near = (((uc > PULLOUT_U[0]) & (uc < PULLOUT_U[1]))[:, None] & ((wc > PULLOUT_W[0]) & (wc < PULLOUT_W[1]))[None, :])
    mat = ground_material()
    pullout = grid_part("pullout", pos, near, {"lateral": d}, mat, frame)
    mod = pullout.modifiers.new("dice", "SUBSURF")
    mod.subdivision_type = "SIMPLE"  # keeps the grid's shape; the displacement adds the detail
    mod.use_adaptive_subdivision = True
    mod.adaptive_space = "PIXEL"
    mod.adaptive_pixel_size = DICING_PX
    attributes = {"lateral": d}
    if trees:
        attributes["dens_hill"] = tree_density(d, X, Y, W)
    terrain = grid_part("terrain", pos, ~near & (corners(d, np.minimum) < LIP_U + GRID_TO), attributes, mat, frame)
    valley = build_valley(mat, frame)
    log(f"Ground: {len(us)} x {len(ws)} grid out to {GROUND_FAR / 1000:g} km; pull-out {len(pullout.data.polygons):,} "
        f"faces (adaptive, {DICING_PX:g} px), the rest {len(terrain.data.polygons):,}; valley rings "
        f"{len(valley.data.polygons):,}; {time.time() - t:.0f} s")
    return terrain


# ---------------------------------------------------------------------------------------------
# Materials.

def ramp(nt, value, a, b):
    """Smoothstep from 0 at a to 1 at b (a > b runs the other way)."""
    r = porch.node(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    r.inputs["From Min"].default_value, r.inputs["From Max"].default_value = a, b
    nt.links.new(value, r.inputs["Value"])
    return r.outputs[0]


def noise_tex(nt, vector, size, detail):
    """Noise of about `size` metres on a coordinate vector (Fac, mostly 0.3-0.7)."""
    tex = porch.node(nt, "ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 1.0 / size
    tex.inputs["Detail"].default_value = detail
    nt.links.new(vector, tex.inputs["Vector"])
    return tex.outputs["Fac"]


def tiled(nt, vector, size):
    mapping = porch.node(nt, "ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0 / size,) * 3
    nt.links.new(vector, mapping.inputs["Vector"])
    return mapping.outputs["Vector"]


def ground_material():
    """One material for all the ground: asphalt on the road, gravel on the pull-out and the road's shoulder, 05's grass
    and forest floor on the hillside and the drop (its far colour beyond 300 m), with soft, wandering edges. Where the
    ground is subdivided, the gravel (and a little of the asphalt) is truly displaced."""
    mat = bpy.data.materials.new("s2_ground")
    mat.displacement_method = "BOTH"  # true displacement where subdivided; finer than that, as bump
    mat.max_vertex_displacement = GRAVEL_RELIEF / 2
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def m(op, a, b=None, c=None):
        return porch.math_node(nt, op, a, b, c)

    coord = porch.node(nt, "ShaderNodeTexCoord").outputs["Object"]  # the car's frame
    split = porch.node(nt, "ShaderNodeSeparateXYZ")
    nt.links.new(coord, split.inputs[0])
    u, w = split.outputs["X"], split.outputs["Y"]
    across = porch.node(nt, "ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="lateral").outputs["Fac"]
    across = m("ADD", across, m("MULTIPLY", m("SUBTRACT", noise_tex(nt, coord, 0.9, 3.0), 0.5), 5 * EDGE_JITTER))
    road = m("MULTIPLY", ramp(nt, across, ROAD[0] - 0.12, ROAD[0] + 0.12), ramp(nt, across, ROAD[1] + 0.12, ROAD[1] - 0.12))
    gravel = m("MULTIPLY", m("SUBTRACT", 1.0, road),
               m("MULTIPLY", ramp(nt, across, ROAD[0] - 1.0, ROAD[0] - 0.5), ramp(nt, across, LIP_U + 0.9, LIP_U + 0.2)))

    gv, av, hv = tiled(nt, coord, GRAVEL_TILE), tiled(nt, coord, ASPHALT_TILE), tiled(nt, coord, GROUND_TILE)
    gravel_tex = {k: porch.image_node(nt, TEXTURES / "gravel_road" / f"gravel_road_{k}_2k.jpg", k == "diff", gv, False)
                  for k in ("diff", "rough", "disp")}
    asphalt_tex = {k: porch.image_node(nt, TEXTURES / "asphalt_02" / f"asphalt_02_{k}_4k.jpg", k == "diff", av, False)
                   for k in ("diff", "rough", "disp")}
    # 05's ground: grass and forest floor mixed in patches, a procedural forest-floor colour beyond 300-900 m.
    patches = ramp(nt, noise_tex(nt, coord, 25.0, 2.0), 0.42, 0.58)
    near_col = porch.mix(nt, porch.image_node(nt, TEXTURES / "sparse_grass" / "sparse_grass_diff_2k.jpg", True, hv),
                         porch.image_node(nt, TEXTURES / "forrest_ground_01" / "forrest_ground_01_diff_2k.jpg", True, hv),
                         patches)
    near_rough = porch.mix(nt, porch.image_node(nt, TEXTURES / "sparse_grass" / "sparse_grass_rough_2k.jpg", False, hv),
                           porch.image_node(nt, TEXTURES / "forrest_ground_01" / "forrest_ground_01_rough_2k.jpg", False,
                                            hv), patches, "Float")
    far_col = porch.mix(nt, (0.018, 0.026, 0.014, 1), (0.045, 0.042, 0.026, 1), noise_tex(nt, coord, 250.0, 6.0))
    length = porch.node(nt, "ShaderNodeVectorMath", operation="LENGTH")
    nt.links.new(coord, length.inputs[0])
    far = ramp(nt, length.outputs["Value"], 300.0, 900.0)
    ground_col = porch.mix(nt, near_col, far_col, far)
    ground_rough = porch.mix(nt, near_rough, 0.9, far, "Float")

    color = porch.mix(nt, porch.mix(nt, ground_col, gravel_tex["diff"], gravel), asphalt_tex["diff"], road)
    rough = porch.mix(nt, porch.mix(nt, ground_rough, gravel_tex["rough"], gravel, "Float"), asphalt_tex["rough"], road,
                      "Float")
    nt.links.new(color, bsdf.inputs["Base Color"])
    nt.links.new(rough, bsdf.inputs["Roughness"])
    # The forest floor's own height map as a bump, on the near ground only (not the gravel or the road).
    ground_only = m("MULTIPLY", m("MULTIPLY", m("SUBTRACT", 1.0, road), m("SUBTRACT", 1.0, gravel)), m("SUBTRACT", 1.0, far))
    bump = porch.node(nt, "ShaderNodeBump")
    bump.inputs["Distance"].default_value = BANK_BUMP
    nt.links.new(ground_only, bump.inputs["Strength"])
    nt.links.new(porch.image_node(nt, TEXTURES / "forrest_ground_01" / "forrest_ground_01_disp_2k.jpg", False, hv),
                 bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Displacement, faded out before the pull-out object's edges (so it meets the rest of the ground exactly).
    inside = m("MULTIPLY", m("MULTIPLY", ramp(nt, u, PULLOUT_U[0] + 0.3, PULLOUT_U[0] + 0.8),
                             ramp(nt, u, PULLOUT_U[1] - 0.2, PULLOUT_U[1] - 0.6)),
               m("MULTIPLY", ramp(nt, w, PULLOUT_W[0] + 0.5, PULLOUT_W[0] + 2.0),
                 ramp(nt, w, PULLOUT_W[1] - 0.5, PULLOUT_W[1] - 5.0)))
    height = m("ADD", m("MULTIPLY", gravel, m("MULTIPLY", m("SUBTRACT", gravel_tex["disp"], 0.5), GRAVEL_RELIEF)),
               m("MULTIPLY", road, m("MULTIPLY", m("SUBTRACT", asphalt_tex["disp"], 0.5), ASPHALT_RELIEF)))
    disp = porch.node(nt, "ShaderNodeDisplacement")
    nt.links.new(m("MULTIPLY", height, inside), disp.inputs["Height"])
    disp.inputs["Midlevel"].default_value = 0.0
    disp.inputs["Scale"].default_value = 1.0
    nt.links.new(disp.outputs[0], nt.nodes["Material Output"].inputs["Displacement"])
    return mat


# ---------------------------------------------------------------------------------------------
# The guardrail, trees, haze and car.

def build_rail(frame):
    """Square posts every POST_STEP, and a beam on their road side in 4 m lengths, each its own patch of 05's wood."""
    coll = night10.new_collection("guardrail")
    beam_mat = porch.wood_material("guardrail_beam", "wood_table_worn", value=0.8)
    post_mat = porch.wood_material("guardrail_post", "wood_table_worn", value=0.7, repeat=(1.8, 0.55))
    level = -CROWN - APRON_FALL * (RAIL_U - APRON_FLAT)  # the pull-out under the rail, before its unevenness
    ws = np.arange(RAIL_W[0], RAIL_W[1] + 1e-6, POST_STEP)
    up = math.radians(90.0)
    for i, w in enumerate(ws):
        porch.box(f"guardrail_post_{i}", (POST[1] + POST[2], POST[0], POST[0]),
                  (RAIL_U + POST[0] / 2, w, level + (POST[1] - POST[2]) / 2), (0.0, up, 0.0), post_mat, 0.008,
                  coll).parent = frame
    for i, (a, b) in enumerate(zip(ws[:-2:2], ws[2::2], strict=True)):
        porch.box(f"guardrail_beam_{i}", (b - a, BEAM[0], BEAM[1]), (RAIL_U - BEAM[0] / 2, (a + b) / 2,
                                                                      level + BEAM[2] - BEAM[1] / 2),
                  (0.0, 0.0, up), beam_mat, 0.006, coll).parent = frame
    return len(ws)


def build_trees(terrain, log):
    """Handoff 14's light pines and firs (made once, saved beside the originals), with 05's tint and scatter."""
    t = time.time()
    coll = night10.new_collection("trees_hill")
    near, _ = porch.load_light_trees(coll)
    porch.add_tint({m for o in near for m in o.data.materials if m})
    bpy.context.scene.view_layers[0].layer_collection.children[coll.name].exclude = True  # they only feed the scatter
    porch.scatter_modifier(terrain, "scatter_hill", coll, "dens_hill", porch.NEAR_DENSITY, 61, porch.NEAR_SCALE, 4.0, 3.5)
    log(f"Trees: {len(near)} light variants loaded in {time.time() - t:.1f} s, scattered on the hillside")


def build_beam_haze(frame):
    """A box of thin mist low over the drop, in the car's frame: it fades in from the nose to the lip and out toward its
    far end, sides, top and bottom, in flat wisps with clear air between them."""
    lo, hi = HAZE_BOX
    centre = tuple((a + b) / 2 for a, b in zip(lo, hi, strict=True))
    mat = bpy.data.materials.new("beam_haze")
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])

    def m(op, a, b=None, c=None):
        return porch.math_node(nt, op, a, b, c)

    shift = porch.node(nt, "ShaderNodeMapping")
    shift.inputs["Location"].default_value = centre  # the box's own coordinates, moved to the car's frame
    nt.links.new(porch.node(nt, "ShaderNodeTexCoord").outputs["Object"], shift.inputs["Vector"])
    coord = shift.outputs["Vector"]
    split = porch.node(nt, "ShaderNodeSeparateXYZ")
    nt.links.new(coord, split.inputs[0])
    u, w, z = split.outputs["X"], split.outputs["Y"], split.outputs["Z"]
    fade = HAZE_FADE
    shape = m("MULTIPLY", m("MULTIPLY", ramp(nt, u, lo[0], LIP_U), ramp(nt, u, fade["far"][1], fade["far"][0])),
              m("MULTIPLY", ramp(nt, m("ABSOLUTE", w), fade["side"][1], fade["side"][0]),
                m("MULTIPLY", ramp(nt, z, fade["top"][1], fade["top"][0]),
                  ramp(nt, z, fade["bottom"][1], fade["bottom"][0]))))
    flat = porch.node(nt, "ShaderNodeMapping")
    flat.inputs["Scale"].default_value = (1.0, 1.0, 2.5)  # wisps flatter than they're wide
    nt.links.new(coord, flat.inputs["Vector"])
    clear = 0.3 + 0.4 * HAZE_CLEAR  # the noise is mostly 0.3-0.7: below this it's clear air
    patches = ramp(nt, noise_tex(nt, flat.outputs["Vector"], HAZE_WISPS, 4.0), clear, clear + 0.15)
    vol = porch.node(nt, "ShaderNodeVolumeScatter", phase="HENYEY_GREENSTEIN")
    vol.inputs["Anisotropy"].default_value = HAZE_ANISOTROPY
    nt.links.new(m("MULTIPLY", m("MULTIPLY", shape, patches), HAZE_DENSITY), vol.inputs["Density"])
    nt.links.new(vol.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    porch.box("beam_haze", size, centre, mat=mat, bevel=0.0).parent = frame


def place_car(scene):
    """The car on the pull-out, its footprint centred on the frame's origin, nose along u, wheels straight, lights on.
    Returns its two tail-light lamps."""
    car = car_lib.build(scene, paint="guards-red")
    car_lib.set_lights(car, head=1.0, tail=1.0)
    tails = [o for o in car.root.children if o.type == "LIGHT" and o.data in car.tail_lamps]  # before they change type
    for obj in tails:
        obj.data.type = "SPOT"
        spot = bpy.data.lights[obj.data.name]  # the same light, seen as a spot now
        spot.spot_size, spot.spot_blend = math.radians(TAIL_CONE), 0.5
        spot.energy = car_lib.TAILLIGHT_LAMP_W * TAIL_POWER
        obj.rotation_euler = (math.radians(90.0 - TAIL_DOWN), 0.0, 0.0)  # a spot shines down its -Z: back, a little down
    turn = TURN + math.radians(90.0)  # the model faces -Y
    car.root.rotation_euler = (0.0, 0.0, turn)
    car.root.location = to_world(0.0, 0.0, -CROWN - CAR_SINK) - Matrix.Rotation(turn, 3, "Z") @ Vector(
        (0.0, car.centre_y, 0.0))
    bpy.context.view_layer.update()
    if len(tails) != 2 or any(o.data.type != "SPOT" for o in tails):
        raise SystemExit(f"Expected car.py's two tail-light lamps, found {[o.name for o in tails]}")
    return tails


def add_camera(scene, shot):
    cam = bpy.data.cameras.new(shot.name)
    cam.lens, cam.sensor_fit, cam.sensor_width = shot.lens, "HORIZONTAL", 36.0
    cam.clip_start, cam.clip_end = 0.05, 60000.0
    cam.dof.use_dof, cam.dof.focus_distance, cam.dof.aperture_fstop = True, shot.focus, porch.CAM_FSTOP
    obj = porch.link_object(bpy.data.objects.new(shot.name, cam))
    u, w = shot.at
    obj.location = to_world(u, w, height_at(u, w) + shot.height)
    obj.rotation_euler = (math.radians(90.0 + shot.tilt), 0.0, -math.radians(HEADING - shot.look))
    scene.camera = obj
    bpy.context.view_layer.update()
    return obj


def describe(shot):
    u, w = shot.at
    where = f"{-u:.2f} m behind the centre" if u else f"{-w:.2f} m out from its right side"
    return (f"{shot.what}: {where}, {shot.height:.2f} m up, {shot.tilt:g} deg up, {shot.lens:g} mm, "
            f"f/{porch.CAM_FSTOP:g} focused at {shot.focus:g} m")


# ---------------------------------------------------------------------------------------------
# The night, the run and the outputs.

def build_far_ridges():
    """05's far ridges, the nearest lowered to RIDGE1_CREST, in narrow wedges over RIDGE_SPANS. Only this run's copy of
    05's settings changes (05 is loaded as a module per run), and it's put back after."""
    ridges, yaws = porch.FAR_RIDGES, porch.YAW_RANGE
    porch.FAR_RIDGES = [(ridges[0][0], RIDGE1_CREST, ridges[0][2]), *ridges[1:]]
    for lo, hi in RIDGE_SPANS:
        for a, b in itertools.pairwise(np.linspace(lo, hi, RIDGE_SPLIT + 1)):
            porch.YAW_RANGE = (float(a), float(b))
            porch.build_far_ridges()
    porch.FAR_RIDGES, porch.YAW_RANGE = ridges, yaws


def night_sky(scene, frame, fringe):
    """Section 1's frame 69 as 11 renders it, the aurora's top at `fringe` (10's top_share; None keeps 10's). 11's
    show_frame can't be used as it is: it also switches the porch's sconces and 08's lantern."""
    key = frame.key
    porch.SUN_ELEVATION = key.sun
    porch.build_world(scene)
    kf.add_night_sky(scene.world)
    look = night10.LOOKS[frames11.AURORA_LOOK]
    if fringe is not None:
        look = dataclasses.replace(look, top_share=fringe)
    night10.add_aurora(scene.world, look, frame.aurora, frame.phase, **frames11.AURORA_RAYS)
    scene.view_settings.exposure = key.exposure
    porch.GRADE_MASK = tuple(g * 2 ** (kf.GRADE_TUNED_AT - key.exposure) for g in kf.GRADE_MASK_R3)
    porch.build_compositor(scene)


def ram_gb():
    """This Blender process's RAM now and at its peak so far (Windows' working set), in GB."""
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(porch._MemoryCounters), ctypes.c_ulong]
    c = porch._MemoryCounters(cb=ctypes.sizeof(porch._MemoryCounters))
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2**30, c.PeakWorkingSetSize / 2**30


def tail_lights(scene, cam, lamps, path):
    """The tail-lights in the still (px from its top-left), left then right: projected from the lamps, and measured as
    the centre of the brightest red near each projection."""
    r = scene.render
    size = (r.resolution_x * r.resolution_percentage // 100, r.resolution_y * r.resolution_percentage // 100)
    lamps = sorted(lamps, key=lambda o: world_to_camera_view(scene, cam, o.matrix_world.translation).x)
    px = kf.load(path)[::-1, :, :3]  # rows from the top
    red = px[..., 0] * (px[..., 0] - np.maximum(px[..., 1], px[..., 2]) > 0.15)
    found = []
    for lamp in lamps:
        p = world_to_camera_view(scene, cam, lamp.matrix_world.translation)
        x, y = p.x * size[0], (1.0 - p.y) * size[1]
        reach = round(0.04 * size[0])
        x0, y0 = max(round(x) - reach, 0), max(round(y) - reach, 0)
        window = red[y0:round(y) + reach, x0:round(x) + reach]
        ys, xs = np.nonzero(window >= 0.9 * window.max()) if window.max() > 0 else ([], [])
        measured = (x0 + float(np.mean(xs)), y0 + float(np.mean(ys))) if len(xs) else None
        found.append(((x, y), measured))
    return found


def update_summary(name, lines):
    """Replace this still's block in the summary, keeping the other's."""
    blocks = {}
    if SUMMARY.exists():
        for block in re.split(r"(?m)^## ", SUMMARY.read_text(encoding="utf-8"))[1:]:
            head, _, body = block.partition("\n")
            blocks[head.strip()] = body.strip("\n")
    blocks[name] = "\n".join(lines)
    SUMMARY.write_text("Section 2 look, round 2 (handoff 15), 2560x1440: one block per still and sky option, from the run "
                       "that rendered it."
                       "\n\n" + "\n\n".join(f"## {k}\n{blocks[k]}" for k in sorted(blocks)) + "\n", encoding="utf-8")


def parse_args(argv):
    p = argparse.ArgumentParser(prog="13_s2_look.py")
    p.add_argument("--still", required=True, choices=list(SHOTS), help="which shot to render")
    p.add_argument("--sky", required=True, choices=list(SKIES), help="the sky option (see the docstring)")
    p.add_argument("--preview", metavar="NAME",
                   help="1280x720 test into renders/review/15/preview/NAME-<still>-<sky>.webp")
    p.add_argument("--samples", type=int, help="override the sample count (previews only)")
    p.add_argument("--no-trees", action="store_true", help="skip the trees (previews only)")
    args = p.parse_args(argv)
    if (args.samples or args.no_trees) and args.preview is None:
        p.error("--samples and --no-trees are for previews")
    return args


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    shots, fringe = SKIES[args.sky]
    shot, preview = shots[args.still], args.preview is not None
    label = f"{shot.name}-{args.sky}"
    started = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    frame = {int(f.name): f for f in frames11.frame_table()}[FRAME]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    samples = args.samples or (porch.PREVIEW_SAMPLES if preview else porch.FINAL_SAMPLES)
    devices = porch.setup_render(scene, preview, samples)
    scene.cycles.offscreen_dicing_scale = OFFSCREEN_DICING
    night_sky(scene, frame, fringe)  # after setup_render, which sets 05's daytime exposure
    ram = [("sky", ram_gb()[0])]
    set_frame = porch.link_object(bpy.data.objects.new("set", None))
    set_frame.location, set_frame.rotation_euler.z = SET_AT, TURN
    tails = place_car(scene)
    ram.append(("car", ram_gb()[0]))
    terrain = build_ground(set_frame, not args.no_trees, log)
    ram.append(("ground", ram_gb()[0]))
    posts = build_rail(set_frame)
    if not args.no_trees:
        build_trees(terrain, log)
        ram.append(("trees", ram_gb()[0]))
    build_far_ridges()
    porch.build_atmosphere()
    build_beam_haze(set_frame)
    ram.append(("far ridges, mist and haze", ram_gb()[0]))
    cam = add_camera(scene, shot)
    key = frame.key
    log(f"{label}: {describe(shot)}")
    log(f"Night: frame {FRAME} (sun {key.sun:+g} deg, exposure {key.exposure:+g} EV, aurora {frame.aurora:g} at phase "
        f"{frame.phase:g}, rays {frames11.AURORA_RAYS}); night glow {kf.AIRGLOW}; sky option {args.sky}: aurora "
        f"top_share {night10.LOOKS[frames11.AURORA_LOOK].top_share if fringe is None else fringe:g}")
    log(f"Set: shelf {-SET_AT[2]:g} m below the porch, heading {HEADING:g} deg; road u {ROAD}, guardrail at u {RAIL_U:g} "
        f"({posts} posts, w {RAIL_W[0]:g} to {RAIL_W[1]:g}), lip at u {LIP_U:g}; beam haze {HAZE_DENSITY:g}/m in "
        f"{HAZE_WISPS:g} m wisps, anisotropy {HAZE_ANISOTROPY:g}; far ridge 1 crest {RIDGE1_CREST:g} m; tail lamps at "
        f"{TAIL_POWER:g} of car.py's power")
    r = scene.render
    log(f"Render: {r.resolution_x * r.resolution_percentage // 100}x{r.resolution_y * r.resolution_percentage // 100}, "
        f"{samples} samples, on {devices}; trees {'off' if args.no_trees else 'on'}")
    built = time.time() - started
    log(f"Scene built in {built:.0f} s; RAM now {ram_gb()[0]:.1f} GB; rendering ...")
    seconds, peak_mb = kf.render(scene)
    peak = ram_gb()[1]
    log(f"Render time: {seconds:.1f} s; Cycles peak memory {peak_mb:,.0f} MB; peak RAM over the run {peak:.1f} GB")

    if preview:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        path = PREVIEW / f"{args.preview}-{label}.webp"
        kf.save_render(scene, path, "WEBP", 95)
        log(f"Wrote {path.relative_to(ROOT)}")
        with open(PREVIEW / "preview-log.txt", "a", encoding="utf-8") as f:
            f.write(f"--- {args.preview} {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lines) + "\n")
        return

    REVIEW.mkdir(parents=True, exist_ok=True)
    png, webp = REVIEW / f"15-{label}.png", REVIEW / f"15-{label}.webp"
    kf.save_render(scene, png, "PNG")  # 8-bit, as 11's masters
    im = frames09.image_from(kf.load(png), porch.FINAL_RES)  # a new image, so Blender honours the WebP quality
    im.file_format = "WEBP"
    im.save(filepath=str(webp), quality=frames09.NIGHT_QUALITY)
    bpy.data.images.remove(im)
    block = [(f"{label} | {describe(shot)} | render {seconds:.1f} s | Cycles {peak_mb:,.0f} MB | RAM peak "
              f"{peak:.1f} GB | {time.strftime('%Y-%m-%d %H:%M')}"),
             (f"Built in {built:.0f} s. RAM after each step (GB): " + ", ".join(f"{k} {v:.1f}" for k, v in ram)
              + f"; peak {peak:.1f} during the render"),
             (f"Files: {png.name} ({png.stat().st_size / 2**20:.1f} MB), {webp.name} "
              f"({webp.stat().st_size / 2**20:.1f} MB, WebP quality {frames09.NIGHT_QUALITY})")]
    if shot.name == "C1":
        (proj_l, meas_l), (proj_r, meas_r) = tail_lights(scene, cam, tails, png)

        def xy(p):
            return f"({p[0]:.0f}, {p[1]:.0f})" if p else "(not found)"

        block.append(f"Tail-lights, px from the top-left of the 2560x1440 still: projected from the lamps, left "
                     f"{xy(proj_l)}, right {xy(proj_r)}; measured (centre of the brightest red), left {xy(meas_l)}, "
                     f"right {xy(meas_r)}")
    for line in block:
        print(line, flush=True)
    update_summary(label, block)
    print(f"Wrote {png.name}, {webp.name} and its block in {SUMMARY.name}; run time {time.time() - started:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
