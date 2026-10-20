r"""Handoffs 05-07: golden-hour look test. One still from a cabin porch on a ledge above a misty mountain valley.

Run headless from the repo root (Blender isn't on PATH, so call it by full path):
    & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender\05_porch_still.py
    ... -P blender\05_porch_still.py -- --preview NAME [--samples N]   # 1280x720 test into preview/

The full run renders round 3 at 2560x1440 and writes, into renders/review/07/:
    07-porch-r3.png (lossless master), 07-porch-r3.webp, 07-compare-r3.webp, checks/*, 07-render-log.txt
Earlier rounds' files in renders/review/05/ and 06/ are never written (round 2's still is read for the compare sheet).
Safe to re-run: it starts from an empty scene every time. Missing Poly Haven assets are downloaded
on first run (official API only, md5-checked) into assets-source/models|textures/<id>/.
"""

import ctypes
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from itertools import pairwise
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector, noise

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets-source"
REF_DIR = ASSETS / "reference" / "golden-hour"
PREVIOUS = ROOT / "renders" / "review" / "06" / "06-porch-r2.webp"  # round 2, read only, for the compare sheet
OUT = ROOT / "renders" / "review" / "07"
TAG = "07"  # file names: 07-porch-r3.png, checks/07-grey-r3.webp, ...
ROUND = "r3"
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


# ---------------------------------------------------------------------------------------------
# Poly Haven downloads: official API only, every file md5-checked, every path kept inside the
# asset's own folder. Assets already on disk are used as they are.

API = "https://api.polyhaven.com"
ALLOWED_HOSTS = {"api.polyhaven.com", "dl.polyhaven.org"}


def _check_url(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise SystemExit(f"Refusing download from outside Poly Haven: {url}")


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only if its target passes the same https + Poly Haven host check."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_CheckedRedirects)


def _open(url):
    _check_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": "scroll-journey-05"})
    return _OPENER.open(req, timeout=120)


def _inside(base, relative):
    """base/relative, or stop if that would land outside base (absolute paths, '..', drives)."""
    base = base.resolve()
    dest = (base / relative).resolve()
    if not dest.is_relative_to(base) or dest == base:
        raise SystemExit(f"Refusing path outside {base}: {relative!r}")
    return dest


def _download(entry, dest):
    """Stream one API file entry ({url, md5, size}) to dest; only a verified file gets the name."""
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()
    try:
        with _open(entry["url"]) as resp, open(part, "wb") as f:
            while chunk := resp.read(1 << 20):
                md5.update(chunk)
                f.write(chunk)
        if md5.hexdigest() != entry["md5"]:
            raise SystemExit(f"md5 mismatch for {entry['url']}; file deleted.")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)  # a failed or rejected download leaves nothing behind


def _api_files(asset_id):
    if not asset_id.replace("_", "").isalnum():
        raise SystemExit(f"Bad asset id {asset_id!r}")
    with _open(f"{API}/files/{asset_id}") as resp:
        return json.load(resp)


def check_gltf_paths(gltf_path):
    """Stop if a glTF points at any file outside its own folder (or at a URL / data URI)."""
    data = json.loads(gltf_path.read_text(encoding="utf-8"))
    for item in data.get("buffers", []) + data.get("images", []):
        uri = item.get("uri")
        if uri is None:
            continue
        if ":" in uri:
            raise SystemExit(f"{gltf_path.name}: refusing non-file uri {uri[:60]!r}")
        _inside(gltf_path.parent, urllib.parse.unquote(uri))


def fetch_model(asset_id, res="2k"):
    """Path to assets-source/models/<id>/<id>_<res>.gltf, downloading the set if it's missing."""
    folder = ASSETS / "models" / asset_id
    gltf = folder / f"{asset_id}_{res}.gltf"
    if not gltf.exists():
        print(f"Downloading model {asset_id} ({res}) from Poly Haven ...", flush=True)
        entry = _api_files(asset_id)["gltf"][res]["gltf"]
        for rel, sub in entry.get("include", {}).items():
            dest = _inside(folder, rel)
            if not dest.exists():
                _download(sub, dest)
        _download(entry, _inside(folder, gltf.name))
    check_gltf_paths(gltf)
    return gltf


TEXTURE_MAPS = {"diff": "Diffuse", "nor_gl": "nor_gl", "rough": "Rough"}


def fetch_texture(asset_id, res="2k"):
    """Folder assets-source/textures/<id>/ holding <id>_{diff,nor_gl,rough}_<res>.jpg."""
    folder = ASSETS / "textures" / asset_id
    wanted = {k: folder / f"{asset_id}_{k}_{res}.jpg" for k in TEXTURE_MAPS}
    missing = {k: p for k, p in wanted.items() if not p.exists()}
    if missing:
        print(f"Downloading texture {asset_id} ({res}) from Poly Haven ...", flush=True)
        files = _api_files(asset_id)
        for kind, path in missing.items():
            _download(files[TEXTURE_MAPS[kind]][res]["jpg"], _inside(folder, path.name))
    return folder


# ---------------------------------------------------------------------------------------------
# Settings. Lengths in metres; the deck top is z = 0 and the valley lies toward +Y.

FINAL_RES = (2560, 1440)
PREVIEW_PCT = 50  # previews are 1280x720, a quarter of the pixels
FINAL_SAMPLES = 160  # fits the ~5 min per frame budget (hundreds of frames later); round 1 used 128
PREVIEW_SAMPLES = 48
SEED = 5

# Camera: eye height on the porch, turned right so the rail runs diagonally across the frame.
CAM_LOC = (0.0, -2.5, 1.6)
CAM_YAW = 45.0  # degrees from +Y toward +X
CAM_PITCH = -5.0  # degrees, negative looks down
CAM_LENS = 28.0  # mm on a 36 mm full-frame sensor
CAM_SHIFT_Y = 0.0
CAM_FOCUS = 6.0  # m; just past the 28 mm f/8 hyperfocal distance (~3.3 m)
CAM_FSTOP = 8.0

# Sun: setting just above the far ridge, right of centre. Seen from the camera the skyline here
# (the 8 km ridge) sits at 2.5-2.7 degrees, so at 3.0 the disc clears it by about a third of its width.
SUN_YAW = 55.0
SUN_ELEVATION = 3.0
SUN_ANGLE = 0.53  # degrees, the real sun's size
# The Multiple Scattering sky: air (blue sky, redder sun), aerosol (haze and glow around the sun),
# ozone. 1.0 is the model's standard atmosphere.
SKY_AIR = 1.0
SKY_AEROSOL = 3.0
SKY_OZONE = 1.0
SKY_ALTITUDE = 1500.0  # metres above sea level; less air above a mountain porch keeps the low sun yellower
# Irradiance and colour of the sky's own sun disc at SUN_ELEVATION and the SKY_* settings, measured
# with a white test plane facing the sun (sky with disc minus sky without disc, against a
# strength-1 Sun lamp). The Sun lamp copies it, so the lit porch and the visible sun agree.
# Re-measure if the sun or any SKY_* value changes. (Round 1, 6 deg: 82.0, (1.0, 0.554, 0.158).)
SUN_STRENGTH = 62.2
SUN_COLOR = (1.0, 0.453, 0.074)
EXPOSURE = -2.0  # lower the camera exposure rather than dimming the physical sky
VIEW_TRANSFORM = "AgX"
# Facing the sun, the sky is many stops brighter than the porch. A low-contrast look compresses that
# range the way the references (HDR-style photos) do, so the shadows keep detail and the sky holds.
LOOK = "AgX - Low Contrast"
# White balance, as the camera's: the colour temperature treated as neutral white. 6500 K / tint 10
# is daylight (no change); a higher value renders the scene warmer, like a "cloudy" camera setting.
WB_TEMPERATURE = 6500.0
WB_TINT = 10.0
# Bloom in the compositor (Blender's Glare node): the soft glow a real lens adds around the sun and
# the brightest highlights. Values are in render units before exposure. Measured at 3 deg: the sun's
# core ~540,000, the sky beside it 16-25, the lit deck ~0.7.
BLOOM_TYPE = "Fog Glow"  # Blender's lens-like glow; spreads like a real lens rather than a blur
BLOOM_THRESHOLD = 40.0  # above the brightest sky, so only the sun feeds the glow
BLOOM_MAX = 10000.0  # clamp what feeds it: sets how bright the halo is
BLOOM_STRENGTH = 1.6
BLOOM_SIZE = 0.7  # how far the glow spreads, as a share of the frame
# Grade: AgX fades bright colours toward white, which turns the lit sky and clouds pastel. A mild
# saturation lift before the view transform keeps the colour the sun's light gives them. It fades in
# over this brightness range (render units), so the porch (lit deck ~0.7) stays as rendered.
GRADE_SATURATION = 1.35
GRADE_MASK = (1.5, 6.0)

# Porch layout.
PORCH_X = (-5.0, 13.0)
DECK_Y = (-6.2, 0.08)
POSTS_X = [0.65 + 3.6 * i for i in range(4)]
POST_TOP = 2.55
RAIL_TOP = 0.975
EAVE_SLOPE = 0.25  # rafters rise this much per metre toward the cabin
MUG_AT = (1.3, 0.0)

# Valley.
FLOOR_Z = -380.0  # valley floor, metres below the deck
LEDGE_Z = -1.4  # the ground under the deck
# The drop-off: the cabin sits on a ledge. Just past the rail the ground falls away, steepest at the lip
# (about 72 degrees) and easing toward the valley floor. It falls faster than any line of sight through or
# under the rail, so from the porch no ground shows below the rail: the view drops straight into the mist.
DROP_LIP = 0.6  # metres past the rail line where the fall starts (the deck ends at 0.08)
DROP_SCALE = 120.0  # metres; the fall's e-folding length
# The wooded shoulder right of the porch, where the backlit pines stand. The deck hides everything below the line
# of sight that grazes its outer end (falling 0.62 m per metre past the rail). The shoulder keeps its ground
# SHOULDER_SINK below that line, so neither the 5 cm slit under the rail's bottom board nor the balusters show its
# ground or the pines' bases. Past SHOULDER_END it falls away faster, and its left flank (between the SHOULDER_YAW
# angles, seen from the camera) drops into the cliff.
SHOULDER_SINK = 1.0
SHOULDER_END = 45.0  # metres past the rail line
SHOULDER_YAW = (50.0, 55.0)
# The valley floor's large bumps (hills that rise out of the mist) fade in between these distances past the
# rail. Nearer, the floor stays under the mist, so the view down past the rail is mist only.
BUMPS_FROM = (1500.0, 3000.0)  # round 2: (120, 900), behind the meadow
FAR_RIDGES = [  # (distance m, mean crest height m, crest variation m)
    (8000, 250, 100),
    (13000, 520, 150),
    (20000, 950, 200),
]
YAW_RANGE = (0.0, 100.0)  # terrain and trees are only built in and near the view
EARTH_R = 6_371_000.0

# Mist and haze: one volume whose density falls off with height. The haze has two parts, like real
# air: fine "air" scatter that is bluish and spreads evenly (distance turns paler and bluer), and
# "dust" that is colourless and scatters mostly forward, so looking toward the low sun it glows with
# the sun's own gold. The shares keep the overall visibility close to round 1.
HAZE_FLOOR_DENSITY = 6e-5  # per metre at the valley floor
HAZE_HEIGHT = 600.0  # e-folding height
HAZE_AIR_COLOR = (0.42, 0.66, 1.0, 1.0)  # scatters blue more than red
HAZE_AIR_SHARE = 0.5
HAZE_DUST_SHARE = 0.4
HAZE_DUST_ANISOTROPY = 0.82  # forward scattering of fine dust and haze droplets
MIST_DENSITY = 0.02
MIST_TOP = FLOOR_Z + 35.0
MIST_FADE = 25.0
MIST_DROPLET = 15.0  # micrometres; fog droplets scatter strongly forward (Mie phase)

# High clouds: a thin volume layer lit only by the Sun lamp and the sky, so their colour follows the
# sun when the time-lapse moves it. Noise stretched along the wind makes streaks and broken puffs.
CLOUD_BASE = 5500.0  # metres above the porch, on the curved earth
CLOUD_THICKNESS = 600.0
CLOUD_DENSITY = 0.0011  # per metre where the layer is densest
CLOUD_ANISOTROPY = 0.6  # cloud droplets scatter forward: the clouds glow toward the sun
CLOUD_COVER = 0.58  # noise threshold: higher means fewer, smaller clouds
CLOUD_WIND_YAW = 62.0  # degrees from +Y toward +X; streaks run this way
CLOUD_SIZE = (5000.0, 900.0, 260.0)  # noise scale along the wind, across it, and vertically (m)
# The layer thins out between these distances: keeps the band just above the ridges clear, and the
# sun's line to the porch (which meets the cloud base about 92 km out) free of cloud.
CLOUD_FADE = (68000.0, 84000.0)

# Tall conifers only: the pines on the shoulder use them as they are, the valley (1-9 km) uses decimated copies.
TREE_ASSETS = ["fir_tree_01", "pine_tree_01"]
MID_TREE_POLYS = 120_000  # decimated copies for the 1-9 km band
# The pines on the shoulder stand lower than round 2's (below the rail), so they're a little taller and sparser.
NEAR_DENSITY = 0.03  # trees per square metre where the density map is 1 (Poisson, 3.5 m apart); round 2: 0.06
NEAR_SCALE = (1.0, 1.6)  # 14-33 m tall; round 2: (0.85, 1.45)
MID_DENSITY = 0.03
# Treetops (Nems's pick, b): a few groups of conifers on low knolls in the mist, in the lower left, so only their dark
# tops show, about 350 m below the porch. The mist right behind the mug stays clear, so the mug sits against clean
# mist: the group left of it sits halfway to the post. Each group: (yaw degrees, distance m from the camera,
# radius m), aimed from screen positions. The trees are the decimated valley copies. The mist's top is at -345 and
# fades over 25 m above it, so a top has to stand 35-50 m clear of it to read dark from 2 km: the knolls' crests sit
# 10 m above the mist's top, under the trees and in the mist's fade.
TREETOPS = True  # False gives preview a (mist only)
TREETOP_GROUPS = ((19.8, 1745.0, 75.0), (31.0, 1822.0, 70.0), (21.9, 1074.0, 55.0), (38.3, 2076.0, 60.0))
TREETOP_CREST = -335.0
TREETOP_DENSITY = 0.02  # trees per square metre at a group's centre (Poisson, 4 m apart)
TREETOP_SCALE = (1.2, 2.0)  # 17-41 m tall


# ---------------------------------------------------------------------------------------------
# Small helpers.

def sock(sockets, identifier):
    return next(s for s in sockets if s.identifier == identifier)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def node(nt, kind, **props):
    n = nt.nodes.new(kind)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def math_node(nt, op, a=None, b=None, c=None):
    n = node(nt, "ShaderNodeMath", operation=op)
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, bpy.types.NodeSocket):
            nt.links.new(v, n.inputs[i])
        else:
            n.inputs[i].default_value = v
    return n.outputs[0]


def mix(nt, a, b, factor, kind="Color"):
    """Mix node (colour or float); a, b and factor may be sockets or constants."""
    m = node(nt, "ShaderNodeMix", data_type="RGBA" if kind == "Color" else "FLOAT")
    suffix = "_Color" if kind == "Color" else "_Float"
    for ident, v in (("Factor_Float", factor), ("A" + suffix, a), ("B" + suffix, b)):
        if isinstance(v, bpy.types.NodeSocket):
            nt.links.new(v, sock(m.inputs, ident))
        else:
            sock(m.inputs, ident).default_value = v
    return sock(m.outputs, "Result" + suffix)


def link_object(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


_NOISE_STATS = {}


def fbm_field(xs, ys, scale, offset, octaves=5, ridged=False):
    """Fractal noise at world points, standardised (mean 0, std 1) over a fixed sample so every
    mesh that reads the same field agrees."""
    def one(x, y):
        p = Vector((x / scale + offset, y / scale - offset, offset * 0.37))
        if ridged:
            return noise.ridged_multi_fractal(p, 1.0, 2.0, octaves, 1.0, 2.0)
        return noise.fractal(p, 1.0, 2.0, octaves)

    key = (octaves, ridged)
    if key not in _NOISE_STATS:
        rng = np.random.default_rng(1)
        sample = [one(x, y) for x, y in rng.uniform(-1e6, 1e6, (4000, 2))]
        _NOISE_STATS[key] = (float(np.mean(sample)), float(np.std(sample)) or 1.0)
    mean, std = _NOISE_STATS[key]
    vals = np.fromiter((one(x, y) for x, y in zip(xs.ravel(), ys.ravel(), strict=True)), float, xs.size)
    return ((vals - mean) / std).reshape(xs.shape)


# ---------------------------------------------------------------------------------------------
# Materials.

def image_node(nt, path, color, vector, box=True):
    tex = node(nt, "ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(path), check_existing=True)
    tex.image.colorspace_settings.name = "sRGB" if color else "Non-Color"
    if box:
        tex.projection = "BOX"
        tex.projection_blend = 0.2
    nt.links.new(vector, tex.inputs["Vector"])
    return tex.outputs["Color"]


def wood_material(name, tex_id, repeat=(1.4, 0.55), value=1.0, rough=1.0, hue_jitter=0.025):
    """Poly Haven wood, box-mapped in object space (local X = along the grain, stretched so the
    0.55 m texture repeats less) and shifted per object so no two boards share the same patch,
    with small per-object hue and value changes."""
    folder = fetch_texture(tex_id)
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    info = node(nt, "ShaderNodeObjectInfo")
    coord = node(nt, "ShaderNodeTexCoord")
    shift = node(nt, "ShaderNodeVectorMath", operation="MULTIPLY")
    nt.links.new(info.outputs["Random"], shift.inputs[0])
    shift.inputs[1].default_value = (37.1, 11.3, 5.7)
    moved = node(nt, "ShaderNodeVectorMath", operation="ADD")
    nt.links.new(coord.outputs["Object"], moved.inputs[0])
    nt.links.new(shift.outputs[0], moved.inputs[1])
    mapping = node(nt, "ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1 / repeat[0], 1 / repeat[1], 1 / repeat[1])
    nt.links.new(moved.outputs[0], mapping.inputs["Vector"])
    vec = mapping.outputs["Vector"]

    hsv = node(nt, "ShaderNodeHueSaturation")
    r2 = math_node(nt, "FRACT", math_node(nt, "MULTIPLY", info.outputs["Random"], 7.31))
    nt.links.new(math_node(nt, "MULTIPLY_ADD", info.outputs["Random"], 2 * hue_jitter, 0.5 - hue_jitter),
                 hsv.inputs["Hue"])
    nt.links.new(math_node(nt, "MULTIPLY_ADD", r2, 0.3 * value, 0.85 * value), hsv.inputs["Value"])
    nt.links.new(image_node(nt, folder / f"{tex_id}_diff_2k.jpg", True, vec), hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    rough_map = image_node(nt, folder / f"{tex_id}_rough_2k.jpg", False, vec)
    nt.links.new(math_node(nt, "MULTIPLY", rough_map, rough), bsdf.inputs["Roughness"])
    nmap = node(nt, "ShaderNodeNormalMap")
    nmap.inputs["Strength"].default_value = 0.8
    nt.links.new(image_node(nt, folder / f"{tex_id}_nor_gl_2k.jpg", False, vec), nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def plain_material(name, color, roughness, **inputs):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    for k, v in inputs.items():
        bsdf.inputs[k].default_value = v
    return mat


def glaze_material():
    """Off-white stoneware: a speckled body under a glossy clear glaze."""
    mat = plain_material("mug_glaze", (0.80, 0.76, 0.68), 0.35,
                         **{"Coat Weight": 1.0, "Coat Roughness": 0.04, "Subsurface Weight": 0.05})
    nt = mat.node_tree
    speck = node(nt, "ShaderNodeTexNoise")
    speck.inputs["Scale"].default_value = 900.0
    speck.inputs["Detail"].default_value = 1.0
    ramp = node(nt, "ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.68, (0.80, 0.76, 0.68, 1)
    els[1].position, els[1].color = 0.72, (0.25, 0.20, 0.16, 1)
    nt.links.new(speck.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def ground_material():
    """Near: Poly Haven grass and forest-floor photos mixed by noise. Past ~600 m: a procedural
    forest-floor colour (that ground is mostly under trees and haze)."""
    mat = bpy.data.materials.new("ground")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    geo = node(nt, "ShaderNodeNewGeometry")
    mapping = node(nt, "ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1 / 2.5, 1 / 2.5, 1 / 2.5)
    nt.links.new(geo.outputs["Position"], mapping.inputs["Vector"])
    vec = mapping.outputs["Vector"]
    grass = ASSETS / "textures" / "sparse_grass"
    floor = ASSETS / "textures" / "forrest_ground_01"
    mixer = node(nt, "ShaderNodeTexNoise")
    mixer.inputs["Scale"].default_value = 0.04
    nt.links.new(geo.outputs["Position"], mixer.inputs["Vector"])
    fac = node(nt, "ShaderNodeMapRange")
    fac.inputs["From Min"].default_value, fac.inputs["From Max"].default_value = 0.42, 0.58
    nt.links.new(mixer.outputs["Fac"], fac.inputs["Value"])

    near_col = mix(nt, image_node(nt, grass / "sparse_grass_diff_2k.jpg", True, vec, box=False),
                   image_node(nt, floor / "forrest_ground_01_diff_2k.jpg", True, vec, box=False), fac.outputs[0])
    near_rough = mix(nt, image_node(nt, grass / "sparse_grass_rough_2k.jpg", False, vec, box=False),
                     image_node(nt, floor / "forrest_ground_01_rough_2k.jpg", False, vec, box=False),
                     fac.outputs[0], "Float")
    far_noise = node(nt, "ShaderNodeTexNoise")
    far_noise.inputs["Scale"].default_value = 0.004
    far_noise.inputs["Detail"].default_value = 6.0
    nt.links.new(geo.outputs["Position"], far_noise.inputs["Vector"])
    far_col = mix(nt, (0.018, 0.026, 0.014, 1), (0.045, 0.042, 0.026, 1), far_noise.outputs["Fac"])
    cam = node(nt, "ShaderNodeVectorMath", operation="DISTANCE")
    nt.links.new(geo.outputs["Position"], cam.inputs[0])
    cam.inputs[1].default_value = CAM_LOC
    dist = node(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    dist.inputs["From Min"].default_value, dist.inputs["From Max"].default_value = 300.0, 900.0
    nt.links.new(cam.outputs["Value"], dist.inputs["Value"])
    nt.links.new(mix(nt, near_col, far_col, dist.outputs[0]), bsdf.inputs["Base Color"])
    nt.links.new(mix(nt, near_rough, 0.9, dist.outputs[0], "Float"), bsdf.inputs["Roughness"])
    return mat


def ridge_material(name):
    """Distant forested ridges: dark conifer greens in patches, lighter where it's steep rock."""
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    geo = node(nt, "ShaderNodeNewGeometry")
    patches = node(nt, "ShaderNodeTexNoise")
    patches.inputs["Scale"].default_value = 0.0025
    patches.inputs["Detail"].default_value = 8.0
    patches.inputs["Roughness"].default_value = 0.65
    nt.links.new(geo.outputs["Position"], patches.inputs["Vector"])
    ramp = node(nt, "ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.35, (0.020, 0.032, 0.018, 1)
    els[1].position, els[1].color = 0.70, (0.070, 0.075, 0.040, 1)
    nt.links.new(patches.outputs["Fac"], ramp.inputs["Fac"])
    sep = node(nt, "ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], sep.inputs[0])
    rock = node(nt, "ShaderNodeMapRange")
    rock.inputs["From Min"].default_value, rock.inputs["From Max"].default_value = 0.85, 0.6
    nt.links.new(sep.outputs["Z"], rock.inputs["Value"])
    color = mix(nt, ramp.outputs["Color"], (0.16, 0.15, 0.13, 1), rock.outputs[0])
    nt.links.new(color, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    bump = node(nt, "ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    nt.links.new(patches.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# ---------------------------------------------------------------------------------------------
# Porch and mug.

def box(name, size, loc, rot=(0.0, 0.0, 0.0), mat=None, bevel=0.004, coll=None):
    """A board: size = (length along local X, width, height), centred at loc."""
    sx, sy, sz = (s / 2 for s in size)
    verts = [(x * sx, y * sy, z * sz) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    if mat:
        mesh.materials.append(mat)
    obj = link_object(bpy.data.objects.new(name, mesh), coll)
    obj.location = loc
    obj.rotation_euler = rot
    if bevel:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = min(bevel, min(size) / 3)
        mod.segments = 2
        mod.limit_method = "ANGLE"
    return obj


def build_porch(rng):
    coll = bpy.data.collections.new("porch")
    bpy.context.scene.collection.children.link(coll)
    deck = wood_material("deck", "wood_table_worn", value=0.95, rough=1.05)
    rail = wood_material("rail", "wood_table_worn", value=0.85)
    post = wood_material("post", "wood_table_worn", value=0.75, repeat=(1.8, 0.55))
    ceiling = wood_material("ceiling", "roof_planks", repeat=(1.5, 1.5), value=0.9, hue_jitter=0.0)
    turn = math.radians(90)

    # Deck: boards run out toward the valley; random lengths so the butt joints stagger.
    width, gap, thick = 0.14, 0.006, 0.032
    x, boards = PORCH_X[0], 0
    while x < PORCH_X[1]:
        y = DECK_Y[0]
        while y < DECK_Y[1] - 0.05:
            length = min(rng.uniform(2.4, 4.8), DECK_Y[1] - y)
            box(f"deck_{boards}", (length - 0.004, width, thick),
                (x + width / 2, y + length / 2, -thick / 2 + rng.uniform(-0.0015, 0.0015)),
                (rng.uniform(-0.004, 0.004), 0.0, turn + rng.uniform(-0.0015, 0.0015)), deck, 0.003, coll)
            y += length
            boards += 1
        x += width + gap

    # Posts, header beam, rails and balusters.
    for j, px in enumerate(POSTS_X):
        box(f"post_{j}", (POST_TOP, 0.14, 0.14), (px, 0.0, POST_TOP / 2), (0.0, turn, 0.0), post, 0.008, coll)
    left, right = PORCH_X[0], POSTS_X[-1] + 0.5
    length, mid = right - left, (left + right) / 2
    box("header", (length, 0.14, 0.22), (mid, 0.0, POST_TOP + 0.11), mat=post, bevel=0.008, coll=coll)
    box("rail_cap", (length, 0.14, 0.035), (mid, 0.0, RAIL_TOP - 0.0175), mat=rail, bevel=0.006, coll=coll)
    box("rail_top", (length, 0.04, 0.09), (mid, 0.0, RAIL_TOP - 0.08), mat=rail, coll=coll)
    box("rail_bottom", (length, 0.09, 0.045), (mid, 0.0, 0.0925), mat=rail, coll=coll)
    b_low, b_high = 0.115, RAIL_TOP - 0.125
    balusters = 0
    for a, b in pairwise(POSTS_X):
        n = int((b - a - 0.14) / 0.115)
        for s in range(1, n + 1):
            bx = a + 0.07 + s * (b - a - 0.14) / (n + 1)
            box(f"baluster_{balusters}", (b_high - b_low, 0.038, 0.038), (bx, 0.0, (b_low + b_high) / 2),
                (0.0, turn, rng.uniform(-0.05, 0.05)), rail, 0.003, coll)
            balusters += 1

    # Roof: rafters resting on the header, rising toward the cabin; ceiling boards on top.
    slope = math.atan(EAVE_SLOPE)
    y0, y1 = DECK_Y[0], 0.45
    rafter_len = (y1 - y0) / math.cos(slope)
    rafter_mid_y = (y0 + y1) / 2
    base = POST_TOP + 0.22 + 0.08
    rz = base - rafter_mid_y * EAVE_SLOPE
    rx = PORCH_X[0] + 0.1
    while rx < PORCH_X[1]:
        box(f"rafter_{rx:.2f}", (rafter_len, 0.05, 0.16), (rx, rafter_mid_y, rz), (0.0, slope, turn),
            post, 0.004, coll)
        rx += 0.61
    box("ceiling", (PORCH_X[1] - PORCH_X[0], rafter_len, 0.02),
        (sum(PORCH_X) / 2, rafter_mid_y, rz + 0.09), (-slope, 0.0, 0.0), ceiling, 0.0, coll)
    box("fascia", (PORCH_X[1] - PORCH_X[0], 0.025, 0.2),
        (sum(PORCH_X) / 2, y1 + 0.0125, base - y1 * EAVE_SLOPE), mat=post, bevel=0.004, coll=coll)

    # Cabin wall behind the camera: never seen, but it shades the porch like the real cabin.
    box("cabin_wall", (PORCH_X[1] - PORCH_X[0] + 4, 0.3, 6.0), (sum(PORCH_X) / 2, DECK_Y[0] - 0.15, 2.0),
        mat=plain_material("cabin", (0.12, 0.07, 0.04), 0.8), bevel=0.0, coll=coll)
    return boards, balusters


def build_mug():
    """A lathe-turned stoneware mug with a loop handle and coffee, on the rail cap."""
    profile = [(0.0, 0.0), (0.034, 0.0), (0.0385, 0.003), (0.0405, 0.012), (0.0425, 0.088),
               (0.0432, 0.0955), (0.0405, 0.0965), (0.0392, 0.090), (0.0372, 0.014), (0.032, 0.0085),
               (0.0, 0.0085)]
    bm = bmesh.new()
    verts = [bm.verts.new((r, 0.0, z)) for r, z in profile]
    edges = [bm.edges.new((a, b)) for a, b in pairwise(verts)]
    bmesh.ops.spin(bm, geom=verts + edges, cent=(0, 0, 0), axis=(0, 0, 1), angle=math.tau, steps=72,
                   use_merge=True)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    mesh = bpy.data.meshes.new("mug")
    bm.to_mesh(mesh)
    bm.free()
    glaze = glaze_material()
    mesh.materials.append(glaze)
    mesh.shade_smooth()
    mug = link_object(bpy.data.objects.new("mug", mesh))
    sub = mug.modifiers.new("smooth", "SUBSURF")
    sub.levels = sub.render_levels = 2

    curve = bpy.data.curves.new("mug_handle", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.0062
    curve.bevel_resolution = 6
    spline = curve.splines.new("BEZIER")
    pts = [(0.040, 0.0, 0.078), (0.068, 0.0, 0.074), (0.072, 0.0, 0.040), (0.040, 0.0, 0.024)]
    spline.bezier_points.add(len(pts) - 1)
    for bp, p in zip(spline.bezier_points, pts, strict=True):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    curve.materials.append(glaze)
    link_object(bpy.data.objects.new("mug_handle", curve)).parent = mug

    coffee = bpy.data.meshes.new("coffee")
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, radius=0.0378, segments=72)
    bm.to_mesh(coffee)
    bm.free()
    coffee.materials.append(plain_material("coffee", (0.028, 0.013, 0.005), 0.06,
                                           **{"Coat Weight": 0.5, "Coat Roughness": 0.02}))
    surface = link_object(bpy.data.objects.new("coffee", coffee))
    surface.parent = mug
    surface.location.z = 0.079

    mug.location = (MUG_AT[0], MUG_AT[1], RAIL_TOP)
    mug.rotation_euler.z = math.radians(-35)  # handle turned toward the camera's right
    return mug


# ---------------------------------------------------------------------------------------------
# Valley: terrain built only in a wedge around the view, in polar rings so detail follows distance.

def polar(r0, r1, nr, na):
    rs = r0 * (r1 / r0) ** np.linspace(0.0, 1.0, nr)
    yaws = np.radians(np.linspace(*YAW_RANGE, na))
    R, A = np.meshgrid(rs, yaws, indexing="ij")
    return R, A, CAM_LOC[0] + R * np.sin(A), CAM_LOC[1] + R * np.cos(A)


def grid_mesh(name, X, Y, Z, mat):
    nr, na = X.shape
    idx = np.arange(nr * na).reshape(nr, na)
    faces = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(np.stack([X, Y, Z], -1).reshape(-1, 3).tolist(), [], faces.tolist())
    mesh.shade_smooth()
    mesh.materials.append(mat)
    return link_object(bpy.data.objects.new(name, mesh))


def treetop_mask(X, Y):
    """1 at the centre of each treetop group, fading out over its radius; all 0 when TREETOPS is off."""
    mask = np.zeros_like(X)
    for yaw, dist, radius in TREETOP_GROUPS if TREETOPS else ():
        cx = CAM_LOC[0] + dist * math.sin(math.radians(yaw))
        cy = CAM_LOC[1] + dist * math.cos(math.radians(yaw))
        mask = np.maximum(mask, np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / radius**2))
    return mask


def ground_height(R, X, Y):
    v = np.maximum(Y, 0.0)
    # The ledge under the cabin, then the drop-off to the valley floor.
    z = LEDGE_Z - (LEDGE_Z - FLOOR_Z) * (1.0 - np.exp(-np.maximum(v - DROP_LIP, 0.0) / DROP_SCALE))
    ridge_y = 4300.0 + 500.0 * fbm_field(X, X * 0, 5000.0, 2.1, 3)
    crest = 470.0 + 170.0 * fbm_field(X, X * 0, 2600.0, 3.3, 4)
    bump = np.exp(-(((Y - ridge_y) / 1300.0) ** 2)) * smoothstep(1500.0, 3000.0, Y)
    z += bump * crest * (1.0 + 0.12 * fbm_field(X, Y, 800.0, 9.9, 5, ridged=True))
    z += 55.0 * fbm_field(X, Y, 700.0, 1.7) * smoothstep(*BUMPS_FROM, v)
    z += 8.0 * fbm_field(X, Y, 110.0, 5.1, 4) * smoothstep(60.0, 300.0, v)
    fine = 0.8 * fbm_field(X, Y, 35.0, 6.2, 3) + 0.6 * fbm_field(X, Y, 14.0, 7.3, 3)
    z += fine * smoothstep(10.0, 30.0, v)  # none at the lip, so it can't bump up into view
    z += np.maximum(TREETOP_CREST - z, 0.0) * treetop_mask(X, Y)  # knolls under the treetop groups
    # The shoulder right of the porch, just under the line of sight that grazes the deck's outer end.
    yaw = np.degrees(np.arctan2(X - CAM_LOC[0], Y - CAM_LOC[1]))
    sight = -CAM_LOC[2] / (DECK_Y[1] - CAM_LOC[1]) * (v - DECK_Y[1])
    shoulder = np.minimum(LEDGE_Z, sight - SHOULDER_SINK) - 1.5 * np.maximum(v - SHOULDER_END, 0.0)
    z = np.maximum(z, shoulder - 500.0 * (1.0 - smoothstep(*SHOULDER_YAW, yaw)))
    return z - R**2 / (2 * EARTH_R)


def build_terrain():
    R, A, X, Y = polar(2.0, 9000.0, 420, 600)
    obj = grid_mesh("terrain", X, Y, ground_height(R, X, Y), ground_material())
    mesh = obj.data
    nz = np.empty(len(mesh.vertices) * 3)
    mesh.vertex_normals.foreach_get("vector", nz)
    flat = smoothstep(0.55, 0.8, nz[2::3].reshape(R.shape))  # fewer trees on steep ground
    yaw = np.degrees(A)
    in_view = smoothstep(YAW_RANGE[0] + 6, YAW_RANGE[0] + 10, yaw) * \
        (1 - smoothstep(YAW_RANGE[1] - 10, YAW_RANGE[1] - 6, yaw))
    clumps = smoothstep(-1.3, -0.4, fbm_field(X, Y, 170.0, 11.1, 4))
    # Near band: only the pines on the shoulder right of the porch, and only where their shadow (cast away from the
    # low sun) passes right of the porch's end, so the deck keeps its light.
    shadow_x = X - Y * math.tan(math.radians(SUN_YAW))  # where a tree's shadow line crosses the rail line
    shoulder = smoothstep(0.0, 10.0, shadow_x - PORCH_X[1]) * (1 - smoothstep(SHOULDER_END - 10.0, SHOULDER_END, Y))
    near = clumps * smoothstep(20.0, 50.0, R) * (1 - smoothstep(250.0, 300.0, R)) * flat * in_view * shoulder
    tops = treetop_mask(X, Y)
    mid = smoothstep(-1.8, -0.8, fbm_field(X, Y, 520.0, 13.3, 4)) * smoothstep(900.0, 1100.0, R) * flat * in_view
    mid *= 1 - tops  # the treetop groups get their own scatter
    tops = tops**2
    for name, values in (("near", near), ("mid", mid), ("tops", tops)):
        attr = mesh.attributes.new(f"dens_{name}", "FLOAT", "POINT")
        attr.data.foreach_set("value", values.ravel().astype(np.float32))
    # Rough cell areas, for the expected tree counts in the log.
    area = R * np.gradient(R, axis=0) * np.radians((YAW_RANGE[1] - YAW_RANGE[0]) / (R.shape[1] - 1))
    expected = {k: float((v * area).sum()) for k, v in (("near", near), ("mid", mid), ("tops", tops))}
    return obj, expected


def build_far_ridges():
    for i, (dist, crest, var) in enumerate(FAR_RIDGES):
        R, A, X, Y = polar(dist - 2600.0, dist + 2200.0, 120, 700)
        arc = dist * A
        top = crest + var * fbm_field(arc, arc * 0 + i * 1000, 4000.0, 20.0 + i, 3)
        t = (R - dist) / np.where(R < dist, 1800.0, 1300.0)
        profile = np.exp(-1.2 * t * t)
        rise = (top - FLOOR_Z) * profile
        Z = FLOOR_Z + rise * (1.0 + 0.04 * fbm_field(X, Y, 900.0, 30.0 + i, 4, ridged=True))
        Z += 25.0 * fbm_field(X, Y, 300.0, 40.0 + i, 4) - R**2 / (2 * EARTH_R)
        grid_mesh(f"ridge_{i + 1}", X, Y, Z, ridge_material(f"ridge_{i + 1}"))


# ---------------------------------------------------------------------------------------------
# Trees: Poly Haven conifers scattered with Geometry Nodes. Each copy gets its own variant,
# spin, tilt, scale and a "tint" value that the needle and bark materials read.

def import_variants(assets, coll):
    objs = []
    for asset in assets:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(fetch_model(asset)))
        for o in set(bpy.data.objects) - before:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            if o.type != "MESH":
                bpy.data.objects.remove(o)
                continue
            coll.objects.link(o)
            o.location = (0.0, 0.0, 0.0)
            objs.append(o)
    return sorted(objs, key=lambda o: o.name)


def add_tint(materials):
    for mat in materials:
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        link = bsdf and next((lk for lk in nt.links if lk.to_socket == bsdf.inputs["Base Color"]), None)
        if link is None:
            continue
        tint = node(nt, "ShaderNodeAttribute", attribute_type="INSTANCER", attribute_name="tint").outputs["Fac"]
        hsv = node(nt, "ShaderNodeHueSaturation")
        if "twig" in mat.name or "grass" in mat.name:  # needles, grass: hue, saturation, brightness
            nt.links.new(math_node(nt, "MULTIPLY_ADD", tint, 0.06, 0.47), hsv.inputs["Hue"])
            nt.links.new(math_node(nt, "MULTIPLY_ADD", tint, 0.30, 0.85), hsv.inputs["Saturation"])
            nt.links.new(math_node(nt, "MULTIPLY_ADD", tint, 0.45, 0.75), hsv.inputs["Value"])
        else:  # bark and dead branches: brightness only
            nt.links.new(math_node(nt, "MULTIPLY_ADD", tint, 0.30, 0.85), hsv.inputs["Value"])
        nt.links.new(link.from_socket, hsv.inputs["Color"])
        nt.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])


def random_value(ng, data_type, lo, hi, seed):
    n = node(ng, "FunctionNodeRandomValue", data_type=data_type)
    n.inputs["Min"].default_value = lo
    n.inputs["Max"].default_value = hi
    n.inputs["Seed"].default_value = seed
    return n.outputs["Value"]


def scatter_modifier(target, name, coll, attr, density, seed, scale, tilt_deg, min_dist=None):
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin, gout = node(ng, "NodeGroupInput"), node(ng, "NodeGroupOutput")
    dens = node(ng, "GeometryNodeInputNamedAttribute", data_type="FLOAT")
    dens.inputs["Name"].default_value = attr
    dist = node(ng, "GeometryNodeDistributePointsOnFaces", distribute_method="POISSON" if min_dist else "RANDOM")
    ng.links.new(gin.outputs[0], dist.inputs["Mesh"])
    if min_dist:
        dist.inputs["Distance Min"].default_value = min_dist
        dist.inputs["Density Max"].default_value = density
        ng.links.new(dens.outputs["Attribute"], dist.inputs["Density Factor"])
    else:
        ng.links.new(math_node(ng, "MULTIPLY", dens.outputs["Attribute"], density), dist.inputs["Density"])
    dist.inputs["Seed"].default_value = seed

    info = node(ng, "GeometryNodeCollectionInfo", transform_space="ORIGINAL")
    info.inputs["Collection"].default_value = coll
    info.inputs["Separate Children"].default_value = True
    info.inputs["Reset Children"].default_value = True
    tilt = math.radians(tilt_deg)
    euler = node(ng, "FunctionNodeEulerToRotation")
    ng.links.new(random_value(ng, "FLOAT_VECTOR", (-tilt, -tilt, 0.0), (tilt, tilt, math.tau), seed + 2),
                 euler.inputs[0])
    inst = node(ng, "GeometryNodeInstanceOnPoints")
    ng.links.new(dist.outputs["Points"], inst.inputs["Points"])
    ng.links.new(info.outputs[0], inst.inputs["Instance"])
    inst.inputs["Pick Instance"].default_value = True
    ng.links.new(random_value(ng, "INT", 0, len(coll.objects) - 1, seed + 1), inst.inputs["Instance Index"])
    ng.links.new(euler.outputs[0], inst.inputs["Rotation"])
    ng.links.new(random_value(ng, "FLOAT", scale[0], scale[1], seed + 3), inst.inputs["Scale"])
    store = node(ng, "GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="INSTANCE")
    store.inputs["Name"].default_value = "tint"
    ng.links.new(inst.outputs[0], store.inputs["Geometry"])
    ng.links.new(random_value(ng, "FLOAT", 0.0, 1.0, seed + 4), store.inputs["Value"])
    join = node(ng, "GeometryNodeJoinGeometry")
    ng.links.new(store.outputs[0], join.inputs[0])
    ng.links.new(gin.outputs[0], join.inputs[0])
    ng.links.new(join.outputs[0], gout.inputs[0])
    target.modifiers.new(name, "NODES").node_group = ng


def build_trees(terrain, log):
    scene = bpy.context.scene
    colls = {}
    for key in ("trees_near", "trees_mid"):
        colls[key] = bpy.data.collections.new(key)
        scene.collection.children.link(colls[key])
    t = time.time()
    near = import_variants(TREE_ASSETS, colls["trees_near"])
    log(f"Trees: imported {len(near)} tree variants in {time.time() - t:.0f} s")

    # Mid band (1-9 km) and the treetop groups: decimated copies.
    t = time.time()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for o in near:
        mod = o.modifiers.new("decimate", "DECIMATE")
        mod.ratio = min(1.0, MID_TREE_POLYS / len(o.data.polygons))
        depsgraph.update()
        mesh = bpy.data.meshes.new_from_object(o.evaluated_get(depsgraph))
        o.modifiers.remove(mod)
        colls["trees_mid"].objects.link(bpy.data.objects.new(o.name + "_mid", mesh))
    log(f"Trees: decimated {len(colls['trees_mid'].objects)} mid-band copies in {time.time() - t:.0f} s")

    add_tint({m for o in near for m in o.data.materials if m})
    for key in colls:  # sources only feed the scatter; hide them from the render itself
        scene.view_layers[0].layer_collection.children[key].exclude = True
    scatter_modifier(terrain, "scatter_near", colls["trees_near"], "dens_near", NEAR_DENSITY, 11, NEAR_SCALE, 4.0, 3.5)
    scatter_modifier(terrain, "scatter_mid", colls["trees_mid"], "dens_mid", MID_DENSITY, 31, (0.85, 1.45), 3.0)
    if TREETOPS:
        scatter_modifier(terrain, "scatter_tops", colls["trees_mid"], "dens_tops", TREETOP_DENSITY, 51, TREETOP_SCALE,
                         3.0, 4.0)


# ---------------------------------------------------------------------------------------------
# Mist, haze, sky, sun and camera.

def build_atmosphere():
    """One volume box over the whole valley. Density = haze that thins with height (fades the
    far ridges; an air part and a forward-scattering dust part) + low mist banks lying on the valley
    floor. It never fills the sky, so the sun still reaches the porch."""
    lo, hi = (-30000.0, -8000.0, FLOOR_Z - 80.0), (30000.0, 32000.0, 2600.0)
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    centre = tuple((a + b) / 2 for a, b in zip(lo, hi, strict=True))
    mat = bpy.data.materials.new("atmosphere")
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    out = nt.nodes["Material Output"]
    geo = node(nt, "ShaderNodeNewGeometry")
    sep = node(nt, "ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    z = sep.outputs["Z"]
    haze = math_node(nt, "MULTIPLY", math_node(nt, "EXPONENT", math_node(
        nt, "MULTIPLY", math_node(nt, "SUBTRACT", z, FLOOR_Z), -1.0 / HAZE_HEIGHT)), HAZE_FLOOR_DENSITY)
    above = math_node(nt, "MAXIMUM", math_node(nt, "SUBTRACT", z, MIST_TOP), 0.0)
    mist_height = math_node(nt, "EXPONENT", math_node(nt, "MULTIPLY", above, -1.0 / MIST_FADE))
    stretch = node(nt, "ShaderNodeMapping")
    stretch.inputs["Scale"].default_value = (1 / 900.0, 1 / 900.0, 1 / 90.0)
    nt.links.new(geo.outputs["Position"], stretch.inputs["Vector"])
    banks = node(nt, "ShaderNodeTexNoise")
    banks.inputs["Scale"].default_value = 1.0
    banks.inputs["Detail"].default_value = 4.0
    banks.inputs["Roughness"].default_value = 0.55
    nt.links.new(stretch.outputs["Vector"], banks.inputs["Vector"])
    holes = node(nt, "ShaderNodeMapRange")
    holes.inputs["From Min"].default_value, holes.inputs["From Max"].default_value = 0.38, 0.62
    nt.links.new(banks.outputs["Fac"], holes.inputs["Value"])
    mist = math_node(nt, "MULTIPLY", math_node(nt, "MULTIPLY", mist_height, holes.outputs[0]), MIST_DENSITY)

    air = node(nt, "ShaderNodeVolumeScatter", phase="RAYLEIGH")
    air.inputs["Color"].default_value = HAZE_AIR_COLOR
    nt.links.new(math_node(nt, "MULTIPLY", haze, HAZE_AIR_SHARE), air.inputs["Density"])
    dust = node(nt, "ShaderNodeVolumeScatter", phase="HENYEY_GREENSTEIN")
    dust.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1)
    dust.inputs["Anisotropy"].default_value = HAZE_DUST_ANISOTROPY
    nt.links.new(math_node(nt, "MULTIPLY", haze, HAZE_DUST_SHARE), dust.inputs["Density"])
    mist_vol = node(nt, "ShaderNodeVolumeScatter", phase="MIE")
    mist_vol.inputs["Color"].default_value = (0.99, 0.99, 0.99, 1)
    mist_vol.inputs["Diameter"].default_value = MIST_DROPLET
    nt.links.new(mist, mist_vol.inputs["Density"])
    haze_vol = node(nt, "ShaderNodeAddShader")
    nt.links.new(air.outputs[0], haze_vol.inputs[0])
    nt.links.new(dust.outputs[0], haze_vol.inputs[1])
    add = node(nt, "ShaderNodeAddShader")
    nt.links.new(haze_vol.outputs[0], add.inputs[0])
    nt.links.new(mist_vol.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Volume"])
    obj = box("atmosphere", size, centre, mat=mat, bevel=0.0)
    obj.visible_shadow = True
    return obj


def build_clouds():
    """Thin high clouds: a volume box whose density is noise inside a layer that follows the earth's
    curvature, so with distance it drops toward the horizon the way a real cloud deck does."""
    top = CLOUD_BASE + CLOUD_THICKNESS
    lo, hi = (-5000.0, 0.0, CLOUD_BASE - 650.0), (95000.0, 95000.0, top + 50.0)
    mat = bpy.data.materials.new("clouds")
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    sep = node(nt, "ShaderNodeSeparateXYZ")
    nt.links.new(node(nt, "ShaderNodeNewGeometry").outputs["Position"], sep.inputs[0])
    dx = math_node(nt, "SUBTRACT", sep.outputs["X"], CAM_LOC[0])
    dy = math_node(nt, "SUBTRACT", sep.outputs["Y"], CAM_LOC[1])
    d2 = math_node(nt, "ADD", math_node(nt, "MULTIPLY", dx, dx), math_node(nt, "MULTIPLY", dy, dy))
    h = math_node(nt, "MULTIPLY_ADD", d2, 1 / (2 * EARTH_R), sep.outputs["Z"])  # height above the curved earth
    w = math.radians(CLOUD_WIND_YAW)
    along = math_node(nt, "ADD", math_node(nt, "MULTIPLY", dx, math.sin(w)), math_node(nt, "MULTIPLY", dy, math.cos(w)))
    across = math_node(nt, "SUBTRACT", math_node(nt, "MULTIPLY", dx, math.cos(w)),
                       math_node(nt, "MULTIPLY", dy, math.sin(w)))

    def noise_at(scale, detail, z):
        vec = node(nt, "ShaderNodeCombineXYZ")
        nt.links.new(math_node(nt, "DIVIDE", along, scale[0]), vec.inputs[0])
        nt.links.new(math_node(nt, "DIVIDE", across, scale[1]), vec.inputs[1])
        if isinstance(z, bpy.types.NodeSocket):
            nt.links.new(math_node(nt, "DIVIDE", z, scale[2]), vec.inputs[2])
        else:
            vec.inputs[2].default_value = z
        tex = node(nt, "ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 1.0
        tex.inputs["Detail"].default_value = detail
        tex.inputs["Roughness"].default_value = 0.6
        nt.links.new(vec.outputs[0], tex.inputs["Vector"])
        return tex.outputs["Fac"]

    fine = noise_at(CLOUD_SIZE, 7.0, h)
    clusters = noise_at((CLOUD_SIZE[0] * 5, CLOUD_SIZE[1] * 9, 1.0), 2.0, 0.37)  # clumps and clear gaps
    field = math_node(nt, "MULTIPLY_ADD", math_node(nt, "SUBTRACT", clusters, 0.5), 1.1, fine)

    def ramp(value, a, b):
        m = node(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
        m.inputs["From Min"].default_value, m.inputs["From Max"].default_value = a, b
        nt.links.new(value, m.inputs["Value"])
        return m.outputs[0]

    shape = ramp(field, CLOUD_COVER, CLOUD_COVER + 0.14)
    rise = ramp(h, CLOUD_BASE, CLOUD_BASE + 0.35 * CLOUD_THICKNESS)
    fall = ramp(h, top, CLOUD_BASE + 0.45 * CLOUD_THICKNESS)  # 1 inside the layer, 0 above its top
    fade = ramp(math_node(nt, "SQRT", d2), CLOUD_FADE[1], CLOUD_FADE[0])
    density = math_node(nt, "MULTIPLY", math_node(nt, "MULTIPLY", shape, math_node(nt, "MULTIPLY", rise, fall)),
                        math_node(nt, "MULTIPLY", fade, CLOUD_DENSITY))
    vol = node(nt, "ShaderNodeVolumeScatter", phase="HENYEY_GREENSTEIN")
    vol.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1)
    vol.inputs["Anisotropy"].default_value = CLOUD_ANISOTROPY
    nt.links.new(density, vol.inputs["Density"])
    nt.links.new(vol.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    centre = tuple((a + b) / 2 for a, b in zip(lo, hi, strict=True))
    return box("clouds", size, centre, mat=mat, bevel=0.0)


def sky_node(nt, disc):
    sky = node(nt, "ShaderNodeTexSky", sky_type="MULTIPLE_SCATTERING")
    sky.sun_elevation = math.radians(SUN_ELEVATION)
    sky.sun_rotation = math.radians(SUN_YAW)  # measured from +Y toward +X (checked in a probe)
    sky.sun_disc = disc
    sky.sun_size = math.radians(SUN_ANGLE)
    sky.air_density, sky.aerosol_density, sky.ozone_density = SKY_AIR, SKY_AEROSOL, SKY_OZONE
    sky.altitude = SKY_ALTITUDE
    return sky


def build_world(scene):
    """Two copies of the same sky: the camera sees the one with the sun disc; light bounces see the
    one without it, because the Sun lamp below carries the sun's light and crisp shadows."""
    world = bpy.data.worlds.new("sky")
    nt = world.node_tree
    lit = nt.nodes["Background"]
    seen = node(nt, "ShaderNodeBackground")
    nt.links.new(sky_node(nt, False).outputs["Color"], lit.inputs["Color"])
    nt.links.new(sky_node(nt, True).outputs["Color"], seen.inputs["Color"])
    pick = node(nt, "ShaderNodeMixShader")
    nt.links.new(node(nt, "ShaderNodeLightPath").outputs["Is Camera Ray"], pick.inputs[0])
    nt.links.new(lit.outputs[0], pick.inputs[1])
    nt.links.new(seen.outputs[0], pick.inputs[2])
    nt.links.new(pick.outputs[0], nt.nodes["World Output"].inputs["Surface"])
    scene.world = world


def sun_direction():
    el, az = math.radians(SUN_ELEVATION), math.radians(SUN_YAW)
    return Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))


def build_sun():
    light = bpy.data.lights.new("sun", "SUN")
    light.energy = SUN_STRENGTH
    light.color = SUN_COLOR
    light.angle = math.radians(SUN_ANGLE)
    obj = link_object(bpy.data.objects.new("sun", light))
    obj.rotation_euler = sun_direction().to_track_quat("Z", "Y").to_euler()


def build_camera(scene):
    cam = bpy.data.cameras.new("camera")
    cam.lens = CAM_LENS
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = 36.0
    cam.shift_y = CAM_SHIFT_Y
    cam.clip_start, cam.clip_end = 0.05, 60000.0
    cam.dof.use_dof = True
    cam.dof.focus_distance = CAM_FOCUS
    cam.dof.aperture_fstop = CAM_FSTOP
    obj = link_object(bpy.data.objects.new("camera", cam))
    obj.location = CAM_LOC
    obj.rotation_euler = (math.radians(90 + CAM_PITCH), 0.0, math.radians(-CAM_YAW))
    scene.camera = obj


# ---------------------------------------------------------------------------------------------
# Render, outputs and the review checks.

def setup_render(scene, preview, samples):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.refresh_devices()
    gpus = [d for d in prefs.devices if d.type == "OPTIX"]
    if not gpus:
        raise SystemExit("No OptiX GPU found; refusing to render on the CPU.")
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    scene.render.engine = "CYCLES"
    c = scene.cycles
    c.device = "GPU"
    c.samples = samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.02
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.denoising_use_gpu = True
    c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    c.denoising_prefilter = "ACCURATE"
    c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 8, 3, 3
    c.transmission_bounces, c.volume_bounces, c.transparent_max_bounces = 4, 1, 32
    c.sample_clamp_indirect = 8.0
    c.caustics_reflective = c.caustics_refractive = False
    r = scene.render
    r.resolution_x, r.resolution_y = FINAL_RES
    r.resolution_percentage = PREVIEW_PCT if preview else 100
    v = scene.view_settings
    v.view_transform = VIEW_TRANSFORM
    v.look = LOOK
    v.exposure = EXPOSURE
    v.use_white_balance = True
    v.white_balance_temperature = WB_TEMPERATURE
    v.white_balance_tint = WB_TINT
    return ", ".join(d.name for d in gpus)


def build_compositor(scene):
    """Render -> lens glow -> saturation lift on the bright parts -> output. Exposure, white balance
    and the AgX view are applied after this, as for any render."""
    ng = bpy.data.node_groups.new("compositor", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    layers = node(ng, "CompositorNodeRLayers")
    layers.scene = scene
    glare = node(ng, "CompositorNodeGlare")
    for name, value in (("Type", BLOOM_TYPE), ("Quality", "High"), ("Highlights Threshold", BLOOM_THRESHOLD),
                        ("Clamp Highlights", True), ("Maximum Highlights", BLOOM_MAX),
                        ("Strength", BLOOM_STRENGTH), ("Size", BLOOM_SIZE)):
        glare.inputs[name].default_value = value
    ng.links.new(layers.outputs["Image"], glare.inputs["Image"])
    grade = node(ng, "CompositorNodeHueSat")
    grade.inputs["Saturation"].default_value = GRADE_SATURATION
    ng.links.new(glare.outputs["Image"], grade.inputs["Image"])
    brightness = node(ng, "CompositorNodeRGBToBW")
    ng.links.new(glare.outputs["Image"], brightness.inputs["Image"])
    mask = node(ng, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    mask.inputs["From Min"].default_value, mask.inputs["From Max"].default_value = GRADE_MASK
    ng.links.new(brightness.outputs["Val"], mask.inputs["Value"])
    ng.links.new(mask.outputs["Result"], grade.inputs["Fac"])
    ng.links.new(grade.outputs["Image"], node(ng, "NodeGroupOutput").inputs[0])
    scene.compositing_node_group = ng
    scene.render.use_compositing = True


class _MemoryCounters(ctypes.Structure):  # Windows' PROCESS_MEMORY_COUNTERS
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                             "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                             "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def peak_ram_gb():
    """The most RAM this Blender process has used so far (Windows' peak working set), in GB."""
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MemoryCounters), ctypes.c_ulong]
    counters = _MemoryCounters(cb=ctypes.sizeof(_MemoryCounters))
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 2**30


def read_pixels(path):
    im = bpy.data.images.load(str(path), check_existing=False)
    w, h = im.size
    px = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(px)
    return im, px.reshape(h, w, 4)  # row 0 is the bottom row


def write_pixels(arr, path, quality=92):
    h, w = arr.shape[:2]
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = arr[..., :3] if arr.ndim == 3 else arr[..., None]
    im = bpy.data.images.new(Path(path).stem, w, h)
    im.pixels.foreach_set(rgba.ravel())
    im.file_format = "WEBP"
    im.save(filepath=str(path), quality=quality)


def scaled(path, height):
    im, _ = read_pixels(path)
    w, h = im.size
    im.scale(round(w * height / h), height)
    px = np.empty(im.size[0] * im.size[1] * 4, np.float32)
    im.pixels.foreach_get(px)
    return px.reshape(height, im.size[0], 4)


def save_final(scene, log):
    img = bpy.data.images["Render Result"]
    s = scene.render.image_settings
    png, webp = OUT / f"{TAG}-porch-{ROUND}.png", OUT / f"{TAG}-porch-{ROUND}.webp"
    s.file_format, s.color_mode, s.color_depth, s.compression = "PNG", "RGB", "16", 15
    img.save_render(str(png), scene=scene)
    s.file_format, s.color_mode, s.quality = "WEBP", "RGB", 95
    img.save_render(str(webp), scene=scene)

    checks = OUT / "checks"
    checks.mkdir(exist_ok=True)
    v = scene.view_settings
    v.view_transform, v.look = "False Color", "None"
    img.save_render(str(checks / f"{TAG}-falsecolor-{ROUND}.webp"), scene=scene)
    v.view_transform, v.look = VIEW_TRANSFORM, LOOK

    _, still = read_pixels(webp)
    grey = still[..., 0] * 0.2126 + still[..., 1] * 0.7152 + still[..., 2] * 0.0722
    write_pixels(grey, checks / f"{TAG}-grey-{ROUND}.webp")
    clipped = float((still[..., :3].max(-1) >= 0.995).mean() * 100)
    crushed = float((still[..., :3].max(-1) <= 0.005).mean() * 100)
    log(f"Clipping: {clipped:.3f}% of pixels at full white, {crushed:.3f}% at full black")

    # Four random 360 px crops of the forest, two from each zone, side by side.
    h, w = still.shape[:2]
    seed = int(time.time())
    rng = random.Random(seed)
    size = 360
    crops = []
    for zone in FOREST_ZONES:
        for _ in range(2):
            x0 = rng.randrange(int(zone[0] * w), int(zone[1] * w) - size)
            top = rng.randrange(int(zone[2] * h), int(zone[3] * h) - size)
            crops.append(still[h - top - size:h - top, x0:x0 + size])
            log(f"Forest crop at x={x0}, y={top} (from top-left)")
    write_pixels(np.concatenate(crops, axis=1), checks / f"{TAG}-forest-crops-{ROUND}.webp")

    # Compare sheet: the previous round, this round, then the two references, all 720 px high.
    height, gap = 720, 24
    parts = [scaled(PREVIOUS, height), scaled(webp, height),
             scaled(REF_DIR / "ref-01-porch-misty-valley.png", height),
             scaled(REF_DIR / "ref-02-porch-sun-through-pines.png", height)]
    sheet = np.full((height + 2 * gap, sum(p.shape[1] for p in parts) + gap * (len(parts) + 1), 4), 0.06,
                    np.float32)
    x = gap
    for p in parts:
        sheet[gap:gap + height, x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    compare = OUT / f"{TAG}-compare-{ROUND}.webp"
    write_pixels(sheet, compare, quality=94)
    log(f"Wrote {png.name}, {webp.name}, {compare.name} and checks/ (crop seed {seed})")


# Where the forest is in the frame (x0, x1, y0, y1 as fractions from top-left): the pines right of the porch, and the
# treetop groups in the lower left.
FOREST_ZONES = ((0.66, 0.99, 0.12, 0.56), (0.08, 0.45, 0.52, 0.78))


def parse_args():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="05_porch_still.py")
    p.add_argument("--preview", metavar="NAME", help="quick 1280x720 render into preview/NAME.webp")
    p.add_argument("--samples", type=int, help="override the sample count")
    p.add_argument("--no-trees", action="store_true", help="skip trees (fast composition tests)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    started = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    build_world(scene)
    build_sun()
    build_camera(scene)
    boards, balusters = build_porch(random.Random(SEED))
    build_mug()
    t = time.time()
    terrain, expected = build_terrain()
    build_far_ridges()
    log(f"Terrain built in {time.time() - t:.0f} s")
    if not args.no_trees:
        build_trees(terrain, log)
        log("Trees expected: " + ", ".join(f"{k} ~{v * d:,.0f}" for (k, v), d in
                                           zip(expected.items(), (NEAR_DENSITY, MID_DENSITY, TREETOP_DENSITY),
                                               strict=True)))
    build_atmosphere()
    build_clouds()

    preview = args.preview is not None
    samples = args.samples or (PREVIEW_SAMPLES if preview else FINAL_SAMPLES)
    devices = setup_render(scene, preview, samples)
    build_compositor(scene)
    v = scene.view_settings
    r = scene.render
    log(f"View transform: {v.view_transform} | look: {v.look} | exposure: {v.exposure:+.2f} EV | "
        f"white balance: {v.white_balance_temperature:.0f} K, tint {v.white_balance_tint:g}")
    log(f"Sky: air {SKY_AIR:g}, aerosol {SKY_AEROSOL:g}, ozone {SKY_OZONE:g}, altitude {SKY_ALTITUDE:g} m; "
        f"sun lamp {SUN_STRENGTH:g} x {tuple(SUN_COLOR)}")
    log(f"Glare: {BLOOM_TYPE}, threshold {BLOOM_THRESHOLD:g}, clamp {BLOOM_MAX:g}, strength {BLOOM_STRENGTH:g}, "
        f"size {BLOOM_SIZE:g}; grade: saturation {GRADE_SATURATION:g} above brightness {GRADE_MASK}")
    log(f"Haze: dust share {HAZE_DUST_SHARE:g}, anisotropy {HAZE_DUST_ANISOTROPY:g}; clouds: base {CLOUD_BASE:g} m, "
        f"density {CLOUD_DENSITY:g}, cover {CLOUD_COVER:g}, anisotropy {CLOUD_ANISOTROPY:g}")
    log(f"Render: {r.resolution_x * r.resolution_percentage // 100}x{r.resolution_y * r.resolution_percentage // 100}"
        f", {samples} samples, denoiser {scene.cycles.denoiser}, on {devices}")
    log(f"Porch: {boards} deck boards, {balusters} balusters; camera {CAM_LENS:.0f} mm f/{CAM_FSTOP:g}, "
        f"focus {CAM_FOCUS} m; sun {SUN_ELEVATION} deg up, {SUN_YAW} deg yaw")
    log(f"Ledge: drop from {DROP_LIP:g} m past the rail, e-folding {DROP_SCALE:g} m; shoulder {SHOULDER_SINK:g} m "
        f"under the deck-edge sightline to {SHOULDER_END:g} m; valley bumps from {BUMPS_FROM} m; treetop groups: "
        f"{len(TREETOP_GROUPS) if TREETOPS else 'off'}")

    stats = {"peak_mb": 0.0, "last": ""}

    def on_stats(*args):  # Cycles reports its current memory; keep the highest value seen
        text = " ".join(map(str, args))
        stats["last"] = text
        for amount, unit in re.findall(r"(?:Mem|Peak):\s*([\d.]+)([KMG])", text):
            stats["peak_mb"] = max(stats["peak_mb"], float(amount) * {"K": 1 / 1024, "M": 1, "G": 1024}[unit])

    bpy.app.handlers.render_stats.append(on_stats)
    log(f"Scene built in {time.time() - started:.0f} s (peak RAM so far {peak_ram_gb():.1f} GB); rendering ...")
    t = time.time()
    bpy.ops.render.render()
    log(f"Render time: {time.time() - t:.1f} s")
    log(f"Cycles peak memory (highest reported during the render): {stats['peak_mb']:,.0f} MB; "
        f"peak RAM over the whole run: {peak_ram_gb():.1f} GB")

    OUT.mkdir(parents=True, exist_ok=True)
    if preview:
        folder = OUT / "preview"
        folder.mkdir(exist_ok=True)
        s = scene.render.image_settings
        s.file_format, s.color_mode, s.quality = "WEBP", "RGB", 90
        bpy.data.images["Render Result"].save_render(str(folder / f"{args.preview}.webp"), scene=scene)
        log(f"Wrote preview/{args.preview}.webp")
        log_path = folder / "preview-log.txt"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"--- {args.preview} {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lines) + "\n")
    else:
        save_final(scene, log)
        log(f"Total run time: {time.time() - started:.0f} s")
        (OUT / f"{TAG}-render-log.txt").write_text(
            f"Porch still {ROUND}, {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
