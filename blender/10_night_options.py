r"""Handoff 10: night look options. An aurora over the far ridges, a dimmer porch light from a fixture in view (three
placements) and a darker night glow, as stills for Nems to pick from.

The scene is 08's: the signed-off porch built with blender/05_porch_still.py's own builders, the keyframes, the stars
and the night glow (blender/08_time_keyframes.py, loaded as a module, which loads 05 the same way; both stay
unchanged). On top of it this adds:
- an aurora in the sky shader: curtains of light from about 100 km up, several hundred km away. For every direction
  the camera or a light bounce looks in, it works out where that line of sight passes each curtain, so the aurora sits
  behind the ridges, lets the stars through and lights the scene a little;
- three porch light fixtures built from simple shapes, one in view at a time: (a) a lantern sconce on the left post,
  (b) a bare bulb hanging from the porch ceiling, (c) a candle lantern on the rail. Its light replaces 08's lantern;
- the night balance: 08's night glow turned down, now that the aurora lights the night.

Run headless from the repo root (Blender isn't on PATH, so call it by full path):
    & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender\10_night_options.py
    ... -P blender\10_night_options.py -- --stills fixture-b-k5,fixture-b-k1      # some of the stills
    ... -P blender\10_night_options.py -- --preview NAME [--stills ...] [--no-trees] [--samples N]

The full run builds the scene once and renders the stills at 2560x1440 (about 25 minutes) into renders/review/10/:
    aurora-subtle-k5, aurora-vivid-k5, aurora-subtle-k4 (.webp)        aurora strengths, with the sconce (a)
    fixture-a-k5, fixture-b-k5, fixture-c-k5                           each fixture lit, subtle aurora, night
    fixture-a-k1, fixture-b-k1, fixture-c-k1                           each fixture unlit at golden hour
    10-night-options.webp   labelled contact sheet of the stills on disk, rebuilt after every run
    10-render-log.txt       one section per still, replaced when that still is rendered again
fixture-a-k5 has the same settings as aurora-subtle-k5, so it's rendered once and saved under both names. Previews
(1280x720, labelled) go to renders/review/10/preview/. Nothing is written outside renders/review/10/.
"""

import argparse
import importlib.util
import math
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "renders" / "review" / "10"
LOG_PATH = OUT / "10-render-log.txt"

# 08's keyframes and scene. Its main() doesn't run. Loading it also loads 05 and switches off 05's downloader, so this
# run never touches the network.
sys.dont_write_bytecode = True  # no __pycache__ next to the scripts
_spec = importlib.util.spec_from_file_location("time_keyframes", Path(__file__).with_name("08_time_keyframes.py"))
kf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kf)
porch = kf.porch
kf.OUT = porch.OUT = OUT  # 07's and 08's folders are never written from here


# ---------------------------------------------------------------------------------------------
# Night balance.

# 08's night glow is several stops brighter than a real moonless sky, so the night stayed readable around a 9 W lantern.
# The aurora now lights the night sky and the porch light is dimmer, so the glow comes down to this share of 08's.
NIGHT_GLOW = 0.5
kf.AIRGLOW = tuple(c * NIGHT_GLOW for c in kf.AIRGLOW)  # 08 reads it each time it builds a sky


# ---------------------------------------------------------------------------------------------
# Aurora.

@dataclass(frozen=True)
class Look:
    name: str
    brightness: float  # radiance of a curtain's bright hem, seen face-on, at full strength
    green: tuple  # oxygen's green (557.7 nm), linear
    top: tuple  # the fringe at the top: oxygen's red (630 nm) with some of nitrogen's blue-violet
    top_share: float  # how bright the fringe is next to the green
    glow: float  # radiance of the faint green glow the curtains light in the air below them, at the horizon


# The same aurora at two strengths.
LOOKS = {look.name: look for look in (
    # What the eye sees: a pale grey-green, the red at the top barely there.
    Look("subtle", 0.011, (0.55, 1.0, 0.58), (0.62, 0.45, 0.80), 0.25, 0.0005),
    # What a long exposure records: brighter, saturated green, magenta-red tops.
    Look("vivid", 0.032, (0.12, 1.0, 0.28), (1.0, 0.12, 0.60), 0.6, 0.0015),
)}

# The aurora fades in between these sun heights (degrees): from when the porch light comes on (09's frame 40) to full
# strength at blue hour (k4). It keeps that strength, and the darkening sky reveals it: at blue hour the sky near the
# ridges is about 40 times brighter than at night, so even the vivid aurora doesn't show yet. It first shows around sun
# -8.5 degrees (09's frame 53) and is clearest at night.
AURORA_FADE = (-3.5, -5.0)
# Moves the waves, folds and rays smoothly (every part is a smooth function of it): a later step can step it per frame.
AURORA_PHASE = 0.0


@dataclass(frozen=True)
class Curtain:
    distance: float  # km from the camera to the sheet's nearest point; it's straight, so it's seen at an angle elsewhere
    bearing: float  # yaw of that nearest point, degrees from +Y toward +X like SUN_YAW
    base: float  # height of its lower edge, km
    brightness: float
    fringe: float  # how much of the red-purple top shows (the far band's would sit just under the main arc)
    seed: float


# The camera sees the sky from about 3.5 degrees up (the ridges) to 13-15 degrees (the top of the frame), between yaw 17
# and 59 degrees. A lower edge 100 km up sits at 7.7 degrees from 560 km away and at 3.3 degrees from 820 km.
CURTAINS = (
    Curtain(560, 32, 100, 1.0, 1.0, 1.3),  # the main arc
    Curtain(820, 50, 100, 0.45, 0.15, 4.7),  # farther and lower: a band just above the ridges
    Curtain(400, 10, 110, 0.3, 1.0, 8.9),  # nearer: a faint sheet crossing the top of the sky
)
EARTH_R_KM = porch.EARTH_R / 1000
WAVES = ((0.16, 24.0), (0.035, 5.0))  # (share of the distance, degrees of yaw per wave): large waves, then small folds
PATCH_SIZE = 14.0  # degrees of yaw: some stretches of a sheet are brighter than others
FOLD_MAX = 3.0  # a fold seen edge-on looks brighter (a longer path through the sheet), up to this much
AURORA_EDGE = 9.0  # km: the softness of the lower edge
AURORA_RAGGED = 1.5  # km: how far the rays move the lower edge up and down
UNDER_GLOW = (0.12, 30.0)  # a faint glow under the lower edge (share of the hem, km it fades over): no hard dark gap
HEM_HEIGHT = 16.0  # km: the bright band just above the lower edge fades upward over this
RAY_HEIGHT = (25.0, 85.0)  # km: how far up a ray reaches, from the shortest to the longest
RAY_SIZE = 0.3  # degrees of yaw per ray
RAY_PATCH = 3.5  # degrees of yaw: rays bunch together in some stretches and the sheet is smooth in others
AURORA_TOP = (40.0, 90.0, 35.0)  # km above the lower edge: the fringe fades in between the first two, then out
AURORA_EXTINCTION = 0.04  # optical depth of the air straight up: the aurora dims a little toward the horizon
RAY_BEND_SIZE = 2.5  # degrees: with add_aurora's bend, the stretch over which rays crowd together or spread apart


def aurora_strength(sun):
    return float(porch.smoothstep(*AURORA_FADE, sun))


def add_aurora(world, look, strength, phase=AURORA_PHASE, lean_toward=None, bend=0.0, drift=0.0):
    """Adds the aurora to both of the world's sky colours (the one the camera sees and the one light bounces see), so it
    sits behind the ridges with the stars on top and lights the scene a little. Each curtain is a straight sheet of
    light with large waves and small folds. For every direction this works out the height at which that line of sight
    passes the sheet (with the earth's curve) and how bright the sheet is there: a soft, ragged lower edge, a bright hem
    fading upward, rays of different lengths above it and a red-purple fringe at the top. Where a fold turns the sheet
    edge-on to the camera, the line of sight runs through more of it, so it looks brighter.

    The defaults draw handoff 10's aurora. Options added for handoff 11:
    - lean_toward (yaw, elevation in degrees): the rays lean toward this point overhead, as real rays follow the
      earth's magnetic field toward the magnetic zenith. Default: straight up.
    - bend: pushes the rays around by up to this many ray widths, so they crowd together in places and spread apart in
      others. Default 0: fairly even.
    - drift: degrees the folds and bright patches slide along the curtains per unit of phase. Default 0: the phase
      only reshapes them. The rays always just reshape: they're too thin to slide without a crossfade between two
      frames showing them twice."""
    nt = world.node_tree
    lit = nt.nodes["Background"]
    seen = next(n for n in nt.nodes if n.type == "BACKGROUND" and n != lit)

    def m(op, a, b=None, c=None):
        return porch.math_node(nt, op, a, b, c)

    def ramp(value, a, b):  # smoothstep from 0 at a to 1 at b
        r = porch.node(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
        r.inputs["From Min"].default_value, r.inputs["From Max"].default_value = a, b
        nt.links.new(value, r.inputs["Value"])
        return r.outputs[0]

    def signed(value):  # a noise value, spread to about -1..1 (the noise mostly stays within 0.3-0.7)
        r = porch.node(nt, "ShaderNodeMapRange", clamp=True)
        r.inputs["From Min"].default_value, r.inputs["From Max"].default_value = 0.3, 0.7
        r.inputs["To Min"].default_value, r.inputs["To Max"].default_value = -1.0, 1.0
        nt.links.new(value, r.inputs["Value"])
        return r.outputs[0]

    def noise(coords, w, detail):
        vec = porch.node(nt, "ShaderNodeCombineXYZ")
        for i, v in enumerate(coords):
            if isinstance(v, bpy.types.NodeSocket):
                nt.links.new(v, vec.inputs[i])
            else:
                vec.inputs[i].default_value = v
        tex = porch.node(nt, "ShaderNodeTexNoise", noise_dimensions="4D")
        tex.inputs["Scale"].default_value = 1.0
        tex.inputs["Detail"].default_value = detail
        tex.inputs["Roughness"].default_value = 0.5
        tex.inputs["W"].default_value = w
        nt.links.new(vec.outputs[0], tex.inputs["Vector"])
        return tex.outputs["Fac"]

    def scale(color, amount):
        s = porch.node(nt, "ShaderNodeVectorMath", operation="SCALE")
        s.inputs[0].default_value = color
        nt.links.new(amount, s.inputs["Scale"])
        return s.outputs[0]

    def vadd(a, b):
        s = porch.node(nt, "ShaderNodeVectorMath", operation="ADD")
        nt.links.new(a, s.inputs[0])
        nt.links.new(b, s.inputs[1])
        return s.outputs[0]

    def dot(vector, constant):
        d = porch.node(nt, "ShaderNodeVectorMath", operation="DOT_PRODUCT")
        nt.links.new(vector, d.inputs[0])
        d.inputs[1].default_value = constant
        return d.outputs["Value"]

    split = porch.node(nt, "ShaderNodeSeparateXYZ")
    direction = porch.node(nt, "ShaderNodeNewGeometry").outputs["Position"]  # the ray's direction
    nt.links.new(direction, split.inputs[0])
    dx, dy, up = split.outputs["X"], split.outputs["Y"], split.outputs["Z"]
    yaw = m("ARCTAN2", dx, dy)
    slope = m("DIVIDE", up, m("MAXIMUM", m("SQRT", m("ADD", m("MULTIPLY", dx, dx), m("MULTIPLY", dy, dy))), 1e-3))

    def per_degrees(value, degrees):  # yaw (radians) in units of so many degrees
        return m("MULTIPLY", value, 1 / math.radians(degrees))

    slide = math.radians(drift) * phase

    def slid(at):  # a position along the curtains, moved by the drift; the curtains themselves stay where they are
        return m("ADD", at, slide) if slide else at

    ray_at = yaw  # where a ray sits along the curtains: by yaw, the rays run straight up
    if lean_toward:
        # By the angle around the point they lean toward instead: every line of sight on one great circle through that
        # point gets the same value, so each ray follows one of those circles. With the point straight up, it's the yaw.
        az, el = (math.radians(a) for a in lean_toward)
        pole = Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
        north = (Vector((0.0, 1.0, 0.0)) - pole * pole.y).normalized()
        ray_at = m("ARCTAN2", dot(direction, north.cross(pole)), dot(direction, north))

    green_sum = top_sum = None
    for c in CURTAINS:
        def distance(at, c=c):
            spread, along = 1.0, slid(at)
            for i, (depth, size) in enumerate(WAVES):
                wave = signed(noise((per_degrees(along, size), c.seed + 10 * i, 0.0), phase * 0.3 + c.seed, 1.0))
                spread = m("MULTIPLY_ADD", wave, depth, spread)
            facing = m("MAXIMUM", m("COSINE", m("SUBTRACT", at, math.radians(c.bearing))), 0.12)
            return m("DIVIDE", m("MULTIPLY", spread, c.distance), facing)

        step = math.radians(0.1)
        near = distance(yaw)
        turn = m("DIVIDE", m("SUBTRACT", distance(m("ADD", yaw, step)), near), m("MULTIPLY", near, step))
        fold = m("MINIMUM", m("SQRT", m("MULTIPLY_ADD", turn, turn, 1.0)), FOLD_MAX)
        height = m("MULTIPLY_ADD", near, slope, m("DIVIDE", m("MULTIPLY", near, near), 2 * EARTH_R_KM))
        z = m("SUBTRACT", height, c.base)
        across = per_degrees(ray_at, RAY_SIZE)
        if bend:
            push = noise((per_degrees(ray_at, RAY_BEND_SIZE), c.seed + 2.9, 0.0), phase * 0.2 + c.seed, 1.0)
            across = m("MULTIPLY_ADD", signed(push), bend, across)
        ray = noise((across, m("DIVIDE", z, 120.0), c.seed), phase + c.seed, 3.0)
        bunched = ramp(noise((per_degrees(ray_at, RAY_PATCH), c.seed + 5.3, 0.0), phase * 0.5 + c.seed, 1.0),
                       0.48, 0.66)
        rays = m("MULTIPLY_ADD", m("MULTIPLY", signed(ray), bunched), 0.7, 1.0)  # 1 where the sheet is smooth
        reach = ramp(noise((per_degrees(ray_at, RAY_SIZE * 1.7), c.seed + 3.1, 0.0), phase + c.seed, 1.0), 0.35, 0.7)
        z = m("MULTIPLY_ADD", signed(ray), AURORA_RAGGED, z)  # each ray's foot a little higher or lower
        above = m("MAXIMUM", z, 0.0)
        hem = m("EXPONENT", m("DIVIDE", above, -HEM_HEIGHT))
        reach_km = m("MULTIPLY_ADD", reach, RAY_HEIGHT[1] - RAY_HEIGHT[0], RAY_HEIGHT[0])
        streaks = m("MULTIPLY", m("EXPONENT", m("DIVIDE", m("MULTIPLY", above, -1.0), reach_km)), rays)
        edge = ramp(z, -AURORA_EDGE, 0.4 * AURORA_EDGE)
        under = m("MULTIPLY", m("SUBTRACT", 1.0, edge),
                  m("EXPONENT", m("DIVIDE", m("MINIMUM", z, 0.0), UNDER_GLOW[1])))
        green = m("MULTIPLY_ADD", under, UNDER_GLOW[0], m("MULTIPLY", edge, m(
            "ADD", m("MULTIPLY", hem, m("MULTIPLY_ADD", rays, 0.3, 0.3)), m("MULTIPLY", streaks, 0.45))))
        top = m("MULTIPLY", m("MULTIPLY", ramp(z, AURORA_TOP[0], AURORA_TOP[1]), m("MULTIPLY_ADD", rays, 0.4, 0.6)),
                m("EXPONENT", m("MULTIPLY", m("MAXIMUM", m("SUBTRACT", z, AURORA_TOP[1]), 0.0), -1.0 / AURORA_TOP[2])))
        patches = ramp(noise((per_degrees(slid(yaw), PATCH_SIZE), c.seed + 7.7, 0.0), phase * 0.2 + c.seed, 1.0),
                       0.38, 0.62)
        weight = m("MULTIPLY", fold, m("MULTIPLY_ADD", patches, 0.85 * c.brightness, 0.15 * c.brightness))
        green, top = m("MULTIPLY", green, weight), m("MULTIPLY", top, m("MULTIPLY", weight, c.fringe))
        green_sum = green if green_sum is None else m("ADD", green_sum, green)
        top_sum = top if top_sum is None else m("ADD", top_sum, top)

    extinction = m("EXPONENT", m("DIVIDE", -AURORA_EXTINCTION, m("MAXIMUM", up, 0.02)))
    centre = math.radians(sum(c.bearing for c in CURTAINS[:2]) / 2)
    toward = ramp(m("COSINE", m("SUBTRACT", yaw, centre)), 0.3, 0.85)
    low = m("EXPONENT", m("DIVIDE", m("MAXIMUM", up, 0.0), -0.06))
    color = vadd(vadd(scale(look.green, m("MULTIPLY", green_sum, look.brightness)),
                      scale(look.top, m("MULTIPLY", top_sum, look.brightness * look.top_share))),
                 scale(look.green, m("MULTIPLY", m("MULTIPLY", toward, low), look.glow)))
    aurora = porch.node(nt, "ShaderNodeVectorMath", operation="SCALE")
    nt.links.new(color, aurora.inputs[0])
    nt.links.new(m("MULTIPLY", extinction, strength), aurora.inputs["Scale"])

    for background in (lit, seen):
        before = next(lk.from_socket for lk in nt.links if lk.to_socket == background.inputs["Color"])
        nt.links.new(vadd(before, aurora.outputs[0]), background.inputs["Color"])


# ---------------------------------------------------------------------------------------------
# Porch light fixtures. One is in view in every still and its light is on from blue hour; it replaces 08's lantern out
# of frame. What you see glowing (a frosted bulb, a filament, a candle flame) only shows itself: a small point light in
# the same place lights the scene, so the lighting stays clean and the fixture's frame casts the shadows.

SCONCE_AT = (porch.POSTS_X[0], -0.07, 1.70)  # (a): the porch side of the left post, at the lantern's base
CEILING_AT = (5.165, -0.5)  # (b): the cord meets the ceiling here, between two rafters
BULB_TIP = 2.10  # (b): height of the bulb's lowest point above the deck
RAIL_LANTERN_AT = (3.15, 0.0, porch.RAIL_TOP)  # (c): on the rail cap, right of centre
RAIL_LANTERN_TURN = 8.0  # degrees off square to the rail


@dataclass(frozen=True)
class LightSpec:
    power: float  # W
    kelvin: float
    radius: float  # m: the size of the glowing part, which softens the shadows


LIGHTS = {"a": LightSpec(1.6, 2700.0, 0.015), "b": LightSpec(1.0, 2700.0, 0.008), "c": LightSpec(1.2, 1900.0, 0.005)}
FIXTURE_NAMES = {"a": "sconce on the left post", "b": "bare bulb from the ceiling", "c": "candle lantern on the rail"}


@dataclass
class Fixture:
    parts: list  # every object; in the render only while this fixture is the one in view
    light: object  # the point light
    glow: list  # (emission strength socket, value when lit)
    lit_only: list  # objects there only while lit (the candle flame)


def new_collection(name):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


def empty(name, loc, coll, parent=None):
    obj = porch.link_object(bpy.data.objects.new(name, None), coll)
    obj.parent, obj.location = parent, loc
    return obj


def mesh_object(name, bm, mat, coll, smooth=True):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mat)
    if smooth:
        mesh.shade_smooth()
        mesh.set_sharp_from_angle(angle=math.radians(35))  # crisp rims, smooth curves
    return porch.link_object(bpy.data.objects.new(name, mesh), coll)


def lathe(name, profile, mat, coll, segments=40):
    """A round part turned from (radius, height) points, like 05's mug."""
    bm = bmesh.new()
    verts = [bm.verts.new((r, 0.0, z)) for r, z in profile]
    edges = [bm.edges.new((a, b)) for a, b in pairwise(verts)]
    bmesh.ops.spin(bm, geom=verts + edges, cent=(0, 0, 0), axis=(0, 0, 1), angle=math.tau, steps=segments,
                   use_merge=True)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    return mesh_object(name, bm, mat, coll)


def pyramid(name, bottom, top, height, mat, coll):
    """A square roof with its base at z = 0 (bottom and top are half-widths)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=4, radius1=bottom * math.sqrt(2),
                          radius2=top * math.sqrt(2), depth=height,
                          matrix=Matrix.Translation((0, 0, height / 2)) @ Matrix.Rotation(math.radians(45), 4, "Z"))
    obj = mesh_object(name, bm, mat, coll, smooth=False)
    bevel = obj.modifiers.new("bevel", "BEVEL")
    bevel.width, bevel.segments, bevel.limit_method = 0.002, 2, "ANGLE"
    return obj


def ring(name, major, minor, mat, coll, segments=(36, 10)):
    """A ring standing in the XZ plane, centred on the origin: a lantern's carrying handle."""
    bm = bmesh.new()
    grid = [[bm.verts.new(((major + minor * math.cos(v)) * math.cos(u), minor * math.sin(v),
                           (major + minor * math.cos(v)) * math.sin(u)))
             for v in np.linspace(0, math.tau, segments[1], endpoint=False)]
            for u in np.linspace(0, math.tau, segments[0], endpoint=False)]
    for i in range(segments[0]):
        for j in range(segments[1]):
            i2, j2 = (i + 1) % segments[0], (j + 1) % segments[1]
            bm.faces.new((grid[i][j], grid[i2][j], grid[i2][j2], grid[i][j2]))
    return mesh_object(name, bm, mat, coll)


def filament(name, mat, coll, centre, strands=6, radius=0.008, height=0.028):
    """A squirrel-cage filament: thin strands bowing out between a top and a bottom support. Returns the object and its
    glowing area."""
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth, curve.bevel_resolution = 0.0004, 2
    length = 0.0
    for k in range(strands):
        a = math.tau * (k + 0.5) / strands
        pts = [((0.3 + 0.7 * math.sin(math.pi * t)) * radius * math.cos(a),
                (0.3 + 0.7 * math.sin(math.pi * t)) * radius * math.sin(a), centre + (t - 0.5) * height)
               for t in np.linspace(0.0, 1.0, 9)]
        spline = curve.splines.new("POLY")
        spline.points.add(len(pts) - 1)
        for p, co in zip(spline.points, pts, strict=True):
            p.co = (*co, 1.0)
        length += sum((Vector(b) - Vector(a)).length for a, b in pairwise(pts))
    curve.materials.append(mat)
    return porch.link_object(bpy.data.objects.new(name, curve), coll), length * math.tau * curve.bevel_depth


def area(obj):
    return sum(p.area for p in obj.data.polygons)


def glow_only(obj):
    """What you see of a lit bulb or flame shows to the camera and through glass; the point light does the lighting,
    the shadows and the highlights."""
    obj.visible_diffuse = obj.visible_glossy = obj.visible_volume_scatter = obj.visible_shadow = False


def glowing_material(name, color, roughness, kelvin, **inputs):
    """A surface that glows when its fixture is lit (a bulb, a filament). Not a light itself: the point light is."""
    mat = porch.plain_material(name, color, roughness, **inputs)
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    body = porch.node(nt, "ShaderNodeBlackbody")
    body.inputs["Temperature"].default_value = kelvin
    nt.links.new(body.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.0
    mat.cycles.emission_sampling = "NONE"
    return mat, bsdf.inputs["Emission Strength"]


def flame_material(kelvin):
    """A candle flame: brightest through its middle, see-through at its outline."""
    mat = bpy.data.materials.new("fixture_flame")
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    body = porch.node(nt, "ShaderNodeBlackbody")
    body.inputs["Temperature"].default_value = kelvin
    emit = porch.node(nt, "ShaderNodeEmission")
    nt.links.new(body.outputs["Color"], emit.inputs["Color"])
    facing = porch.node(nt, "ShaderNodeLayerWeight").outputs["Facing"]  # 0 facing the camera, 1 at the outline
    mixed = porch.node(nt, "ShaderNodeMixShader")
    nt.links.new(porch.math_node(nt, "POWER", porch.math_node(nt, "SUBTRACT", 1.0, facing), 0.8), mixed.inputs[0])
    nt.links.new(porch.node(nt, "ShaderNodeBsdfTransparent").outputs[0], mixed.inputs[1])
    nt.links.new(emit.outputs[0], mixed.inputs[2])
    nt.links.new(mixed.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    mat.cycles.emission_sampling = "NONE"
    return mat, emit.inputs["Strength"]


def fixture_materials():
    return {
        "metal": porch.plain_material("fixture_metal", (0.03, 0.026, 0.022), 0.38,
                                      **{"Coat Weight": 0.3, "Coat Roughness": 0.25}),  # dark bronze paint
        "glass": porch.plain_material("fixture_glass", (1.0, 1.0, 1.0), 0.0, **{"Transmission Weight": 1.0}),
        "brass": porch.plain_material("fixture_brass", (0.50, 0.36, 0.17), 0.3, Metallic=1.0),
        "cord": porch.plain_material("fixture_cord", (0.012, 0.011, 0.010), 0.7),
        # Light travels a few mm into the wax; about as far in every colour, so it stays cream rather than orange.
        "wax": porch.plain_material("fixture_wax", (0.85, 0.79, 0.68), 0.45,
                                    **{"Subsurface Weight": 0.6, "Subsurface Scale": 0.006,
                                       "Subsurface Radius": (1.0, 0.8, 0.6)}),
        "wick": porch.plain_material("fixture_wick", (0.03, 0.025, 0.02), 0.9),
    }


def point_light(name, spec, parent, loc, coll):
    light = bpy.data.lights.new(name, "POINT")
    light.energy = spec.power
    light.use_temperature = True
    light.temperature = spec.kelvin
    light.shadow_soft_size = spec.radius
    obj = porch.link_object(bpy.data.objects.new(name, light), coll)
    obj.parent, obj.location = parent, loc
    obj.visible_transmission = False  # through the glass you see the glowing part, not this light's sphere
    return obj


def lantern(prefix, width, glass_h, mats, coll, parent, glass_bottom=False):
    """A square lantern around its own origin (the middle of its underside): a base, four corner posts with glass between
    them, a top band, a pyramid roof and a cap. With glass_bottom the base is a frame round a glass floor, so the light
    also falls downward. Returns its parts, the heights of its glass's bottom and top, and its overall height."""
    parts = []

    def part(obj):
        obj.parent = parent
        parts.append(obj)
        return obj

    metal, post, z0, w = mats["metal"], 0.009, 0.022, width
    for name, size, z, bevel in (("base", w + 0.02, 0.006, 0.003), ("foot", w + 0.006, 0.017, 0.002)):
        h = 0.012 if name == "base" else 0.010
        if not glass_bottom:
            part(porch.box(f"{prefix}_{name}", (size, size, h), (0, 0, z), mat=metal, bevel=bevel, coll=coll))
            continue
        bar = 0.016 if name == "base" else 0.008
        for i, (x, y, sx, sy) in enumerate(((0, -1, size, bar), (0, 1, size, bar), (-1, 0, bar, size - 2 * bar),
                                            (1, 0, bar, size - 2 * bar))):
            part(porch.box(f"{prefix}_{name}_{i}", (sx, sy, h), (x * (size - bar) / 2, y * (size - bar) / 2, z),
                           mat=metal, bevel=min(bevel, 0.0015), coll=coll))
    if glass_bottom:
        part(porch.box(f"{prefix}_glass_floor", (w - 0.012, w - 0.012, 0.003), (0, 0, 0.006), mat=mats["glass"],
                       bevel=0.0, coll=coll)).visible_shadow = False
    c = w / 2 - post / 2
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        part(porch.box(f"{prefix}_post_{i}", (post, post, glass_h), (sx * c, sy * c, z0 + glass_h / 2), mat=metal,
                       bevel=0.0015, coll=coll))
    for i, (x, y, turn) in enumerate(((0, -c, 0), (0, c, 0), (-c, 0, 90), (c, 0, 90))):
        pane = part(porch.box(f"{prefix}_glass_{i}", (2 * c, 0.002, glass_h), (x, y, z0 + glass_h / 2),
                              (0, 0, math.radians(turn)), mats["glass"], 0.0, coll))
        pane.visible_shadow = False  # the light inside shines through
    top = z0 + glass_h
    part(porch.box(f"{prefix}_band", (w + 0.012, w + 0.012, 0.014), (0, 0, top + 0.007), mat=metal, bevel=0.002,
                   coll=coll))
    part(pyramid(f"{prefix}_roof", w / 2 + 0.014, 0.012, 0.05, metal, coll)).location.z = top + 0.014
    cap = ((0, 0), (0.011, 0), (0.011, 0.016), (0.016, 0.016), (0.016, 0.021), (0.004, 0.024), (0, 0.024))
    part(lathe(f"{prefix}_cap", cap, metal, coll, 24)).location.z = top + 0.064
    return parts, z0, top, top + 0.088


def build_sconce(mats):
    """(a) A wall lantern on the porch side of the left post: a back plate, a short arm and a glass-floored lantern with
    a small frosted globe bulb hanging from its top, so it lights the post, the rail and the deck below it."""
    coll = new_collection("fixture_a")
    root = empty("fixture_a", SCONCE_AT, coll)
    plate = porch.box("sconce_plate", (0.075, 0.012, 0.28), (0, -0.006, 0.15), mat=mats["metal"], bevel=0.003,
                      coll=coll)
    arm = porch.box("sconce_arm", (0.016, 0.06, 0.016), (0, -0.042, 0.19), mat=mats["metal"], bevel=0.002, coll=coll)
    plate.parent = arm.parent = root
    w, glass_h = 0.13, 0.19
    body = empty("sconce_lantern", (0, -0.072 - w / 2, 0.0), coll, root)
    parts, _, top, _ = lantern("sconce", w, glass_h, mats, coll, body, glass_bottom=True)
    r = 0.02
    centre = top - 0.034 - r
    socket = lathe("sconce_socket", ((0, top), (0.011, top), (0.011, top - 0.031), (0.008, top - 0.034),
                                     (0, top - 0.034)), mats["metal"], coll, 24)
    bulb_mat, glow = glowing_material("sconce_bulb", (0.92, 0.90, 0.86), 0.45, LIGHTS["a"].kelvin,
                                      **{"Transmission Weight": 0.5})
    bulb = lathe("sconce_bulb", [(r * math.sin(t), centre - r * math.cos(t)) for t in np.linspace(0, math.pi, 17)],
                 bulb_mat, coll, 32)
    for obj in (socket, bulb):
        obj.parent = body
    glow_only(bulb)
    light = point_light("sconce_light", LIGHTS["a"], body, (0, 0, centre), coll)
    return Fixture([plate, arm, socket, bulb, *parts], light, [(glow, radiance(LIGHTS["a"], area(bulb)))], [])


def build_bulb(mats):
    """(b) A clear bulb (ST64 shape) with a squirrel-cage filament, in a brass socket on a fabric cord from a canopy on
    the sloping ceiling."""
    coll = new_collection("fixture_b")
    x, y = CEILING_AT
    ceiling = porch.POST_TOP + 0.38 - porch.EAVE_SLOPE * y  # underside of the ceiling boards, on top of the rafters
    root = empty("fixture_b", (x, y, ceiling), coll)
    canopy = lathe("bulb_canopy", ((0, 0), (0.04, 0), (0.04, -0.006), (0.033, -0.02), (0.012, -0.026), (0, -0.026)),
                   mats["brass"], coll)
    canopy.rotation_euler.x = -math.atan(porch.EAVE_SLOPE)  # flat against the ceiling, which rises toward the cabin
    tip = BULB_TIP - ceiling
    socket_z = tip + 0.117  # the glass is 11.7 cm long
    cord = lathe("bulb_cord", ((0, -0.02), (0.0035, -0.02), (0.0035, socket_z + 0.05), (0, socket_z + 0.05)),
                 mats["cord"], coll, 12)
    socket = lathe("bulb_socket", [(r, socket_z + z) for r, z in (
        (0, 0.0), (0.0135, 0.0), (0.0145, 0.002), (0.0175, 0.005), (0.0175, 0.030), (0.0166, 0.032), (0.0166, 0.036),
        (0.0175, 0.038), (0.0175, 0.048), (0.011, 0.055), (0, 0.055))], mats["brass"], coll, 32)
    envelope = lathe("bulb_glass", [(r, socket_z + z) for r, z in (
        (0, 0.0), (0.0125, 0.0), (0.013, -0.008), (0.016, -0.020), (0.024, -0.036), (0.0305, -0.054), (0.032, -0.068),
        (0.030, -0.084), (0.024, -0.098), (0.015, -0.109), (0.006, -0.115), (0, -0.117))], mats["glass"], coll, 48)
    envelope.visible_shadow = False
    centre = socket_z - 0.06
    wire_mat, glow = glowing_material("bulb_filament", (0.25, 0.24, 0.23), 0.4, LIGHTS["b"].kelvin, Metallic=1.0)
    wire, wire_area = filament("bulb_filament", wire_mat, coll, centre)
    glow_only(wire)
    for obj in (canopy, cord, socket, envelope, wire):
        obj.parent = root
    light = point_light("bulb_light", LIGHTS["b"], root, (0, 0, centre), coll)
    return Fixture([canopy, cord, socket, envelope, wire], light, [(glow, radiance(LIGHTS["b"], wire_area))], [])


def build_rail_lantern(mats):
    """(c) A small candle lantern standing on the rail cap, with a carrying ring on top and a pillar candle inside."""
    coll = new_collection("fixture_c")
    root = empty("fixture_c", RAIL_LANTERN_AT, coll)
    root.rotation_euler.z = math.radians(RAIL_LANTERN_TURN)
    parts, _, _, top = lantern("rail_lantern", 0.105, 0.16, mats, coll, root)
    handle = ring("rail_lantern_ring", 0.019, 0.0025, mats["metal"], coll)
    handle.location.z = top + 0.019 - 0.0025
    # A slim pillar: a wide one would shade everything more than about 25 degrees below the flame, the deck included.
    candle = lathe("candle", [(r, 0.022 + z) for r, z in (
        (0, 0.0), (0.018, 0.0), (0.019, 0.002), (0.019, 0.0625), (0.018, 0.065), (0.0155, 0.0625), (0, 0.06))],
        mats["wax"], coll, 40)
    wick = lathe("candle_wick", ((0, 0.081), (0.0007, 0.081), (0.0007, 0.094), (0, 0.094)), mats["wick"], coll, 8)
    wick.visible_shadow = False
    flame_mat, glow = flame_material(LIGHTS["c"].kelvin)
    flame = lathe("candle_flame", [(r, 0.091 + z) for r, z in (
        (0, 0.0), (0.0025, 0.001), (0.0042, 0.0045), (0.0047, 0.008), (0.0042, 0.012), (0.003, 0.0165),
        (0.0015, 0.0205), (0, 0.024))], flame_mat, coll, 24)
    glow_only(flame)
    for obj in (handle, candle, wick, flame):
        obj.parent = root
    light = point_light("candle_light", LIGHTS["c"], root, (0, 0, 0.104), coll)
    return Fixture([handle, candle, wick, flame, *parts], light, [(glow, radiance(LIGHTS["c"], area(flame)))], [flame])


def radiance(spec, glowing_area):
    """How bright the glowing part looks when lit: the light's power spread over its surface."""
    return spec.power / (math.pi * glowing_area)


def build_fixtures():
    mats = fixture_materials()
    fixtures = {"a": build_sconce(mats), "b": build_bulb(mats), "c": build_rail_lantern(mats)}
    bpy.context.view_layer.update()  # world positions of the parented parts, for screen_box()
    return fixtures


def show_fixture(fixtures, which, lit):
    for name, f in fixtures.items():
        here = name == which
        for obj in f.parts:
            obj.hide_render = not here or (obj in f.lit_only and not lit)
        f.light.hide_render = not (here and lit)
        for socket, value in f.glow:
            socket.default_value = value if lit else 0.0


def screen_box(scene, objs):
    """Where the fixture is in the frame: x0, x1, y0, y1 in pixels from the top-left of the 2560x1440 frame."""
    xs, ys = [], []
    for obj in objs:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            p = world_to_camera_view(scene, scene.camera, obj.matrix_world @ Vector(corner))
            xs.append(p.x * porch.FINAL_RES[0])
            ys.append((1 - p.y) * porch.FINAL_RES[1])
    return round(min(xs)), round(max(xs)), round(min(ys)), round(max(ys))


# ---------------------------------------------------------------------------------------------
# The stills.

@dataclass(frozen=True)
class Still:
    file: str  # output name, without .webp
    key: str  # 08's keyframe
    look: str  # aurora look
    fixture: str
    same_as: str = ""  # the same settings as another still: rendered once, saved under both names
    note: str = ""  # added to its label


STILLS = {s.file: s for s in (
    Still("aurora-subtle-k5", "k5", "subtle", "a"),
    Still("aurora-vivid-k5", "k5", "vivid", "a"),
    Still("aurora-subtle-k4", "k4", "subtle", "a", note="not visible yet"),
    Still("fixture-a-k5", "k5", "subtle", "a", same_as="aurora-subtle-k5"),
    Still("fixture-b-k5", "k5", "subtle", "b"),
    Still("fixture-c-k5", "k5", "subtle", "c"),
    Still("fixture-a-k1", "k1", "subtle", "a"),
    Still("fixture-b-k1", "k1", "subtle", "b"),
    Still("fixture-c-k1", "k1", "subtle", "c"),
)}


def describe(still):
    key = kf.KEYS[still.key]
    strength = aurora_strength(key.sun)
    aurora = f"{still.look} aurora" + (f" at {strength:.0%}" if 0 < strength < 1 else "") if strength else "no aurora"
    light = "lit" if key.porch_light else "unlit"
    return f"{key.moment}, {aurora}, {FIXTURE_NAMES[still.fixture]} ({still.fixture}, {light})"


def still_label(still):
    twin = f" (= {still.same_as})" if still.same_as else ""
    note = f"; {still.note}" if still.note else ""
    return f"{still.file}{twin}   ·   {describe(still)}{note}"


def quality(still):
    return 95 if still.key == "k1" else 98  # 98 at night keeps the faint stars (09)


def contact_sheet():
    """All nine stills on disk at 1280x720, labelled, three to a row: the aurora, the fixtures lit, the fixtures unlit."""
    w, h, gap = 1280, 720, 24
    tiles = []
    for still in STILLS.values():
        path = OUT / f"{still.file}.webp"
        if path.exists():
            tiles.append(kf.labelled(porch.scaled(path, h), still_label(still)))
        else:
            tiles.append(kf.labelled(np.full((h, w, 4), 0.06, np.float32), f"{still.file}: not rendered yet"))
    th = tiles[0].shape[0]
    sheet = np.full((3 * th + 4 * gap, 3 * w + 4 * gap, 4), 0.06, np.float32)
    for i, tile in enumerate(tiles):
        row, col = divmod(i, 3)
        top = gap + row * (th + gap)
        sheet[sheet.shape[0] - top - th:sheet.shape[0] - top, gap + col * (w + gap):gap + col * (w + gap) + w] = tile
    porch.write_pixels(sheet, OUT / "10-night-options.webp", quality=92)


def save_log_section(still, lines):
    """10-render-log.txt keeps one section per still; rendering a still again replaces its section."""
    sections = {}
    if LOG_PATH.exists():
        for block in LOG_PATH.read_text(encoding="utf-8").split("\n=== ")[1:]:
            sections[block.split(" ", 1)[0]] = "=== " + block.strip("\n")
    sections[still.file] = f"=== {still.file} {describe(still)}, rendered {time.strftime('%Y-%m-%d %H:%M')}\n" + \
        "\n".join(lines)
    LOG_PATH.write_text("Night options (handoff 10). One section per still, from the run that rendered it.\n" + "".join(
        f"\n{sections[name]}\n" for name in STILLS if name in sections), encoding="utf-8")


# ---------------------------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(prog="10_night_options.py")
    p.add_argument("--stills", default=",".join(STILLS), help="stills to render, comma-separated (default: all)")
    p.add_argument("--preview", metavar="NAME", help="1280x720 labelled tests into preview/NAME-<still>.webp")
    p.add_argument("--samples", type=int, help="override the sample count")
    p.add_argument("--no-trees", action="store_true", help="skip the trees (fast look tests)")
    args = p.parse_args(argv)
    args.stills = args.stills.split(",")
    unknown = [s for s in args.stills if s not in STILLS]
    if unknown:
        p.error(f"unknown still(s) {unknown}; choose from {list(STILLS)}")
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

    preview = args.preview is not None
    samples = args.samples or (porch.PREVIEW_SAMPLES if preview else porch.FINAL_SAMPLES)
    wanted = [STILLS[name] for name in args.stills]
    # Each picture is rendered once: a still with a twin is copied from the twin, which is rendered if it isn't asked.
    todo = list(dict.fromkeys(STILLS[s.same_as] if s.same_as else s for s in wanted))
    todo.sort(key=lambda s: kf.KEYS[s.key].sun)  # the night first: the golden hour renders take longest
    OUT.mkdir(parents=True, exist_ok=True)

    lamps = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in dict.fromkeys(s.key for s in todo):
            key = kf.KEYS[name]
            lamps[name] = kf.measure_lamp(key, Path(tmp)) if key.lamp_from else (0.0, (0.0, 0.0, 0.0))
    log(f"Sun lamp measured in {time.time() - started:.0f} s")
    scene, devices = kf.build_scene(not args.no_trees, preview, samples, log)
    fixtures = build_fixtures()
    r = scene.render
    log(f"Scene built in {time.time() - started:.0f} s (peak RAM so far {porch.peak_ram_gb():.1f} GB)")
    log(f"Render: {r.resolution_x * r.resolution_percentage // 100}x{r.resolution_y * r.resolution_percentage // 100}, "
        f"{samples} samples, denoiser {scene.cycles.denoiser}, on {devices}; trees {'off' if args.no_trees else 'on'}")
    log(f"Night glow at {NIGHT_GLOW:g} of 08's; aurora fades in from sun {AURORA_FADE[0]:+g} to {AURORA_FADE[1]:+g} deg, "
        f"phase {AURORA_PHASE:g}; 08's lantern out of frame stays off")

    for still in todo:
        lines = list(shared)

        def note(msg, lines=lines):
            print(msg, flush=True)
            lines.append(msg)

        key = kf.KEYS[still.key]
        kf.apply_key(scene, replace(key, porch_light=False), lamps[key.name])
        look, strength = LOOKS[still.look], aurora_strength(key.sun)
        if strength > 0:
            add_aurora(scene.world, look, strength)
        show_fixture(fixtures, still.fixture, key.porch_light)
        spec = LIGHTS[still.fixture]
        x0, x1, y0, y1 = screen_box(scene, fixtures[still.fixture].parts)
        note(f"{still.file}: {describe(still)}")
        note(f"Sun {key.sun:+.1f} deg; exposure {key.exposure:+.2f} EV; aurora {look.name} at {strength:.0%} "
             f"(edge radiance {look.brightness * strength:.4g})")
        note(f"Fixture {still.fixture}: {FIXTURE_NAMES[still.fixture]}, "
             f"{f'lit, {spec.power:g} W at {spec.kelvin:.0f} K' if key.porch_light else 'unlit'}; in the frame at "
             f"x {x0}-{x1}, y {y0}-{y1} px (of 2560x1440)")
        seconds, peak_mb = kf.render(scene)
        note(f"Render time: {seconds:.1f} s")
        note(f"Cycles peak memory (highest reported during the render): {peak_mb:,.0f} MB; "
             f"peak RAM so far in this run: {porch.peak_ram_gb():.1f} GB")
        twins = [s for s in STILLS.values() if s.same_as == still.file]  # kept in step with the still they copy
        if preview:
            folder = OUT / "preview"
            folder.mkdir(exist_ok=True)
            path = folder / f"{args.preview}-{still.file}.webp"
            kf.save_render(scene, path, "WEBP", 95)
            porch.write_pixels(kf.labelled(kf.load(path), f"{args.preview}   ·   {still_label(still)}"), path, quality=90)
            note(f"Wrote preview/{path.name}")
            with open(folder / "preview-log.txt", "a", encoding="utf-8") as f:
                f.write(f"--- {args.preview} {still.file} {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lines) + "\n")
            continue
        path = OUT / f"{still.file}.webp"
        kf.save_render(scene, path, "WEBP", quality(still))
        note(f"Wrote {path.name} (WebP quality {quality(still)})")
        px = kf.load(path)[..., :3]
        note(f"Clipping: {(px.max(-1) >= 0.995).mean():.3%} of pixels at full white, "
             f"{(px.max(-1) <= 0.005).mean():.3%} at full black")
        note(f"Run time so far: {time.time() - started:.0f} s")
        save_log_section(still, lines)
        for twin in twins:
            shutil.copyfile(path, OUT / f"{twin.file}.webp")
            save_log_section(twin, lines + [f"Saved as {twin.file}.webp too: the same settings as {still.file}"])
            print(f"Wrote {twin.file}.webp (a copy of {path.name})", flush=True)
    if not preview:
        contact_sheet()
        print(f"Wrote 10-night-options.webp; run time {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
