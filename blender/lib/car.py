r"""Shared 1975 911 (930) Turbo setup that every scene script imports.

    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))  # in blender\<script>.py
    from lib import car as car_lib
    car = car_lib.build(scene, paint="guards-red")
    car_lib.drive(car, curve_obj, 1, 120, lambda f: 0.4 * f)
    car_lib.fade_lights(car, 30, 50, head=(0, 1), tail=(0, 1))

The car's origin (`car.root`) sits on the ground under the middle of the rear axle and the car faces
-Y, like 01. The rear axle follows a path exactly; the front wheels steer to match its bends.
Animation is plain keyframes only: no drivers, since Python drivers would need "Auto Run Python
Scripts", which this project never enables. The .glb is only read, never changed.
"""

import bisect
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import bpy
from mathutils import Matrix, Vector, geometry

ROOT = Path(__file__).resolve().parents[2]
CAR_GLB = ROOT / "assets-source" / "free_1975_porsche_911_930_turbo.glb"

CAR_LENGTH = 4.29  # metres bumper to bumper, like a real 930 Turbo
SHADOW_PLANE_MATERIAL = "material_0"  # flat baked-shadow sheet that ships with the model
COAT_SHELL_MATERIAL = "coat"  # 24%-opaque grey copy of the body; would grey down every paint colour
STEER_SMOOTHING = 3.0  # metres of road the steering looks over, like a driver easing into a bend


@dataclass(frozen=True)
class Paint:
    name: str
    srgb: str  # hex, as on a paint chip; converted to linear for the shader
    metallic: bool


# The one place paint colours live. Values are approximations from period photos and chips (no
# colorimetric source exists for 1975 paint); tune by eye. Metallic/solid per 1975-76 Turbo listings.
PAINTS = {
    "guards-red": Paint("Guards Red", "#C4141C", metallic=False),
    "grand-prix-white": Paint("Grand Prix White", "#EEECE4", metallic=False),
    "black": Paint("Black", "#0E0E10", metallic=False),
    "minerva-blue": Paint("Minerva Blue Metallic", "#24406E", metallic=True),
    "silver": Paint("Silver Metallic", "#A9ACAE", metallic=True),
    "copper-brown": Paint("Copper Brown Metallic", "#6E3B22", metallic=True),
}

# Light texture areas (UV boxes: u0, v0, u1, v1) that glow. Found by mapping the shared
# `930_lights` texture to the lamps on the car; the amber indicators, fog lamps, reverse lights and
# the rear reflector strip are left unlit.
HEADLIGHT_UV = [(0.02, 0.13, 0.39, 0.51), (0.26, 0.53, 0.36, 0.61), (0.90, 0.42, 0.94, 0.50)]
TAILLIGHT_UV = [(0.44, 0.75, 0.56, 0.87), (0.81, 0.12, 0.99, 0.25)]
HEADLIGHT_COLOR = (1.0, 0.82, 0.62)  # warm halogen
TAILLIGHT_COLOR = (1.0, 0.03, 0.01)
HEADLIGHT_GLOW = 40.0  # emission strength at full brightness
TAILLIGHT_GLOW = 25.0
HEADLIGHT_LAMP_W = 120.0  # spot light per headlight, so the beam lights the road
TAILLIGHT_LAMP_W = 3.0

# Lamp positions in the .glb's own coordinates (before scaling), from the lamp meshes.
HEADLIGHT_RAW = [(0.73, -2.45, 0.81), (-0.73, -2.45, 0.81)]
TAILLIGHT_RAW = [(0.55, 2.16, 0.70), (-0.55, 2.16, 0.70)]


@dataclass
class Car:
    root: bpy.types.Object  # move and turn this; origin under the rear axle, front towards -Y
    steer: dict  # corner ("FL", "FR", "RL", "RR") -> empty that turns about Z (left is positive)
    spin: dict  # corner -> empty that rolls about X (positive rolls forward)
    wheel_radius: float
    wheelbase: float
    front_track: float
    centre_y: float  # body centre along local Y, measured from the rear axle (negative = forwards)
    paint: bpy.types.Material
    head_value: bpy.types.NodeSocket  # 0..1 brightness, shared by both headlights
    tail_value: bpy.types.NodeSocket
    head_lamps: list = field(default_factory=list)
    tail_lamps: list = field(default_factory=list)


def srgb_to_linear(hex_color):
    def channel(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    h = hex_color.lstrip("#")
    return tuple(channel(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4))


def _vertex_bounds(objects):
    lo, hi = Vector((math.inf,) * 3), Vector((-math.inf,) * 3)
    for obj in objects:
        if obj.type != "MESH":
            continue
        mw = obj.matrix_world
        for v in obj.data.vertices:
            p = mw @ v.co
            for i in range(3):
                lo[i], hi[i] = min(lo[i], p[i]), max(hi[i], p[i])
    return lo, hi


def _uses(obj, material_name):
    return obj.type == "MESH" and any(s.material and s.material.name == material_name for s in obj.material_slots)


# ---------------------------------------------------------------- materials

def _box_mask(nt, uv, boxes):
    """1 inside any of the UV boxes, else 0."""
    total = None
    for u0, v0, u1, v1 in boxes:
        terms = []
        for axis, edge, op in ((0, u0, "GREATER_THAN"), (0, u1, "LESS_THAN"), (1, v0, "GREATER_THAN"), (1, v1, "LESS_THAN")):
            m = nt.nodes.new("ShaderNodeMath")
            m.operation = op
            nt.links.new(uv.outputs[axis], m.inputs[0])
            m.inputs[1].default_value = edge
            terms.append(m)
        box = terms[0]
        for t in terms[1:]:
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            nt.links.new(box.outputs[0], mul.inputs[0])
            nt.links.new(t.outputs[0], mul.inputs[1])
            box = mul
        if total is None:
            total = box
        else:
            mx = nt.nodes.new("ShaderNodeMath")
            mx.operation = "MAXIMUM"
            nt.links.new(total.outputs[0], mx.inputs[0])
            nt.links.new(box.outputs[0], mx.inputs[1])
            total = mx
    return total


def _setup_light_glow(mat):
    """Emission = texture colour x lamp tint x mask x brightness, for headlights and tail-lights."""
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = bsdf.inputs["Base Color"].links[0].from_node
    uv = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["UV"], uv.inputs[0])

    parts = []
    for name, boxes, color, glow in (("HeadLights", HEADLIGHT_UV, HEADLIGHT_COLOR, HEADLIGHT_GLOW),
                                     ("TailLights", TAILLIGHT_UV, TAILLIGHT_COLOR, TAILLIGHT_GLOW)):
        value = nt.nodes.new("ShaderNodeValue")
        value.name = value.label = name
        value.outputs[0].default_value = 0.0
        strength = nt.nodes.new("ShaderNodeMath")
        strength.operation = "MULTIPLY"
        nt.links.new(_box_mask(nt, uv, boxes).outputs[0], strength.inputs[0])
        nt.links.new(value.outputs[0], strength.inputs[1])
        tint = nt.nodes.new("ShaderNodeMix")
        tint.data_type = "RGBA"
        tint.blend_type = "MULTIPLY"
        tint.inputs["Factor"].default_value = 1.0
        nt.links.new(tex.outputs["Color"], tint.inputs["A"])
        tint.inputs["B"].default_value = (*color, 1.0)
        scaled = nt.nodes.new("ShaderNodeVectorMath")
        scaled.operation = "SCALE"
        nt.links.new(tint.outputs["Result"], scaled.inputs[0])
        glow_amount = nt.nodes.new("ShaderNodeMath")
        glow_amount.operation = "MULTIPLY"
        nt.links.new(strength.outputs[0], glow_amount.inputs[0])
        glow_amount.inputs[1].default_value = glow
        nt.links.new(glow_amount.outputs[0], scaled.inputs["Scale"])
        parts.append((value, scaled))

    add = nt.nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    nt.links.new(parts[0][1].outputs[0], add.inputs[0])
    nt.links.new(parts[1][1].outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.0
    return parts[0][0].outputs[0], parts[1][0].outputs[0]


def _thin_glass(mat, tint, normal_image=None):
    """Single-sheet glass: see-through, with reflections that grow at grazing angles (Fresnel).
    The model's windows are single surfaces, so real refraction would bend the view wrongly."""
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    clear.inputs["Color"].default_value = (*tint, 1.0)
    gloss = nt.nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Roughness"].default_value = 0.0
    fresnel = nt.nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.52
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fresnel.outputs[0], mix.inputs[0])
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(gloss.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if normal_image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = normal_image
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs[0], gloss.inputs["Normal"])
        nt.links.new(nmap.outputs[0], fresnel.inputs["Normal"])


def _make_paint():
    """Car paint: preset colour over a solid or metallic base, under a glossy clear coat. The model's
    paint normal map is dropped: it only holds stone chips and scratches (wear is out of scope)."""
    mat = bpy.data.materials.new("CarPaint")
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.02
    bsdf.inputs["Coat IOR"].default_value = 1.5
    return mat


def set_paint(car, key):
    p = PAINTS[key]
    bsdf = car.paint.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(p.srgb), 1.0)
    bsdf.inputs["Metallic"].default_value = 0.75 if p.metallic else 0.0
    bsdf.inputs["Roughness"].default_value = 0.32 if p.metallic else 0.4


def _tune_materials(objects):
    mats = {s.material.name: s.material for o in objects if o.type == "MESH" for s in o.material_slots if s.material}

    bsdf = mats["930_tire"].node_tree.nodes["Principled BSDF"]  # matte rubber, not plastic-shiny
    for name, value in (("Metallic", 0.0), ("Roughness", 0.88)):
        for link in list(bsdf.inputs[name].links):
            mats["930_tire"].node_tree.links.remove(link)
        bsdf.inputs[name].default_value = value
    bsdf.inputs["Specular IOR Level"].default_value = 0.35

    chrome = mats["930_chromes"].node_tree  # mirror-like, rougher only where the texture says
    rough = chrome.nodes["Principled BSDF"].inputs["Roughness"].links[0].from_node
    rough.inputs[1].default_value = 0.2

    rim = mats["930_rim"].node_tree  # polished alloy rather than dull grey
    rough = rim.nodes["Principled BSDF"].inputs["Roughness"].links[0].from_node
    rough.inputs[1].default_value = 0.35

    black = mats["black"].node_tree.nodes["Principled BSDF"]  # satin black trim, not black metal
    black.inputs["Metallic"].default_value = 0.0
    black.inputs["Roughness"].default_value = 0.5

    lens = mats["930_lights_refraction"]
    lens_normal = next(n.image for n in lens.node_tree.nodes if n.type == "TEX_IMAGE")
    _thin_glass(lens, (1.0, 1.0, 1.0), lens_normal)
    _thin_glass(mats["glass"], (0.86, 0.93, 0.89))  # faint green, like period laminated glass

    return _setup_light_glow(mats["930_lights"])


# ---------------------------------------------------------------- build

def build(scene, paint="guards-red"):
    """Import, clean, scale and rig the car. Returns a Car; the root sits at the world origin."""
    existing = set(scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(CAR_GLB))
    for obj in [o for o in scene.objects if o not in existing]:
        if _uses(obj, SHADOW_PLANE_MATERIAL) or _uses(obj, COAT_SHELL_MATERIAL):
            bpy.data.objects.remove(obj, do_unlink=True)
    imported = [o for o in scene.objects if o not in existing]
    bpy.context.view_layer.update()

    # Wheel corners: each is an empty holding a rim and tyre (front-right also has a valve stem).
    holders = {o.parent for o in imported if _uses(o, "930_rim") or _uses(o, "930_tire")}
    corners = {}
    for h in holders:
        p = h.matrix_world.translation
        side = "L" if p.x > 0 else "R"  # facing -Y, the car's left is +X
        corners.setdefault(("F" if p.y < 0 else "R") + side, []).append(h)

    # The model ships with the front wheels turned; turn them straight about their hub.
    for key in ("FL", "FR"):
        for h in corners[key]:
            mw = h.matrix_world.copy()
            angle = mw.decompose()[1].to_euler("XYZ").z
            pivot = mw.translation.copy()
            h.matrix_world = Matrix.Translation(pivot) @ Matrix.Rotation(-angle, 4, "Z") @ Matrix.Translation(-pivot) @ mw
    bpy.context.view_layer.update()

    lo, hi = _vertex_bounds(imported)
    scale = CAR_LENGTH / (hi.y - lo.y)  # the model's long axis is Y, front at -Y
    hub = {k: v[0].matrix_world.translation.copy() for k, v in corners.items()}
    rear_y = (hub["RL"].y + hub["RR"].y) / 2
    origin = Vector(((lo.x + hi.x) / 2, rear_y, lo.z))
    fix = Matrix.Scale(scale, 4) @ Matrix.Translation(-origin)

    root = bpy.data.objects.new("Car", None)
    scene.collection.objects.link(root)
    for obj in [o for o in imported if o.parent is None]:
        obj.matrix_world = fix @ obj.matrix_world
        obj.parent = root
    bpy.context.view_layer.update()

    # Rig: per corner, a steer empty at the hub (turns about Z) holding a spin empty (rolls about X).
    steer, spin = {}, {}
    for key in corners:
        s = bpy.data.objects.new(f"Steer.{key}", None)
        s.parent = root
        s.location = fix @ hub[key]
        r = bpy.data.objects.new(f"Spin.{key}", None)
        r.parent = s
        for o in (s, r):
            scene.collection.objects.link(o)
            o.empty_display_size = 0.2
        steer[key], spin[key] = s, r
    bpy.context.view_layer.update()
    for key, holders_here in corners.items():
        for h in holders_here:
            mw = h.matrix_world.copy()
            h.parent = spin[key]
            h.matrix_parent_inverse = Matrix.Identity(4)
            h.matrix_world = mw
            for mesh in h.children:
                mesh.cycles.motion_steps = 3  # wheels turn fast; sample more steps for motion blur

    paint_mat = _make_paint()
    for o in imported:
        for slot in getattr(o, "material_slots", []):
            if slot.material and slot.material.name == "paint":
                slot.material = paint_mat
    head_value, tail_value = _tune_materials(imported)

    head_lamps, tail_lamps = [], []
    for i, raw in enumerate(HEADLIGHT_RAW):
        data = bpy.data.lights.new(f"Headlight.{i}", "SPOT")
        data.spot_size, data.spot_blend = math.radians(55), 0.6
        data.shadow_soft_size = 0.05
        data.color = HEADLIGHT_COLOR
        lamp = bpy.data.objects.new(data.name, data)
        lamp.location = fix @ Vector(raw)
        lamp.rotation_euler = (math.radians(-87), 0.0, 0.0)  # forward (-Y), 3 degrees down
        head_lamps.append(data)
        lamp.parent = root
        scene.collection.objects.link(lamp)
    for i, raw in enumerate(TAILLIGHT_RAW):
        data = bpy.data.lights.new(f"Taillight.{i}", "POINT")
        data.shadow_soft_size = 0.04
        data.color = TAILLIGHT_COLOR
        lamp = bpy.data.objects.new(data.name, data)
        lamp.location = fix @ Vector(raw)
        tail_lamps.append(data)
        lamp.parent = root
        scene.collection.objects.link(lamp)

    car = Car(
        root=root, steer=steer, spin=spin,
        wheel_radius=scale * (hub["RL"].z - lo.z),
        wheelbase=scale * (rear_y - (hub["FL"].y + hub["FR"].y) / 2),
        front_track=scale * (hub["FL"].x - hub["FR"].x),
        centre_y=scale * ((lo.y + hi.y) / 2 - rear_y),
        paint=paint_mat, head_value=head_value, tail_value=tail_value,
        head_lamps=head_lamps, tail_lamps=tail_lamps,
    )
    set_paint(car, paint)
    set_lights(car, head=0.0, tail=0.0)
    print(f"Car: x{scale:.4f} to {CAR_LENGTH} m; wheelbase {car.wheelbase:.3f} m, wheel radius "
          f"{car.wheel_radius:.3f} m, front track {car.front_track:.3f} m; paint {PAINTS[paint].name}")
    return car


# ---------------------------------------------------------------- lights

def set_lights(car, head=None, tail=None, frame=None):
    """Set headlight and/or tail-light brightness (0 off .. 1 on); keyframe it if `frame` is given."""
    for value, socket, lamps, watts in ((head, car.head_value, car.head_lamps, HEADLIGHT_LAMP_W),
                                        (tail, car.tail_value, car.tail_lamps, TAILLIGHT_LAMP_W)):
        if value is None:
            continue
        socket.default_value = value
        for lamp in lamps:
            lamp.energy = watts * value
        if frame is not None:
            socket.keyframe_insert("default_value", frame=frame)
            for lamp in lamps:
                lamp.keyframe_insert("energy", frame=frame)


def fade_lights(car, frame_start, frame_end, head=None, tail=None):
    """Fade from a to b over the frame range, e.g. head=(1, 0) for shot 4.3's headlights going out."""
    set_lights(car, head=head and head[0], tail=tail and tail[0], frame=frame_start)
    set_lights(car, head=head and head[1], tail=tail and tail[1], frame=frame_end)


# ---------------------------------------------------------------- driving

def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class CurvePath:
    """A Bezier curve object sampled by distance along it (metres, world space)."""

    def __init__(self, curve_obj, resolution=64):
        splines = curve_obj.data.splines
        if len(splines) != 1 or splines[0].type != "BEZIER":
            raise ValueError(f"{curve_obj.name}: needs exactly one Bezier spline")
        knots = splines[0].bezier_points
        mw = curve_obj.matrix_world
        self.points = []
        for a, b in itertools.pairwise(knots):
            seg = geometry.interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, resolution)
            self.points += [mw @ p for p in (seg[1:] if self.points else seg)]
        self.dist = [0.0]
        for p, q in itertools.pairwise(self.points):
            self.dist.append(self.dist[-1] + (q - p).length)
        self.length = self.dist[-1]

    def point(self, s):
        s = min(max(s, 0.0), self.length)
        i = min(bisect.bisect_right(self.dist, s), len(self.dist) - 1)
        d0, d1 = self.dist[i - 1], self.dist[i]
        t = (s - d0) / (d1 - d0) if d1 > d0 else 0.0
        return self.points[i - 1].lerp(self.points[i], t)

    def heading(self, s, d=0.25):
        """Direction of travel at s, in radians from +X."""
        v = self.point(s + d) - self.point(s - d)
        return math.atan2(v.y, v.x)

    def curvature(self, s, window=STEER_SMOOTHING):
        """Bend at s in 1/metres, averaged over `window` metres; positive bends left."""
        a, b = max(s - window / 2, 0.0), min(s + window / 2, self.length)
        return _wrap(self.heading(b) - self.heading(a)) / (b - a) if b > a else 0.0


def drive(car, curve_obj, frame_start, frame_end, distance_at):
    """Keyframe the car along a Bezier curve: one key per frame on position, heading, wheel spin and
    steering. `distance_at(frame)` gives how far along the curve (metres) the rear axle is."""
    path = CurvePath(curve_obj)
    prev = None
    for frame in range(frame_start, frame_end + 1):
        s = distance_at(frame)
        if not 0.0 <= s <= path.length:
            raise ValueError(f"frame {frame}: {s:.2f} m is off the {path.length:.2f} m curve {curve_obj.name}")
        h = path.heading(s)
        if prev is not None:  # keep the angle continuous so the car never spins round between keys
            h = prev + _wrap(h - prev)
        prev = h
        car.root.location = path.point(s)
        car.root.rotation_euler = (0.0, 0.0, h + math.pi / 2)  # the model's front is -Y
        car.root.keyframe_insert("location", frame=frame)
        car.root.keyframe_insert("rotation_euler", frame=frame)

        kappa = path.curvature(s)
        for key in car.steer:
            car.spin[key].rotation_euler.x = s / car.wheel_radius
            car.spin[key].keyframe_insert("rotation_euler", index=0, frame=frame)
            if key[0] == "F":  # inside wheel turns a little more than the outside one (Ackermann)
                side = 1.0 if key[1] == "L" else -1.0
                car.steer[key].rotation_euler.z = math.atan2(
                    car.wheelbase * kappa, 1.0 - side * kappa * car.front_track / 2)
                car.steer[key].keyframe_insert("rotation_euler", index=2, frame=frame)
    print(f"Drive: {curve_obj.name} ({path.length:.1f} m), frames {frame_start}-{frame_end}, "
          f"{distance_at(frame_start):.1f} m to {distance_at(frame_end):.1f} m")
