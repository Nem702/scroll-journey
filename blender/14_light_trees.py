r"""Handoff 14: light trees, the same look for a fraction of the memory.

The porch scene's trees are Poly Haven's pine_tree_01 and fir_tree_01, three variants each. 05 uses the full models by
the porch (24 M triangles in all) and remakes 120k-triangle copies for the valley on every run (about 2 minutes). This
script makes, once, for every variant:
- a light version for the trees by the porch, under the variant's own name. Each needle (or needle pair) becomes one flat
  strip with the same length and surface area, so it covers the same share of a pixel (checked with rays cast through
  both from the side), and with the needle's own average colour and roughness, measured from its texture; the twig stems
  are decimated to half; bark, trunks and dead branches stay as they are;
- the valley copy (<name>_mid), made by 05's own decimation exactly as before, kept as the fallback;
- a far version for the valley and the treetops in the mist (<name>_far), from that copy. Most of its one- and
  two-triangle needle remnants are dropped and the rest enlarged so the tree covers as much as the copy; its coarser
  twigs and trunk stay as they are.
They're saved beside the originals as assets-source/models/<id>/<id>_light.blend: our own files, no scripts inside; 05
only appends objects from them. 05 uses them when its LIGHT_TREES switch is on (off by default).

Run headless from the repo root (Blender isn't on PATH, so call it by full path):
    & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender\14_light_trees.py -- --make
    ... -P blender\14_light_trees.py -- --still 1 [--no-mist] [--valley mid]
    ... -P blender\14_light_trees.py -- --still 69

--make makes the light trees that are missing (delete a file to make it again): a few minutes, ~12 GB at its peak.
--still renders one of 11's frames full size with the light trees, set up and saved exactly as 11's masters, into
renders/review/14/ (14-f0001-light.png, ...). It copies the matching master beside it (master-11 is only read), and cuts
100% crops of the porch trees and the valley from both into renders/review/14/crops/, with a side-by-side sheet.
--no-mist hides the mist and haze; --valley mid uses the saved valley copies instead of the far version. Each run
replaces its own section of renders/review/14/14-summary.txt. Nothing else is written outside renders/review/14/ and the
two _light.blend files.
"""

import argparse
import hashlib
import importlib.util
import math
import re
import shutil
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "renders" / "review" / "14"
CROPS = REVIEW / "crops"
SUMMARY = REVIEW / "14-summary.txt"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.dont_write_bytecode = True  # no __pycache__ next to the scripts
# 11: the frame table and the scene its masters were rendered with. It loads 09, 10, 08 and 05 as modules (no main()
# runs), and 08 switches off 05's downloader, so nothing here touches the network.
frames11 = _load("night_frames", "11_night_frames.py")
kf, porch = frames11.kf, frames11.porch
kf.OUT = porch.OUT = REVIEW  # where 08's and 05's helpers would write by default: never 11's folder from here
MASTERS = frames11.MASTERS  # renders/time/master-11/, only read

# The light version, for the trees by the porch.
NEEDLE_MAX_VERTS = 25  # needle-material parts this small are single needles or needle pairs (stems have 26 and more)
NEEDLE_MIN_ASPECT = 4.0  # ... and at least this many times longer than wide (buds aren't)
STEM_KEEP = 0.5  # Decimate ratio for the twig stems and buds: keeps 100-102% of their surface (probe, see the plan)

# The far version, for the valley and the treetops in the mist, made from the valley copy.
FAR_SMALL_TRIS = 2  # the copy's needle remnants: parts of one or two triangles, about 75% of its triangles
FAR_KEEP = 0.08  # the share of them kept; the kept ones are enlarged to cover as much as all of them did
AREA_TOLERANCE = 0.03  # a decimation that changes the surface by more than this share isn't used
COVERAGE_MATCH = 0.005  # the far version's coverage is tuned to within this share of the copy's
SEED = 14

# Coverage check: rays cast through a tree from the side, 10 degrees down (about how the porch sees the valley), from
# eight directions, this far apart (m).
RAY_DOWN = 10.0
RAY_STEP = {"near": 0.05, "far": 0.08}

# 100% crops, (x0, y0, x1, y1) from the top-left of the 2560x1440 frame: the trees by the porch (beside the sun at golden
# hour), the treetops in the mist and the forested ridge about 4 km out.
CROP_BOXES = {"porch-trees": (1800, 140, 2560, 800), "valley-treetops": (200, 640, 1150, 1000),
              "valley-hill": (380, 400, 1280, 720)}
LUMA = np.array([0.2126, 0.7152, 0.0722])
COLOR_PROP = {"BYTE_COLOR": "color_srgb", "FLOAT_COLOR": "color"}  # byte colours as stored, no conversion


# ---------------------------------------------------------------------------------------------
# Meshes as arrays. A "piece" is a dict: co (vertex positions), start/size (each face's corners), cv (corner vertices),
# uv and cols (per corner), mat, smooth and area (per face), used (its vertices' indices in the mesh it came from).

def read_mesh(mesh):
    nv, nf, nc = len(mesh.vertices), len(mesh.polygons), len(mesh.loops)
    m = {"co": np.empty(nv * 3, np.float32), "start": np.empty(nf, np.int32), "size": np.empty(nf, np.int32),
         "cv": np.empty(nc, np.int32), "mat": np.empty(nf, np.int32), "smooth": np.empty(nf, bool),
         "area": np.empty(nf, np.float32), "uv": np.empty(nc * 2, np.float32), "cols": {}}
    mesh.vertices.foreach_get("co", m["co"])
    for key, prop in (("start", "loop_start"), ("size", "loop_total"), ("mat", "material_index"),
                      ("smooth", "use_smooth"), ("area", "area")):
        mesh.polygons.foreach_get(prop, m[key])
    mesh.loops.foreach_get("vertex_index", m["cv"])
    mesh.uv_layers[0].data.foreach_get("uv", m["uv"])
    if nf and not np.array_equal(m["start"], np.cumsum(m["size"]) - m["size"]):
        raise SystemExit(f"{mesh.name}: faces' corners aren't in order")
    for a in mesh.color_attributes:
        values = np.empty(len(a.data) * 4, np.float32)
        a.data.foreach_get(COLOR_PROP[a.data_type], values)
        values = values.reshape(-1, 4)
        m["cols"][a.name] = values if a.domain == "CORNER" else values[m["cv"]]
    m["co"], m["uv"] = m["co"].reshape(-1, 3), m["uv"].reshape(-1, 2)
    m["used"] = np.arange(nv)
    return m


def corners_of(start, size):
    """The corner indices of faces with these starts and sizes, face after face."""
    start, size = start.astype(np.int64), size.astype(np.int64)
    return np.repeat(start - (np.cumsum(size) - size), size) + np.arange(size.sum())


def subset(m, keep):
    """The faces where keep is True, as a piece with its own vertices."""
    faces = np.nonzero(keep)[0]
    size = m["size"][faces]
    corners = corners_of(m["start"][faces], size)
    used, cv = np.unique(m["cv"][corners], return_inverse=True)
    return {"co": m["co"][used], "start": (np.cumsum(size) - size).astype(np.int32), "size": size,
            "cv": cv.astype(np.int32), "uv": m["uv"][corners], "cols": {k: v[corners] for k, v in m["cols"].items()},
            "mat": m["mat"][faces], "smooth": m["smooth"][faces], "area": m["area"][faces], "used": used}


def build_mesh(name, pieces, like):
    """One mesh from pieces, with like's materials, UV map name and colour attributes."""
    offsets = np.cumsum([0] + [len(p["co"]) for p in pieces[:-1]])
    size = np.concatenate([p["size"] for p in pieces]).astype(np.int32)
    cv = np.concatenate([p["cv"] + o for p, o in zip(pieces, offsets, strict=True)]).astype(np.int32)
    co = np.concatenate([p["co"] for p in pieces]).astype(np.float32)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(co))
    mesh.vertices.foreach_set("co", co.ravel())
    mesh.loops.add(len(cv))
    mesh.loops.foreach_set("vertex_index", cv)
    mesh.polygons.add(len(size))
    mesh.polygons.foreach_set("loop_start", np.cumsum(size) - size)
    mesh.update(calc_edges=True)
    for mat in like.materials:
        mesh.materials.append(mat)
    mesh.polygons.foreach_set("material_index", np.concatenate([p["mat"] for p in pieces]).astype(np.int32))
    mesh.polygons.foreach_set("use_smooth", np.concatenate([p["smooth"] for p in pieces]))
    uv = mesh.uv_layers.new(name=like.uv_layers[0].name)
    uv.data.foreach_set("uv", np.concatenate([p["uv"] for p in pieces]).astype(np.float32).ravel())
    for a in like.color_attributes:
        attr = mesh.color_attributes.new(a.name, a.data_type, "CORNER")
        attr.data.foreach_set(COLOR_PROP[a.data_type],
                              np.concatenate([p["cols"][a.name] for p in pieces]).astype(np.float32).ravel())
    # The needle and bark materials read the default colour attribute (white): keep the same one.
    mesh.color_attributes.active_color_name = like.color_attributes.active_color_name
    mesh.color_attributes.default_color_name = like.color_attributes.default_color_name
    if mesh.validate():
        raise SystemExit(f"{name}: Blender had to repair the mesh")
    return mesh


def triangles(m):
    return int((m["size"] - 2).sum())


def mesh_triangles(mesh):
    size = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_total", size)
    return int((size - 2).sum())


def parts(piece):
    """A part number for each vertex: vertices joined by edges share a part."""
    nxt = np.arange(len(piece["cv"])) + 1
    nxt[piece["start"] + piece["size"] - 1] = piece["start"]
    a, b = piece["cv"], piece["cv"][nxt]
    label = np.arange(len(piece["co"]), dtype=np.int32)
    while True:
        low = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new[new]]  # follow the labels' own labels: long stems settle in a few rounds
        if np.array_equal(new, label):
            return np.unique(label, return_inverse=True)[1]
        label = new


def twig_slot(mesh):
    return next(i for i, m in enumerate(mesh.materials) if "twig" in m.name)


def per_vertex(piece, values):
    out = np.empty((len(piece["co"]), values.shape[1]), values.dtype)
    out[piece["cv"]] = values
    return out


def decimated(piece, ratio, like):
    """The piece through Blender's Decimate (collapse) at ratio. Returns it and the share of its surface kept."""
    mesh = build_mesh("decimate_in", [piece], like)
    obj = bpy.data.objects.new("decimate_in", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.modifiers.new("decimate", "DECIMATE").ratio = ratio
    out = bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    result = read_mesh(out)
    bpy.data.objects.remove(obj)
    for m in (mesh, out):
        bpy.data.meshes.remove(m)
    return result, float(result["area"].sum() / max(piece["area"].sum(), 1e-12))


def decimated_if_kept(piece, ratio, like):
    """The piece decimated at ratio if that keeps its surface within AREA_TOLERANCE, else unchanged. Returns the piece
    and the share of the surface it has."""
    if ratio < 1.0:
        result, kept = decimated(piece, ratio, like)
        if abs(kept - 1.0) <= AREA_TOLERANCE:
            return result, kept
    return piece, 1.0


def coverage(obj, step):
    """The share of rays that hit obj: rays RAY_DOWN degrees down from eight directions, step apart, over its box."""
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    box = np.array([v[:] for v in obj.bound_box])
    lo, hi = box.min(0), box.max(0)
    centre = Vector((lo + hi) / 2)
    radius = float(np.linalg.norm(hi[:2] - lo[:2])) / 2
    down, back = math.radians(RAY_DOWN), radius + 30.0
    hits = total = 0
    for k in range(8):
        az = k * math.pi / 4
        d = Vector((math.cos(az) * math.cos(down), math.sin(az) * math.cos(down), -math.sin(down)))
        side = Vector((-math.sin(az), math.cos(az), 0.0))
        for u in np.arange(-radius, radius, step):
            origin = centre + side * float(u) - d * back
            for z in np.arange(lo[2] - 1.0, hi[2] + 1.0, step):
                origin.z = float(z) + back * math.sin(down)  # the ray passes height z at the tree's centre
                total += 1
                hits += bvh.ray_cast(origin, d)[0] is not None
    return hits / total


def shown(path):
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def mesh_hash(mesh):
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    cv = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", cv)
    return hashlib.sha1(co.tobytes() + cv.tobytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------
# The light version (trees by the porch) and the far version (valley).

def light_version(obj):
    """obj's light mesh: needles as strips, stems decimated, bark, trunk and dead branches as they are."""
    mesh = obj.data
    m = read_mesh(mesh)
    normals = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertex_normals.foreach_get("vector", normals)
    twig = twig_slot(mesh)
    rest, tw = subset(m, m["mat"] != twig), subset(m, m["mat"] == twig)
    del m
    co = tw["co"].astype(np.float64)
    part = parts(tw)
    n = part.max() + 1
    count = np.bincount(part, minlength=n).astype(np.float64)
    centre = np.stack([np.bincount(part, co[:, k], n) for k in range(3)], 1) / count[:, None]
    d = co - centre[part]
    cov = np.empty((n, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            cov[:, i, j] = cov[:, j, i] = np.bincount(part, d[:, i] * d[:, j], n) / count
    vecs = np.linalg.eigh(cov)[1]  # columns: the part's axes, least spread first
    # Each part's plane: its mean normal (the stored normals are the smooth ones), or its flattest axis.
    nv = normals.reshape(-1, 3)[tw["used"]].astype(np.float64)
    nrm = np.stack([np.bincount(part, nv[:, k], n) for k in range(3)], 1)
    norm = np.linalg.norm(nrm, axis=1)
    nrm = np.where(norm[:, None] > 1e-6, nrm / np.maximum(norm, 1e-12)[:, None], vecs[:, :, 0])
    axis = vecs[:, :, 2] - (vecs[:, :, 2] * nrm).sum(1)[:, None] * nrm  # its length, in that plane
    axis /= np.maximum(np.linalg.norm(axis, axis=1), 1e-12)[:, None]
    side = np.cross(nrm, axis)
    t = (d * axis[part]).sum(1)  # along the part
    order = np.lexsort((t, part))
    first = np.searchsorted(part[order], np.arange(n))
    base, tip = order[first], order[np.append(first[1:], len(order)) - 1]
    length = t[tip] - t[base]
    area = np.bincount(part[tw["cv"][tw["start"]]], tw["area"], n)
    width = area / np.maximum(length, 1e-9)
    needle = (count <= NEEDLE_MAX_VERTS) & (length > 1e-4) & (length >= NEEDLE_MIN_ASPECT * width)
    ids = np.nonzero(needle)[0]
    # One strip per needle, in its plane: from its base to its tip along its length, as wide as its surface over its
    # length, so it covers as much of a pixel.
    half = (width[ids] / 2)[:, None] * side[ids]
    at_base = centre[ids] + t[base[ids]][:, None] * axis[ids]
    at_tip = centre[ids] + t[tip[ids]][:, None] * axis[ids]
    strip_co = np.stack([at_base - half, at_tip - half, at_tip + half, at_base + half], 1).reshape(-1, 3)
    # Its colour: each needle's texture is a thin needle drawn on black, and how much of the black a strip catches
    # changes how dark it reads. So each strip takes one texel of its needle (all four corners on it), and as vertex
    # colour the dimming that turns that texel into the needle's own average colour (the material multiplies the two).
    face_part = part[tw["cv"][tw["start"]]]
    pick_uv, tint, colour_stats = needle_colours(tw, face_part, ids, needle_textures(mesh.materials[twig]),
                                                 mesh.color_attributes.default_color_name)
    cols = {k: np.repeat(per_vertex(tw, v)[base[ids]], 4, 0) for k, v in tw["cols"].items()}
    cols[mesh.color_attributes.default_color_name][:, :3] = np.repeat(linear_to_srgb(tint), 4, 0)
    strips = {"co": strip_co, "start": np.arange(0, 4 * len(ids), 4, dtype=np.int32),
              "size": np.full(len(ids), 4, np.int32), "cv": np.arange(4 * len(ids), dtype=np.int32),
              "uv": np.repeat(pick_uv, 4, 0).astype(np.float32), "cols": cols,
              "mat": np.full(len(ids), twig, np.int32), "smooth": np.ones(len(ids), bool),
              "area": (length * width)[ids].astype(np.float32)}
    strip_kept = float(strips["area"].sum() / area[ids].sum())
    stems_in = subset(tw, ~needle[face_part])
    stems, stem_kept = decimated_if_kept(stems_in, STEM_KEEP, mesh)
    light = build_mesh(obj.name + "_light", [rest, stems, strips], mesh)
    return light, {"needles": len(ids), "needle_tris": int(triangles(tw) - triangles(stems_in)),
                   "strip_tris": triangles(strips), "strip_kept": strip_kept, "needle_area": float(area[ids].sum()),
                   "stems_in": triangles(stems_in), "stems": triangles(stems), "stem_kept": stem_kept,
                   "rest": triangles(rest), **colour_stats}


# Six points inside a triangle (barycentric) whose plain average is exact for colours varying up to cubically across it.
_A, _B, _C = 0.659027622374092, 0.231933368553031, 0.109039009072877
TRIANGLE_POINTS = np.array([(_A, _B, _C), (_A, _C, _B), (_B, _A, _C), (_B, _C, _A), (_C, _A, _B), (_C, _B, _A)])


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0.0), 1 / 2.4) - 0.055)


def pixels(image):
    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    return px.reshape(h, w, 4)  # rows from the bottom


def needle_textures(material):
    """The needle material's base colour texture (the image it multiplies by the vertex colour), as linear RGB, and
    its roughness (the green channel of the packed texture feeding Roughness). Stops if the material isn't built that
    way (05's imported needle materials are)."""
    nt = material.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    mix = bsdf.inputs["Base Color"].links[0].from_node
    feeds = [link.from_node for link in nt.links if link.to_node == mix]
    tex = next((f for f in feeds if f.type == "TEX_IMAGE"), None)
    split = bsdf.inputs["Roughness"].links[0].from_node if bsdf.inputs["Roughness"].is_linked else None
    rough = split and next((lk.from_node for lk in nt.links if lk.to_node == split), None)
    if (mix.type != "MIX" or mix.blend_type != "MULTIPLY" or mix.inputs[0].is_linked or mix.inputs[0].default_value != 1
            or tex is None or not any(f.type == "VERTEX_COLOR" and f.layer_name == "" for f in feeds)
            or tex.inputs["Vector"].is_linked or tex.image.colorspace_settings.name != "sRGB"
            or split is None or split.type != "SEPARATE_COLOR" or bsdf.inputs["Roughness"].links[0].from_socket.name
            != "Green" or rough is None or rough.type != "TEX_IMAGE" or rough.inputs["Vector"].is_linked):
        raise SystemExit(f"{material.name}: not built as 05's imported needle materials are")
    return srgb_to_linear(pixels(tex.image)[..., :3]), pixels(rough.image)[..., 1:2]


def sample(tex, uv):
    """Bilinear texture lookups at uv (repeating, as Blender's image texture)."""
    h, w = tex.shape[:2]
    x, y = uv[:, 0] * w - 0.5, uv[:, 1] * h - 0.5
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, y0 = x0.astype(np.int64) % w, y0.astype(np.int64) % h
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    return ((tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy)
            + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy)


def needle_colours(tw, face_part, ids, textures, colour_name, chunk=1_000_000):
    """For each needle: its average base colour (texture times vertex colour, linear) and roughness over its surface;
    the UV of its sample that can be dimmed to that colour and is nearest that roughness; and the dimming."""
    tex, rough_tex = textures
    rank = np.full(face_part.max() + 1, -1)
    rank[ids] = np.arange(len(ids))
    faces = np.nonzero(rank[face_part] >= 0)[0]
    if not np.all(tw["size"][faces] == 3):
        raise SystemExit("needle faces aren't all triangles")
    corners = tw["start"][faces][:, None] + np.arange(3)
    needle_of, weight = rank[face_part[faces]], tw["area"][faces].astype(np.float64)
    vertex_colour = srgb_to_linear(tw["cols"][colour_name][corners][..., :3]).mean(1)
    n = len(ids)

    def samples(sl):  # each sample point's UV (faces, points, 2), colour and roughness (faces, points, 3 and 1)
        pts = np.einsum("pk,fkd->fpd", TRIANGLE_POINTS, tw["uv"][corners[sl]])
        flat = pts.reshape(-1, 2)
        return (pts, sample(tex, flat).reshape(len(pts), -1, 3) * vertex_colour[sl][:, None],
                sample(rough_tex, flat).reshape(len(pts), -1, 1))

    total = np.zeros((n, 4))
    for i in range(0, len(faces), chunk):
        sl = slice(i, i + chunk)
        _, col, rough = samples(sl)
        both = np.concatenate([col, rough], 2).mean(1) * weight[sl][:, None]
        for k in range(4):
            total[:, k] += np.bincount(needle_of[sl], both[:, k], n)
    mean = total / np.bincount(needle_of, weight, n)[:, None]  # colour (3), roughness
    best, pick = np.full(n, np.inf), np.zeros((n, 2))
    for i in range(0, len(faces), chunk):
        sl = slice(i, i + chunk)
        pts, col, rough = samples(sl)
        who = np.repeat(needle_of[sl], pts.shape[1])
        reach = (col.reshape(-1, 3) / np.maximum(mean[who, :3], 1e-6)).min(1)  # below 1: can't be dimmed to it
        cost = np.abs(rough.reshape(-1) - mean[who, 3]) + 10.0 * np.maximum(1.0 - reach, 0.0)
        order = np.lexsort((cost, who))
        head = np.append(0, np.nonzero(np.diff(who[order]))[0] + 1)  # each needle's cheapest sample in this chunk
        top, at = who[order[head]], order[head]
        better = cost[at] < best[top]
        best[top[better]] = cost[at[better]]
        pick[top[better]] = pts.reshape(-1, 2)[at[better]]
    texel = sample(tex, pick)
    tint = np.clip(mean[:, :3] / np.maximum(texel, 1e-6), 0.0, 1.0)
    share = np.bincount(needle_of, weight, n)
    share /= share.sum()
    stats = {"albedo": float(mean[:, :3] @ LUMA @ share), "albedo_strips": float((texel * tint) @ LUMA @ share),
             "rough": float(mean[:, 3] @ share), "rough_strips": float(sample(rough_tex, pick)[:, 0] @ share),
             "clipped": float((texel < mean[:, :3] - 1e-6).any(1).mean())}
    return pick, tint, stats


def far_version(mid, target):
    """The far version of a valley copy: most of its needle remnants dropped and the kept ones enlarged, so the tree
    covers what the copy does (target, from coverage()). Its coarser twigs and trunk stay as they are (Decimate changed
    their surface by far more than AREA_TOLERANCE)."""
    mesh = mid.data
    m = read_mesh(mesh)
    twig = twig_slot(mesh)
    other, tw = subset(m, m["mat"] != twig), subset(m, m["mat"] == twig)
    part = parts(tw)
    face_part = part[tw["cv"][tw["start"]]]
    n = part.max() + 1
    small = np.bincount(face_part, minlength=n) <= FAR_SMALL_TRIS
    keep = np.random.default_rng(SEED).random(n) < FAR_KEEP
    big = subset(tw, ~small[face_part])
    kept = subset(tw, (small & keep)[face_part])
    kept_part = part[kept["used"]]
    count = np.maximum(np.bincount(kept_part, minlength=n), 1)[:, None]
    centre = np.stack([np.bincount(kept_part, kept["co"][:, k], n) for k in range(3)], 1) / count
    grow = math.sqrt(tw["area"][small[face_part]].sum() / kept["area"].sum())  # keeps the remnants' surface

    def build(factor):
        piece = dict(kept, co=centre[kept_part] + (kept["co"] - centre[kept_part]) * grow * math.sqrt(factor))
        far_mesh = build_mesh(mid.name.replace("_mid", "_far"), [other, big, piece], mesh)
        obj = bpy.data.objects.new(far_mesh.name, far_mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj, coverage(obj, RAY_STEP["far"])

    # Fewer, bigger pieces overlap less, so the same surface covers a little more: tune the surface to the copy's cover.
    tries = []
    factor = 1.0
    for _ in range(5):
        obj, cov = build(factor)
        tries.append((abs(cov / target - 1), factor, cov, obj))
        if tries[-1][0] < COVERAGE_MATCH:
            break
        if len(tries) == 1:
            factor *= (target / cov) ** 2
        else:
            (_, f1, c1, _), (_, f2, c2, _) = tries[-2:]
            factor = f2 + (target - c2) * (f2 - f1) / (c2 - c1) if c2 != c1 else f2
        factor = min(max(factor, 0.2), 5.0)
    tries.sort(key=lambda x: x[0])
    for _, _, _, obj in tries[1:]:
        mesh_ = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh_)
    _, factor, cov, far = tries[0]
    far.name = far.data.name = mid.name.removesuffix("_mid") + "_far"  # the tries took .001 and so on
    return far, {"remnants": int(small.sum()), "remnants_kept": int((small & keep).sum()),
                 "grow": grow * math.sqrt(factor), "big": triangles(big), "other": triangles(other), "coverage": cov}


# ---------------------------------------------------------------------------------------------
# Runs.

def make(log):
    """The light trees that are missing. Returns True if it made any."""
    made = False
    for asset in porch.TREE_ASSETS:
        path = porch.light_tree_path(asset)
        if path.exists():
            log(f"{asset}: {shown(path)} is already made (delete it to make it again)")
            continue
        made = True
        started = time.time()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        coll = bpy.data.collections.new("trees")
        bpy.context.scene.collection.children.link(coll)
        full = porch.import_variants([asset], coll)
        log(f"{asset}: imported {len(full)} variants in {time.time() - started:.0f} s")
        depsgraph = bpy.context.evaluated_depsgraph_get()
        saved = []
        for obj in full:
            t = time.time()
            full_tris = mesh_triangles(obj.data)
            mid = porch.decimated_copy(obj, depsgraph)
            coll.objects.link(mid)
            mid_tris, mid_verts, mid_hash = mesh_triangles(mid.data), len(mid.data.vertices), mesh_hash(mid.data)
            cov_full = coverage(obj, RAY_STEP["near"])
            light, ls = light_version(obj)
            old, obj.data = obj.data, light
            bpy.data.meshes.remove(old)
            light_tris = mesh_triangles(light)
            cov_light = coverage(obj, RAY_STEP["near"])
            cov_mid = coverage(mid, RAY_STEP["far"])
            far, fs = far_version(mid, cov_mid)
            far_tris = mesh_triangles(far.data)
            log(f"{obj.name}: full {full_tris:,} triangles -> light {light_tris:,} ({light_tris / full_tris:.0%}). "
                f"Needles: {ls['needles']:,} strips ({ls['needle_tris']:,} -> {ls['strip_tris']:,} triangles, surface "
                f"{ls['needle_area']:.1f} m2, {ls['strip_kept']:.1%} kept); stems {ls['stems_in']:,} -> {ls['stems']:,} "
                f"({ls['stem_kept']:.1%} of their surface); bark, trunk and dead branches {ls['rest']:,} as they "
                f"were. Needle colour (mean albedo): needles {ls['albedo']:.4f}, strips {ls['albedo_strips']:.4f}; mean "
                f"roughness {ls['rough']:.3f}, strips {ls['rough_strips']:.3f}; "
                f"{ls['clipped']:.1%} of strips can't be dimmed all the way. Coverage from the side: full "
                f"{cov_full:.4f}, light {cov_light:.4f} "
                f"({cov_light / cov_full:.1%}).")
            log(f"{obj.name}: valley copy (05's decimation) {mid_tris:,} triangles, {mid_verts:,} vertices, hash "
                f"{mid_hash} -> far {far_tris:,} ({far_tris / mid_tris:.0%}). Needle remnants {fs['remnants']:,} -> "
                f"{fs['remnants_kept']:,}, enlarged {fs['grow']:.2f}x; coarser twigs {fs['big']:,} and bark and trunk "
                f"{fs['other']:,} as they were. Coverage: copy {cov_mid:.4f}, far {fs['coverage']:.4f} "
                f"({fs['coverage'] / cov_mid:.1%}). "
                f"{time.time() - t:.0f} s")
            saved += [obj, mid, far]
        names = {o.name for o in saved}
        expected = {o.name for o in full} | {o.name + "_mid" for o in full} | {o.name + "_far" for o in full}
        if names != expected:
            raise SystemExit(f"{asset}: unexpected object names {sorted(names ^ expected)}")
        part = path.with_name(path.stem + "_part.blend")  # renamed once complete: a cut-off run never leaves a broken one
        bpy.data.libraries.write(str(part), set(saved), path_remap="NONE", compress=False)
        part.replace(path)
        log(f"{asset}: wrote {shown(path)} ({path.stat().st_size / 2**20:,.0f} MB) in "
            f"{time.time() - started:.0f} s; peak RAM so far {porch.peak_ram_gb():.1f} GB")
    return made


def render_still(number, mist, valley, log):
    """One of 11's frames, full size, with the light trees. Returns the summary section's name."""
    by_number = {int(f.name): f for f in frames11.frame_table()}
    if number not in by_number:
        raise SystemExit(f"No frame {number}: the frames are 1 to {len(by_number)}")
    frame = by_number[number]
    porch.LIGHT_TREES, porch.VALLEY_TREES = True, valley
    name = f"14-f{frame.name}-light" + ("" if valley == "far" else f"-valley-{valley}") + ("" if mist else "-nomist")
    started = time.time()
    lamps = frames11.measure_lamps([frame])
    scene, extras = frames11.build(True, False, porch.FINAL_SAMPLES, log)
    if not mist:
        bpy.data.objects["atmosphere"].hide_render = True  # 05's one box of mist and haze
    log(f"Scene built in {time.time() - started:.0f} s (peak RAM so far {porch.peak_ram_gb():.1f} GB); "
        f"mist and haze {'on' if mist else 'off'}; valley trees: {valley}")
    frames11.show_frame(scene, extras, frame, lamps[frame.name])
    seconds, peak_mb = kf.render(scene)
    path = REVIEW / f"{name}.png"
    kf.save_render(scene, path, "PNG")  # 8-bit, as 11's masters
    log(f"{name} | {frames11.frame_line(frame, lamps[frame.name], seconds, peak_mb)}")
    master = REVIEW / f"14-f{frame.name}-master.png"
    if not master.exists():
        shutil.copyfile(MASTERS / f"{frame.name}.png", master)
    if mist:
        crops(name.replace("-light", ""), path, master, log)  # 14-f0001-porch-trees-light.png, ...  14-f0001-crops.png
    return name.removeprefix("14-")


def crops(name, light_path, master_path, log):
    """100% crops of CROP_BOXES from the master and the light still, and a sheet with each pair side by side."""
    CROPS.mkdir(exist_ok=True)
    master, light = (kf.load(p)[::-1, :, :3] for p in (master_path, light_path))  # rows from the top
    rows = []
    for region, (x0, y0, x1, y1) in CROP_BOXES.items():
        pair = [img[y0:y1, x0:x1] for img in (master, light)]
        for label, img in zip(("master", "light"), pair, strict=True):
            kf.write_png(img[::-1], CROPS / f"{name}-{region}-{label}.png")
        lum = [float((img @ LUMA).mean()) for img in pair]
        log(f"Crop {region}: mean brightness master {lum[0]:.4f}, light {lum[1]:.4f} ({lum[1] / lum[0] - 1:+.2%})")
        gap = np.full((y1 - y0, 24, 3), 0.06, np.float32)
        rows.append(np.concatenate([pair[0], gap, pair[1]], 1))
    width = max(r.shape[1] for r in rows)
    sheet = [np.full((24, width, 3), 0.06, np.float32)]
    for r in rows:
        sheet += [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=0.06),
                  np.full((24, width, 3), 0.06, np.float32)]
    sheet = np.concatenate(sheet, 0)[::-1]  # rows from the bottom, as Blender stores them; the label goes on top
    label = kf.label("left: master-11   ·   right: light trees   ·   100%", width)[..., :3]
    kf.write_png(np.concatenate([sheet, label], 0), REVIEW / f"{name}-crops.png")


def update_summary(key, lines):
    """Replace this run's section of the summary, keeping the others."""
    sections = {}
    if SUMMARY.exists():
        for block in re.split(r"(?m)^## ", SUMMARY.read_text(encoding="utf-8"))[1:]:
            head, _, body = block.partition("\n")
            sections[head.strip()] = body.strip("\n")
    sections[key] = "\n".join(lines)
    order = sorted(sections, key=lambda k: (not k.startswith("make"), k))
    SUMMARY.write_text("Light trees (handoff 14): one section per run, replaced when that run is done again.\n\n"
                       + "\n\n".join(f"## {k}\n{sections[k]}" for k in order) + "\n", encoding="utf-8")


def parse_args(argv):
    p = argparse.ArgumentParser(prog="14_light_trees.py")
    p.add_argument("--make", action="store_true", help="make the light trees that are missing")
    p.add_argument("--still", type=int, metavar="FRAME", help="render one of 11's frames with the light trees")
    p.add_argument("--no-mist", action="store_true", help="with --still: hide the mist and haze")
    p.add_argument("--valley", choices=("far", "mid"), help="with --still: the valley trees (default: 05's VALLEY_TREES)")
    args = p.parse_args(argv)
    if args.make == (args.still is not None):
        p.error("give --make or --still FRAME")
    if (args.no_mist or args.valley) and args.still is None:
        p.error("--no-mist and --valley go with --still")
    return args


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    REVIEW.mkdir(parents=True, exist_ok=True)
    started = time.time()
    lines = [time.strftime("%Y-%m-%d %H:%M")]

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    if args.make:
        key = "make" if make(log) else "make, run again"
    else:
        key = render_still(args.still, not args.no_mist, args.valley or porch.VALLEY_TREES, log)
    log(f"Run time: {time.time() - started:.0f} s; peak RAM over the run: {porch.peak_ram_gb():.1f} GB")
    update_summary(key, lines)


if __name__ == "__main__":
    main()
