"""Generate a few basic glTF test files (no Blender needed, standard library only).

Each file tests one basic feature. Objects sit in a row along X, at most
3 units wide, facing +Z (glTF is Y-up).

Usage:
    python examples/gltf_tests/make_gltf_tests.py   # -> examples/gltf_tests/generated/*.glb
"""

import json
import math
import os
import struct
import zlib

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
TEX_DIR = os.path.join(OUT_DIR, "textures")  # copies of every texture, for inspection / other renderers


# ---------------------------------------------------------------- PNG textures

def png(w, h, px, rgba=False):
    """Tiny PNG encoder. px(x, y) -> (r, g, b) or (r, g, b, a) in 0..255."""
    raw = b"".join(b"\0" + b"".join(bytes(px(x, y)) for x in range(w)) for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if rgba else 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def tex_quadrants(n=64):
    """Top-left red, top-right green, bottom-left blue, bottom-right yellow."""
    cols = {(0, 0): (255, 0, 0), (1, 0): (0, 255, 0), (0, 1): (0, 0, 255), (1, 1): (255, 255, 0)}
    return png(n, n, lambda x, y: cols[(x * 2 // n, y * 2 // n)])


def tex_checker(n=128, cells=8):
    return png(n, n, lambda x, y: (230, 230, 230) if (x * cells // n + y * cells // n) % 2 else (40, 40, 40))


def tex_alpha_checker(n=128, cells=8):
    """Orange checker whose ALPHA alternates 0.75 / 0.25 (colour is the same in every cell)."""
    return png(n, n, lambda x, y: (255, 128, 0, 191 if (x * cells // n + y * cells // n) % 2 else 64), rgba=True)


# Anisotropy textures (linear data): R,G = direction in tangent space mapped 0..1 -> -1..1,
# B = strength multiplier. (255,128) = along the tangent, (128,255) = along the bitangent.
ANISO_H = (255, 128, 255)
ANISO_V = (128, 255, 255)


def tex_solid(rgb):
    return png(1, 1, lambda x, y: rgb)


def tex_aniso_checker(n=128, cells=8):
    """Cells alternate between horizontal and vertical brushing direction."""
    return png(n, n, lambda x, y: ANISO_H if (x * cells // n + y * cells // n) % 2 else ANISO_V)


def tex_stripes(n=128, stripes=8):
    """White / black horizontal stripes (used as an emissive mask)."""
    return png(n, n, lambda x, y: (255, 255, 255) if (y * stripes // n) % 2 == 0 else (0, 0, 0))


def tex_tiny_4x4():
    """4x4 texels of distinct colours, to be magnified a lot."""
    pal = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255),
           (0, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    return png(4, 4, lambda x, y: pal[(x + 3 * y) % len(pal)])


# Size of the 1-texel checker used by the minification test (18). With the default camera a
# quad is ~235 px wide on screen: 256 gives ~1 texel per pixel (strongest moire); larger values
# average to grey even without mipmaps, smaller ones are no longer minified at all.
PIXEL_CHECKER_SIZE = 256


def tex_pixel_checker(n=PIXEL_CHECKER_SIZE):
    """1-texel black/white checker: averages to grey when properly minified."""
    return png(n, n, lambda x, y: (255, 255, 255) if (x + y) % 2 else (0, 0, 0))


def tex_bumps(n=128, cells=4, directx=False):
    """Tangent-space normal map with round bumps.
    directx=False: OpenGL convention (green = +Y up), which is what glTF requires.
    directx=True:  DirectX convention (green = +Y down), i.e. the green channel is inverted."""
    c = n / cells

    def f(x, y):
        dx, dy = (x % c) / c * 2 - 1, -((y % c) / c * 2 - 1)
        if directx:
            dy = -dy
        r2 = dx * dx + dy * dy
        nz = math.sqrt(0.64 - r2) if r2 < 0.64 else None
        v = (dx * 0.8, dy * 0.8, nz) if nz is not None else (0, 0, 1)
        ln = math.sqrt(sum(a * a for a in v))
        return tuple(int((a / ln * 0.5 + 0.5) * 255) for a in v)

    return png(n, n, f)


# High-resolution normal map built from a height field (test 06d) -------------------------------
HIRES_NORMAL_SIZE = 1024   # texture size in pixels (square)


def height_field_regions(n=HIRES_NORMAL_SIZE):
    """Height field in pixel units, image split into 4 regions:
    top-left: 3x3 raised domes        top-right: 3x3 sunken dents (same shape, inverted)
    bottom-left: 3x3 raised pyramids  bottom-right: fine concentric ripples (detail / resolution)."""
    half, cell = n // 2, n // 6
    R = cell * 0.4                       # radius of domes / dents / pyramids
    A = R * 0.4                          # their height -> max slope ~0.6 (~30 degrees)
    ring = max(n / 64.0, 4.0)            # ripple wavelength in pixels
    h = [[0.0] * n for _ in range(n)]
    for y in range(n):
        row = h[y]
        for x in range(n):
            if y < half or x < half:      # the three 3x3 grid regions (ripples are done below)
                dx = (x % cell) - cell / 2 + 0.5
                dy = (y % cell) - cell / 2 + 0.5
                if y < half:              # domes (x < half) / dents (x >= half)
                    r = math.hypot(dx, dy)
                    v = A * math.cos(math.pi * r / (2 * R)) ** 2 if r < R else 0.0
                    row[x] = v if x < half else -v
                else:                     # pyramids, bottom-left
                    m = max(abs(dx), abs(dy))
                    row[x] = A * (1 - m / R) if m < R else 0.0
        if y >= half:                     # ripples, bottom-right
            for x in range(half, n):
                dx, dy = x - half * 1.5 + 0.5, y - half * 1.5 + 0.5
                r = math.hypot(dx, dy)
                fade = max(0.0, 1 - r / (half * 0.48))
                row[x] = ring / (2 * math.pi) * 0.6 * math.sin(2 * math.pi * r / ring) * fade
    return h


def tex_normal_from_height(h):
    """OpenGL tangent-space normal map (green = up) from a height field, by central differences."""
    n = len(h)

    def f(x, y):
        hx = (h[y][min(x + 1, n - 1)] - h[y][max(x - 1, 0)]) / 2
        hy_up = (h[max(y - 1, 0)][x] - h[min(y + 1, n - 1)][x]) / 2   # image y goes down, glTF +Y is up
        nx, ny, nz = -hx, -hy_up, 1.0
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        return tuple(int(round((a / ln * 0.5 + 0.5) * 255)) for a in (nx, ny, nz))

    return png(n, n, f)


def tex_height_preview(h):
    """The same height field as a grey colour image: low = dark, high = light (visual reference)."""
    n = len(h)
    lo = min(min(r) for r in h)
    hi = max(max(r) for r in h)

    def f(x, y):
        v = int(round(40 + 200 * (h[y][x] - lo) / (hi - lo)))
        return (v, v, v)

    return png(n, n, f)



def tex_grazing_checker(n=64, cell=4):
    """Black/white checker, `cell` x `cell` texels per square (test 18b). n=64 -> 16 squares
    (power of two); n=48 -> 12 squares (NOT a power of two)."""
    return png(n, n, lambda x, y: (235, 235, 235) if (x // cell + y // cell) % 2 else (20, 20, 20))

# ---------------------------------------------------------------- geometry

def sphere(r=0.5, seg=48, rings=24):
    pos, nrm, uv, tan, idx = [], [], [], [], []
    for i in range(rings + 1):
        th = math.pi * i / rings
        for j in range(seg + 1):
            ph = 2 * math.pi * j / seg
            n = (math.sin(th) * math.sin(ph), math.cos(th), math.sin(th) * math.cos(ph))
            pos.append([r * a for a in n]); nrm.append(n); uv.append((j / seg, i / rings))
            tan.append((math.cos(ph), 0, -math.sin(ph), 1))
    for i in range(rings):
        for j in range(seg):
            a = i * (seg + 1) + j
            b = a + seg + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return dict(pos=pos, nrm=nrm, uv=uv, tan=tan, idx=idx)


def quad(size=1.0, uv_scale=1.0):
    h, s = size / 2, uv_scale
    return dict(pos=[(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], nrm=[(0, 0, 1)] * 4,
                uv=[(0, s), (s, s), (s, 0), (0, 0)], tan=[(1, 0, 0, 1)] * 4, idx=[0, 1, 2, 0, 2, 3])



def floor_strip(width, z_near, z_far, u_tiles, v_tiles):
    """Horizontal strip (normal +Y) from z_near (close to the camera) to z_far (far away).
    The texture repeats u_tiles times across and v_tiles times along the strip."""
    h = width / 2
    return dict(pos=[(-h, 0, z_near), (h, 0, z_near), (h, 0, z_far), (-h, 0, z_far)], nrm=[(0, 1, 0)] * 4,
                uv=[(0, 0), (u_tiles, 0), (u_tiles, v_tiles), (0, v_tiles)],
                tan=[(1, 0, 0, 1)] * 4, idx=[0, 1, 2, 0, 2, 3])

def box(sx, sy, sz, center=(0, 0, 0)):
    pos, nrm, idx = [], [], []
    for n, u, v in [((1, 0, 0), (0, 0, -1), (0, 1, 0)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
                    ((0, 1, 0), (1, 0, 0), (0, 0, -1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
                    ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0))]:
        base = len(pos)
        for cu, cv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = [(n[k] + cu * u[k] + cv * v[k]) * s / 2 + c for k, s, c in zip(range(3), (sx, sy, sz), center)]
            pos.append(p); nrm.append(n)
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return dict(pos=pos, nrm=nrm, idx=idx)


# ---------------------------------------------------------------- glTF builder

class Gltf:
    def __init__(self, name):
        self.name = name
        self.pngs = {}  # file name -> PNG bytes, also written to TEX_DIR
        self.d = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}],
                  "nodes": [], "meshes": [], "materials": [], "accessors": [], "bufferViews": []}
        self.bin = bytearray()

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        bv = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.bin += data
        self.d["bufferViews"].append(bv)
        return len(self.d["bufferViews"]) - 1

    def _accessor(self, rows, typ, fmt="f", comp=5126, target=34962):
        flat = [c for r in rows for c in (r if isinstance(r, (list, tuple)) else [r])]
        acc = {"bufferView": self._view(struct.pack("<%d%s" % (len(flat), fmt), *flat), target),
               "componentType": comp, "count": len(rows), "type": typ}
        if typ == "VEC3" and target == 34962:  # min/max is required on POSITION
            acc["min"] = [min(r[i] for r in rows) for i in range(3)]
            acc["max"] = [max(r[i] for r in rows) for i in range(3)]
        self.d["accessors"].append(acc)
        return len(self.d["accessors"]) - 1

    def texture(self, png_bytes, name, sampler=None):
        self.pngs["%s_%s.png" % (self.name, name)] = png_bytes
        self.d.setdefault("images", []).append({"bufferView": self._view(png_bytes), "mimeType": "image/png"})
        tex = {"source": len(self.d["images"]) - 1}
        if sampler is not None:
            self.d.setdefault("samplers", []).append(sampler)
            tex["sampler"] = len(self.d["samplers"]) - 1
        self.d.setdefault("textures", []).append(tex)
        return {"index": len(self.d["textures"]) - 1}

    def material(self, **m):
        self.d["materials"].append(m)
        return len(self.d["materials"]) - 1

    def pbr(self, color=(0.8, 0.8, 0.8, 1), metal=0.0, rough=0.5, **extra):
        pbr = {"baseColorFactor": list(color), "metallicFactor": metal, "roughnessFactor": rough}
        pbr.update({k: extra.pop(k) for k in list(extra) if k.endswith("Texture") and k.startswith(("base", "metallic"))})
        return self.material(pbrMetallicRoughness=pbr, **extra)

    def mesh(self, g, material=None, tangents=False):
        attrs = {"POSITION": self._accessor(g["pos"], "VEC3"), "NORMAL": self._accessor(g["nrm"], "VEC3", target=34962)}
        if "uv" in g:
            attrs["TEXCOORD_0"] = self._accessor(g["uv"], "VEC2")
        if tangents:
            attrs["TANGENT"] = self._accessor(g["tan"], "VEC4")
        prim = {"attributes": attrs, "indices": self._accessor(g["idx"], "SCALAR", "I", 5125, 34963)}
        if material is not None:
            prim["material"] = material
        self.d["meshes"].append({"primitives": [prim]})
        return len(self.d["meshes"]) - 1

    def node(self, mesh, **trs):
        self.d["nodes"].append(dict(mesh=mesh, **trs))
        self.d["scenes"][0]["nodes"].append(len(self.d["nodes"]) - 1)

    def row(self, meshes, spacing=1.2):
        for i, m in enumerate(meshes):
            self.node(m, translation=[(i - (len(meshes) - 1) / 2) * spacing, 0, 0])

    def save(self, path):
        os.makedirs(TEX_DIR, exist_ok=True)
        for fname, data in self.pngs.items():
            with open(os.path.join(TEX_DIR, fname), "wb") as f:
                f.write(data)
        self.d["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.d).encode()
        js += b" " * (-len(js) % 4)
        self.bin += b"\0" * (-len(self.bin) % 4)
        body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(self.bin), b"BIN\0") + bytes(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


# ---------------------------------------------------------------- the tests
#
# Test index (search for "# 06a" etc. to jump to a test). The numbers come from the order of
# the TESTS list at the bottom of the file; keep this index and the banners in sync if you reorder.
#
#   01   Base colour factor (plain colours)
#   02   Metallic factor ramp
#   03   Roughness factor ramp
#   04   Base colour texture: UV orientation
#   05   Base colour texture on a curved surface
#   06a  Normal map: OpenGL convention (glTF standard) + tangent w flip
#   06b  Normal map: DirectX convention (green inverted) + tangent w flip
#   06c  Normal map strength (normalTexture.scale)
#   06d  Normal map: high-res 1024 on a plane, vs height reference
#   07   Emission: factor, KHR_materials_emissive_strength, texture
#   08   Alpha modes: OPAQUE / BLEND / MASK (cutoff)
#   09   Double-sided vs single-sided (back faces)
#   10   Node transforms: scale / rotation / translation
#   11   Axis orientation (Y-up -> Z-up)
#   12   Anisotropy strength
#   13   Anisotropy rotation
#   14   Anisotropy direction texture
#   15   Anisotropy strength from texture (blue channel)
#   16   Anisotropy: TANGENT given / missing / funky (kept or recalculated?)
#   17   Sampler wrap modes
#   18a  Sampler filtering (magnification / mipmaps)
#   18b  Mipmapping at a grazing angle (minFilter modes, default sampler, NPOT texture)
#   19   Metallic-roughness texture vs factors (red ignored)
#   20   Metallic-roughness texture channels (G = rough, B = metal)
#   21   Occlusion texture strength
#   22   Occlusion + metallic-roughness merge (Nyx ARM texture)
#   23   All basic features together
#
# Each function fills a Gltf. The function name becomes the file / image name.

# ================================================================================================
# 01  | Base colour factor (plain colours)
# ================================================================================================
def base_color_red_green_blue(g):
    g.row([g.mesh(sphere(), g.pbr(c)) for c in ((0.8, 0.05, 0.05, 1), (0.05, 0.8, 0.05, 1), (0.05, 0.05, 0.8, 1))])


# ================================================================================================
# 02  | Metallic factor ramp
# ================================================================================================
def metallic_0_05_1(g):
    """Same roughness (0.2), metallic 0 -> 0.5 -> 1 left to right."""
    g.row([g.mesh(sphere(), g.pbr((1, 0.77, 0.34, 1), m, 0.2)) for m in (0.0, 0.5, 1.0)])


# ================================================================================================
# 03  | Roughness factor ramp
# ================================================================================================
def roughness_0_05_1(g):
    """Metallic 1, roughness 0 -> 0.5 -> 1 left to right."""
    g.row([g.mesh(sphere(), g.pbr((1, 0.77, 0.34, 1), 1.0, r)) for r in (0.0, 0.5, 1.0)])


# ================================================================================================
# 04  | Base colour texture: UV orientation
# ================================================================================================
def base_color_texture_quadrants(g):
    """Front-facing quad: top-left red, top-right green, bottom-left blue, bottom-right yellow."""
    g.node(g.mesh(quad(), g.pbr((1, 1, 1, 1), baseColorTexture=g.texture(tex_quadrants(), "quadrants"))))


# ================================================================================================
# 05  | Base colour texture on a curved surface
# ================================================================================================
def base_color_texture_on_sphere(g):
    """Checker texture on a sphere, plain white factor."""
    g.node(g.mesh(sphere(), g.pbr((1, 1, 1, 1), baseColorTexture=g.texture(tex_checker(), "checker"))))


def flip_tangent_w(geo):
    """Same geometry with TANGENT.w = -1: the bitangent (normal-map green axis) points the other way."""
    geo = dict(geo)
    geo["tan"] = [(x, y, z, -w) for x, y, z, w in geo["tan"]]
    return geo


# ================================================================================================
# 06a | Normal map: OpenGL convention (glTF standard) + tangent w flip
# ================================================================================================
def normal_map_gl_off_on_tangent_w_flip(g):
    """OpenGL normal map, the convention glTF requires (green = up). Same layout as 06b.
    Left: no normal map.
    Middle: GL map, normal tangents (w = +1) -> CORRECT: round convex bumps, lit consistently
            (the side facing the light is bright both up/down and left/right).
    Right: same GL map, but TANGENT.w = -1 (bitangent flipped) -> bumps lit from the wrong side
           up/down only. Must look IDENTICAL to the middle sphere of 06b, if the renderer uses
           the tangent w sign."""
    bumps = g.texture(tex_bumps(), "normal_bumps_gl")
    g.row([g.mesh(sphere(), g.pbr(rough=0.8), tangents=True),
           g.mesh(sphere(), g.pbr(rough=0.8, normalTexture=bumps), tangents=True),
           g.mesh(flip_tangent_w(sphere()), g.pbr(rough=0.8, normalTexture=bumps), tangents=True)])


# ================================================================================================
# 06b | Normal map: DirectX convention (green inverted) + tangent w flip
# ================================================================================================
def normal_map_dx_off_on_tangent_w_flip(g):
    """DirectX normal map (green inverted). glTF has NO field or extension to declare this:
    the spec says normal maps are OpenGL, so a correct importer reads this map as if it were GL.
    Left: no normal map.
    Middle: DX map, normal tangents (w = +1) -> WRONG on purpose: the bumps still look right
            left/right, but are lit from the wrong side up/down (they look like a bump horizontally
            and a dent vertically). Compare with the middle sphere of 06a, and it must look IDENTICAL to the right sphere of 06a.
    Right: same DX map, but TANGENT.w = -1, which flips the bitangent. That cancels the inverted
           green, so it must look IDENTICAL to the middle sphere of 06a, if the renderer uses the
           tangent w sign."""
    bumps = g.texture(tex_bumps(directx=True), "normal_bumps_dx")
    g.row([g.mesh(sphere(), g.pbr(rough=0.8), tangents=True),
           g.mesh(sphere(), g.pbr(rough=0.8, normalTexture=bumps), tangents=True),
           g.mesh(flip_tangent_w(sphere()), g.pbr(rough=0.8, normalTexture=bumps), tangents=True)])


# ================================================================================================
# 06c | Normal map strength (normalTexture.scale)
# ================================================================================================
def normal_map_gl_scale_0_1_4(g):
    """Same OpenGL bump map on all three spheres, only normalTexture.scale changes
    (it multiplies the X/Y tilt of the normals):
    Left: scale 0 -> must look IDENTICAL to a sphere with no normal map (left of 06a / 06b).
    Middle: scale 1 (the glTF default) -> must look IDENTICAL to the middle sphere of 06a.
    Right: scale 4 -> much deeper bumps; strong shading on the bump edges. Path tracers may show
           dark rims there (very tilted normals), which is a known normal-map limitation."""
    tex = g.texture(tex_bumps(), "normal_bumps_gl")
    g.row([g.mesh(sphere(), g.pbr(rough=0.8, normalTexture=dict(tex, scale=k)), tangents=True)
           for k in (0.0, 1.0, 10.0)])


# ================================================================================================
# 06d | Normal map: high-res 1024 on a plane, vs height reference
# ================================================================================================
def normal_map_gl_1024_plane_vs_height_reference(g):
    """High-resolution (1024x1024) OpenGL normal map on a flat plane facing the camera.
    Left: grey plane with ONLY the normal map. Right: no normal map, the same height field shown
    as a colour texture (light = high, dark = low): this is the shape the left plane must show.
    Regions (both planes): top-left raised domes, top-right sunken dents, bottom-left raised
    pyramids, bottom-right fine concentric ripples.
    What to check on the left plane:
    - domes and dents are lit from OPPOSITE sides (dents look like holes, not bumps);
    - pyramids show 4 flat facets with uniform shading each, sharp diagonal edges;
    - ripples are crisp rings, not blurry or blocky (texture resolution kept);
    - any region looking inverted only up/down = green channel read the wrong way."""
    h = height_field_regions()
    nrm = g.texture(tex_normal_from_height(h), "normal_hires_gl")
    ref = g.texture(tex_height_preview(h), "height_reference")
    g.row([g.mesh(quad(1.1), g.pbr(rough=0.35, normalTexture=nrm), tangents=True),
           g.mesh(quad(1.1), g.pbr((1, 1, 1, 1), rough=0.9, baseColorTexture=ref))], spacing=1.25)


# ================================================================================================
# 07  | Emission: factor, KHR_materials_emissive_strength, texture
# ================================================================================================
def emissive_none_factor_strength4_texture(g):
    """Black spheres, left to right:
    1. no emission
    2. emissiveFactor orange
    3. same factor x KHR_materials_emissive_strength 4 (must be clearly brighter than 2)
    4. same factor x emissiveTexture of white/black stripes -> orange glowing stripes,
       black (non-glowing) stripes in between."""
    g.d["extensionsUsed"] = ["KHR_materials_emissive_strength"]
    orange = [1, 0.4, 0]
    g.row([g.mesh(sphere(), g.pbr((0, 0, 0, 1))),
           g.mesh(sphere(), g.pbr((0, 0, 0, 1), emissiveFactor=orange)),
           g.mesh(sphere(), g.pbr((0, 0, 0, 1), emissiveFactor=orange,
                                  extensions={"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}})),
           g.mesh(sphere(), g.pbr((0, 0, 0, 1), emissiveFactor=orange,
                                  emissiveTexture=g.texture(tex_stripes(), "emissive_stripes")))])


# ================================================================================================
# 08  | Alpha modes: OPAQUE / BLEND / MASK (cutoff)
# ================================================================================================
def alpha_opaque_blend_mask(g):
    """Left: alpha 0.3, OPAQUE (alpha ignored -> solid). Middle: alpha 0.3, BLEND (see-through).
    Right: MASK with the default alphaCutoff 0.5 and a checker texture whose alpha is 0.75 / 0.25:
    0.75 cells must be fully SOLID (not semi-transparent), 0.25 cells fully cut out. The sphere is
    double-sided so the inside of the far half shows through the holes."""
    col = (1, 0.5, 0, 0.3)
    mask = g.pbr((1, 1, 1, 1), alphaMode="MASK", doubleSided=True,
                 baseColorTexture=g.texture(tex_alpha_checker(), "alpha_checker"))
    g.row([g.mesh(sphere(), g.pbr(col, alphaMode="OPAQUE")),
           g.mesh(sphere(), g.pbr(col, alphaMode="BLEND")),
           g.mesh(sphere(), mask)])


# ================================================================================================
# 09  | Double-sided vs single-sided (back faces)
# ================================================================================================
def double_sided_on_off(g):
    """Two quads seen from the BACK: left doubleSided (visible), right single-sided (culled)."""
    flip = [0, 1, 0, 0]  # 180 deg around Y
    tex = g.texture(tex_quadrants(), "quadrants")
    g.node(g.mesh(quad(), g.pbr(baseColorTexture=tex, doubleSided=True)), translation=[-0.6, 0, 0], rotation=flip)
    g.node(g.mesh(quad(), g.pbr(baseColorTexture=tex)), translation=[0.6, 0, 0], rotation=flip)


# ================================================================================================
# 10  | Node transforms: scale / rotation / translation
# ================================================================================================
def node_transforms(g):
    """Left: sphere squashed (scale y 0.5). Middle: cube rotated 45deg on Z. Right: cube translated up 0.3."""
    g.node(g.mesh(sphere(), g.pbr((0.1, 0.6, 0.6, 1))), translation=[-1.2, 0, 0], scale=[1, 0.5, 1])
    s = math.sin(math.radians(22.5))
    g.node(g.mesh(box(0.6, 0.6, 0.6), g.pbr((0.8, 0.4, 0.1, 1))), rotation=[0, 0, s, math.cos(math.radians(22.5))])
    g.node(g.mesh(box(0.6, 0.6, 0.6), g.pbr((0.5, 0.2, 0.8, 1))), translation=[1.2, 0.3, 0])


# ================================================================================================
# 11  | Axis orientation (Y-up -> Z-up)
# ================================================================================================
def axes_x_red_y_green_z_blue(g):
    """glTF axes: red bar +X, green bar +Y (should point up), blue bar +Z (toward camera)."""
    for size, center, col in (((1, 0.1, 0.1), (0.5, 0, 0), (0.9, 0.05, 0.05, 1)),
                              ((0.1, 1, 0.1), (0, 0.5, 0), (0.05, 0.8, 0.05, 1)),
                              ((0.1, 0.1, 1), (0, 0, 0.5), (0.05, 0.1, 0.9, 1))):
        g.node(g.mesh(box(*size, center=center), g.pbr(col)))


# ---- metallicRoughnessTexture + occlusionTexture -----------------------------------
# glTF: metallicRoughnessTexture -> G = roughness, B = metalness, R is IGNORED (both multiplied
# by their factors). occlusionTexture -> R = occlusion, blended by "strength". Both are linear.
# Nyx stores all three in a single "ARM" texture (R = AO, G = roughness, B = metal), so the
# importer has to merge two glTF textures into one: these tests target that merge.

GOLD = (1.0, 0.77, 0.34, 1)


def tex_mr_grid(n=128, cells=4):
    """Roughness (G) increases left -> right, metalness (B) increases top -> bottom. R = 255."""
    return png(n, n, lambda x, y: (255, 255 * (x * cells // n) // (cells - 1), 255 * (y * cells // n) // (cells - 1)))


def tex_mr_vertical_bands(n=128, bands=4):
    """Vertical bands alternating polished metal (rough 0.2, metal 1) / satin plastic (rough 0.6, metal 0)."""
    return png(n, n, lambda x, y: (255, 51, 255) if (x * bands // n) % 2 == 0 else (255, 153, 0))


def tex_ao_horizontal_bands(n=64, bands=8):
    """Occlusion (R) in horizontal bands alternating 0.15 (dark) / 1.0 (none). G, B unused."""
    return png(n, n, lambda x, y: (38, 255, 255) if (y * bands // n) % 2 == 0 else (255, 255, 255))


# ================================================================================================
# 19  | Metallic-roughness texture vs factors (red ignored)
# ================================================================================================
def metallic_roughness_texture_vs_factors(g):
    """Four gold spheres that must ALL LOOK IDENTICAL (polished metal: metallic 1, roughness 0.2):
    1. factors only (reference)
    2. 1x1 texture (R 255, G 0.2, B 1) with factors 1
    3. same texture but R = 0: R must be ignored. Darker than 1 => the importer reads R as AO
    4. texture G 1.0 x roughnessFactor 0.2: texture and factor must multiply"""
    g.row([g.mesh(sphere(), g.pbr(GOLD, 1.0, 0.2)),
           g.mesh(sphere(), g.pbr(GOLD, 1.0, 1.0, metallicRoughnessTexture=g.texture(tex_solid((255, 51, 255)), "mr_r255"))),
           g.mesh(sphere(), g.pbr(GOLD, 1.0, 1.0, metallicRoughnessTexture=g.texture(tex_solid((0, 51, 255)), "mr_r0"))),
           g.mesh(sphere(), g.pbr(GOLD, 1.0, 0.2, metallicRoughnessTexture=g.texture(tex_solid((255, 255, 255)), "mr_white")))])


# ================================================================================================
# 20  | Metallic-roughness texture channels (G = rough, B = metal)
# ================================================================================================
def metallic_roughness_texture_grid(g):
    """Quad + sphere with a 4x4 metallicRoughnessTexture: roughness increases left -> right,
    metalness increases top -> bottom. Top row = plastic, bottom row = gold metal; left column
    sharp reflections, right column blurry. Rotated/mirrored grid => G and B channels swapped."""
    mat = g.pbr(GOLD, 1.0, 1.0, metallicRoughnessTexture=g.texture(tex_mr_grid(), "mr_grid"))
    g.row([g.mesh(quad(), mat), g.mesh(sphere(), mat)])


# ================================================================================================
# 21  | Occlusion texture strength
# ================================================================================================
def occlusion_none_strength_0_05_1(g):
    """White matte spheres with an occlusionTexture of dark horizontal bands. Left to right:
    no occlusion texture, strength 0 (must equal no texture), strength 0.5 (light bands),
    strength 1 (dark bands). Note: a path tracer computes real occlusion itself and may
    legitimately ignore AO textures, so compare in Forward first."""
    ao = g.texture(tex_ao_horizontal_bands(), "ao_bands")
    g.row([g.mesh(sphere(), g.pbr((0.9, 0.9, 0.9, 1), 0.0, 0.8))] +
          [g.mesh(sphere(), g.pbr((0.9, 0.9, 0.9, 1), 0.0, 0.8, occlusionTexture=dict(ao, strength=st)))
           for st in (0.0, 0.5, 1.0)])


# ================================================================================================
# 22  | Occlusion + metallic-roughness merge (Nyx ARM texture)
# ================================================================================================
def occlusion_and_mr_separate_packed_ao_only_mr_only(g):
    """How the importer merges glTF's two textures into Nyx's single ARM texture:
    1. SEPARATE images: MR = vertical bands (polished gold / satin plastic), AO = dark horizontal
       bands. The AO image is 64x64 and the MR image 128x128 on purpose (merge must resample).
    2. PACKED: one ORM image used for both slots. Must look identical to 1.
    3. AO only, metal/rough from factors (metallic 1, roughness 0.2): polished gold + dark
       horizontal bands, no vertical bands (missing MR must count as 1 so the factors apply).
    4. MR only, no AO: vertical bands, no dark horizontal bands (missing AO must count as 1)."""
    mr = g.texture(tex_mr_vertical_bands(), "mr_vertical_bands")
    ao = g.texture(tex_ao_horizontal_bands(), "ao_horizontal_bands")
    n = 128
    orm = png(n, n, lambda x, y: (38 if (y * 8 // n) % 2 == 0 else 255,) +
              ((51, 255) if (x * 4 // n) % 2 == 0 else (153, 0)))
    packed = g.texture(orm, "orm_packed")
    g.row([g.mesh(sphere(), g.pbr(GOLD, 1.0, 1.0, metallicRoughnessTexture=mr, occlusionTexture=ao)),
           g.mesh(sphere(), g.pbr(GOLD, 1.0, 1.0, metallicRoughnessTexture=packed, occlusionTexture=packed)),
           g.mesh(sphere(), g.pbr(GOLD, 1.0, 0.2, occlusionTexture=ao)),
           g.mesh(sphere(), g.pbr(GOLD, 1.0, 1.0, metallicRoughnessTexture=mr))])


# ---- samplers -------------------------------------------------------------------
# glTF constants: wrap REPEAT 10497, CLAMP_TO_EDGE 33071, MIRRORED_REPEAT 33648,
# filter NEAREST 9728, LINEAR 9729, LINEAR_MIPMAP_LINEAR 9987.

# ================================================================================================
# 17  | Sampler wrap modes
# ================================================================================================
def sampler_wrap_repeat_clamp_mirror_mixed(g):
    """Quadrant texture (top-left red, top-right green, bottom-left blue, bottom-right yellow)
    on quads whose UVs go 0..3, so the texture is tiled 3x3. Left to right:
    1. REPEAT: 9 identical tiles.
    2. CLAMP_TO_EDGE: one tile in the top-left corner, the edge colours smeared over the rest
       (green/yellow stripes to the right, blue/yellow below, yellow in the bottom-right).
    3. MIRRORED_REPEAT: tiles flipped every other time, so same colours touch across tile borders.
    4. Mixed: wrapS REPEAT, wrapT CLAMP_TO_EDGE: repeats horizontally, smeared vertically
       (catches an importer that uses one wrap mode for both axes)."""
    def q(ws, wt):
        tex = g.texture(tex_quadrants(), "quadrants", sampler={"wrapS": ws, "wrapT": wt, "magFilter": 9729, "minFilter": 9729})
        return g.mesh(quad(uv_scale=3.0), g.pbr((1, 1, 1, 1), baseColorTexture=tex))
    g.row([q(10497, 10497), q(33071, 33071), q(33648, 33648), q(10497, 33071)])


# ================================================================================================
# 18a | Sampler filtering (magnification / mipmaps)
# ================================================================================================
def sampler_filter_mag_nearest_linear_min_nearest_mipmap(g):
    """Left to right:
    1. 4x4 texel texture, magFilter NEAREST: crisp 4x4 grid of coloured squares.
    2. same texture, magFilter LINEAR: blurry colour gradients.
    3. 256x256 one-texel black/white checker (PIXEL_CHECKER_SIZE), minFilter NEAREST: noisy / moire pattern.
    4. same, minFilter LINEAR_MIPMAP_LINEAR: smooth uniform grey.
    Note: the path tracers average many samples per pixel, which hides most of the difference
    between 3 and 4; compare those two in Forward."""
    def q(png_bytes, name, mag, mn):
        tex = g.texture(png_bytes, name, sampler={"magFilter": mag, "minFilter": mn})
        return g.mesh(quad(), g.pbr((1, 1, 1, 1), baseColorTexture=tex))
    g.row([q(tex_tiny_4x4(), "tiny_4x4", 9728, 9728),
           q(tex_tiny_4x4(), "tiny_4x4", 9729, 9729),
           q(tex_pixel_checker(), "pixel_checker", 9729, 9728),
           q(tex_pixel_checker(), "pixel_checker", 9729, 9987)])


# ================================================================================================
# 18b | Mipmapping at a grazing angle (minFilter modes, default sampler, NPOT texture)
# ================================================================================================
def mipmaps_grazing_point_bilinear_trilinear_default_npot_mipbands(g):
    """Six long floor strips going away from the camera (grazing angle), all with the same
    black/white checker (4x4 texels per square, ~3 screen pixels per texel at the near end).
    glTF has no "generate mipmaps" flag: mipmaps are requested through sampler.minFilter, and the
    importer must build them (PNG/JPEG can't store mips). Left to right:
    1. POINT, no mips (mag NEAREST, min NEAREST): sharp squares near, heavy aliasing / moire far.
    2. BILINEAR, no mips (mag LINEAR, min LINEAR): soft edges near, still aliasing / moire far.
    3. TRILINEAR (mag LINEAR, min LINEAR_MIPMAP_LINEAR): soft near, fades smoothly to flat grey far.
    4. NO SAMPLER at all: the importer picks the filtering. Should look like 3 (mipmapped).
    5. TRILINEAR on a 48x48 texture (NOT a power of two, same square size): must look like 3.
       If it aliases like 2, the importer skips mip generation for non-power-of-two textures.
    6. min NEAREST_MIPMAP_NEAREST: no blending between mip levels, so visible BANDS where the
       mip level changes (shows that mips exist and how many levels there are).
    Compare in Forward: the path tracers average many samples per pixel, which hides aliasing.
    Trilinear going grey quickly in the distance is expected without anisotropic filtering
    (glTF has no setting for that; it's up to the renderer)."""
    width, gap, z_near, z_far = 0.5, 0.08, 2.0, -40.0
    v_tiles = (z_near - z_far) / width            # keep the squares square
    rep = {"wrapS": 10497, "wrapT": 10497}
    strips = [  # (texture size, sampler or None)
        (64, dict(rep, magFilter=9728, minFilter=9728)),   # NEAREST / NEAREST
        (64, dict(rep, magFilter=9729, minFilter=9729)),   # LINEAR / LINEAR
        (64, dict(rep, magFilter=9729, minFilter=9987)),   # LINEAR / LINEAR_MIPMAP_LINEAR
        (64, None),                                        # no sampler -> importer default
        (48, dict(rep, magFilter=9729, minFilter=9987)),   # NPOT, trilinear
        (64, dict(rep, magFilter=9729, minFilter=9984)),   # LINEAR / NEAREST_MIPMAP_NEAREST
    ]
    for i, (size, smp) in enumerate(strips):
        tex = g.texture(tex_grazing_checker(size), "checker_%d_strip%d" % (size, i + 1), sampler=smp)
        k = 64 / size                                 # same square size on every strip
        mesh = g.mesh(floor_strip(width, z_near, z_far, k, v_tiles * k),
                      g.pbr((1, 1, 1, 1), rough=1.0, baseColorTexture=tex))
        x = (i - (len(strips) - 1) / 2) * (width + gap)
        g.node(mesh, translation=[x, -0.55, 0])        # just above the ground level of the render script


# ---- KHR_materials_anisotropy -----------------------------------------------------
# Spec: the roughness ALONG the anisotropy direction becomes mix(roughness, 1, strength^2),
# so highlights/reflections get stretched along that direction. Direction = tangent (glTF
# TANGENT, i.e. increasing U) rotated by anisotropyRotation (radians, counter-clockwise).
# On these spheres the tangent runs horizontally (around the vertical axis), so rotation 0
# = horizontally brushed, rotation 90deg = vertically brushed.
# All spheres: polished grey metal (metallic 1, roughness 0.2).

def aniso_mat(g, strength=None, rotation=None, texture=None):
    g.d["extensionsUsed"] = ["KHR_materials_anisotropy"]
    ext = {}
    if strength is not None:
        ext["anisotropyStrength"] = strength
    if rotation is not None:
        ext["anisotropyRotation"] = rotation
    if texture is not None:
        ext["anisotropyTexture"] = texture
    return g.pbr((0.9, 0.9, 0.9, 1), 1.0, 0.2, extensions={"KHR_materials_anisotropy": ext})


# ================================================================================================
# 12  | Anisotropy strength
# ================================================================================================
def anisotropy_strength_0_05_1(g):
    """anisotropyStrength 0 / 0.5 / 1, rotation 0. Left = normal round reflections,
    then increasingly smeared HORIZONTALLY."""
    g.row([g.mesh(sphere(), aniso_mat(g, strength=st), tangents=True) for st in (0.0, 0.5, 1.0)])


# ================================================================================================
# 13  | Anisotropy rotation
# ================================================================================================
def anisotropy_rotation_0_45_90(g):
    """anisotropyStrength 1, rotation 0 / 45 / 90 deg. Smear direction turns from
    horizontal, to diagonal, to vertical. Left must match the right sphere of 12."""
    g.row([g.mesh(sphere(), aniso_mat(g, strength=1.0, rotation=math.radians(a)), tangents=True)
           for a in (0, 45, 90)])


# ================================================================================================
# 14  | Anisotropy direction texture
# ================================================================================================
def anisotropy_texture_h_v_checker(g):
    """anisotropyStrength 1 + anisotropyTexture. Left: texture direction = tangent (must match
    12 right). Middle: direction = bitangent (must match 13 right, vertical). Right: checker
    of horizontal / vertical cells -> patchwork of brushed tiles."""
    g.row([g.mesh(sphere(), aniso_mat(g, strength=1.0, texture=g.texture(tex_solid(ANISO_H), "aniso_horizontal")), tangents=True),
           g.mesh(sphere(), aniso_mat(g, strength=1.0, texture=g.texture(tex_solid(ANISO_V), "aniso_vertical")), tangents=True),
           g.mesh(sphere(), aniso_mat(g, strength=1.0, texture=g.texture(tex_aniso_checker(), "aniso_checker")), tangents=True)])


# ================================================================================================
# 15  | Anisotropy strength from texture (blue channel)
# ================================================================================================
def anisotropy_texture_blue_strength(g):
    """Texture BLUE channel multiplies anisotropyStrength (direction = tangent everywhere).
    Left: factor 1 x blue 0.5. Middle: factor 0.5 x blue 1. Right: factor 1 x blue 0.
    Left and middle must both match 12 middle (strength 0.5); right must match 12 left (none)."""
    g.row([g.mesh(sphere(), aniso_mat(g, strength=1.0, texture=g.texture(tex_solid((255, 128, 128)), "aniso_blue_half")), tangents=True),
           g.mesh(sphere(), aniso_mat(g, strength=0.5, texture=g.texture(tex_solid((255, 128, 255)), "aniso_blue_full")), tangents=True),
           g.mesh(sphere(), aniso_mat(g, strength=1.0, texture=g.texture(tex_solid((255, 128, 0)), "aniso_blue_zero")), tangents=True)])


# ================================================================================================
# 16  | Anisotropy: TANGENT given / missing / funky (are file tangents kept or recalculated?)
# ================================================================================================
def funky_tangents(geo):
    """Same sphere, but in the TOP half the tangents are turned 90 degrees around the normal: they
    point along the meridians (vertical) instead of around the sphere. The bottom half keeps the
    normal tangents. A tangent recalculation from the UVs can never produce this."""
    geo = dict(geo)
    tan = []
    for (x, y, z), t in zip(geo["nrm"], geo["tan"]):
        if y > 1e-6:                                  # top half (glTF +Y is up)
            th, ph = math.acos(max(-1.0, min(1.0, y))), math.atan2(x, z)
            t = (math.cos(th) * math.sin(ph), -math.sin(th), math.cos(th) * math.cos(ph), 1)
        tan.append(t)
    geo["tan"] = tan
    return geo


def anisotropy_tangents_given_none_funky(g):
    """anisotropyStrength 1, rotation 0 (stretch along the tangent). Left to right:
    1. mesh HAS a normal TANGENT attribute -> horizontal smear.
    2. NO TANGENT attribute, the importer must generate it -> must look like 1.
    3. FUNKY TANGENT attribute: top half tangents vertical, bottom half normal.
       - top half smeared VERTICALLY, bottom half horizontally -> the file's tangents are KEPT;
       - whole sphere horizontal, identical to 1 -> the tangents are RECALCULATED (file ignored).
    Render with a light (see EXTRA_LIGHTS in render_gltf_tests.py): Forward only shows anisotropy
    on lights, not on the environment map."""
    g.row([g.mesh(sphere(), aniso_mat(g, strength=1.0), tangents=True),
           g.mesh(sphere(), aniso_mat(g, strength=1.0), tangents=False),
           g.mesh(funky_tangents(sphere()), aniso_mat(g, strength=1.0), tangents=True)])


# ---- everything together -------------------------------------------------------------

def rot_y(deg):
    """glTF quaternion [x, y, z, w] for a rotation around Y."""
    return [0, math.sin(math.radians(deg) / 2), 0, math.cos(math.radians(deg) / 2)]


def tex_combined_base(n=128, cells=8):
    """Checker white / light blue."""
    return png(n, n, lambda x, y: (230, 230, 230) if (x * cells // n + y * cells // n) % 2 else (90, 140, 230))


def tex_combined_orm(n=128):
    """Packed ORM. R (AO): dark band in the middle (v 0.45..0.55).
    G (roughness): top half 0.15 (polished), bottom half 0.7 (rough).
    B (metal): 4 vertical bands alternating metal / non-metal."""
    def f(x, y):
        u, v = x / n, y / n
        return (51 if 0.45 <= v < 0.55 else 255, 38 if v < 0.5 else 179, 255 if int(u * 4) % 2 == 0 else 0)
    return png(n, n, f)


def tex_combined_emissive(n=128):
    """White band at v 0.2..0.27, black elsewhere (tinted cyan by emissiveFactor)."""
    return png(n, n, lambda x, y: (255, 255, 255) if 0.2 <= y / n < 0.27 else (0, 0, 0))


# ================================================================================================
# 23  | All basic features together
# ================================================================================================
def combined_all_basic_features(g):
    """Everything from the single-feature tests at once. Each texture drives a different region
    so every feature stays readable:

    LEFT  sphere and MIDDLE quad share ONE material ("hero"):
      - base colour: white / light-blue checker (texture) x warm tint (factor (1, 0.85, 0.7))
      - metallicRoughness + occlusion from ONE packed ORM texture:
          top half polished, bottom half rough (G), 4 vertical bands metal / non-metal (B),
          dark AO band around the middle (R, Forward only, Blender ignores AO)
      - normal map: 4x4 convex bumps, scale 0.6
      - emissive: glowing CYAN ring/band in the upper part (white texture band x cyan factor)
      - sampler: REPEAT + mipmapped LINEAR
    RIGHT leaf: double-sided quad, alphaMode MASK (cutoff 0.5) with the 0.75/0.25 alpha checker,
      the same bump normal map, seen from its BACK: half the cells cut out, no culling.

    Node hierarchy: everything hangs under one root node (scale 1, translated down 0.05 to
    test nested transforms); sphere rotated 200 deg around Y, quad rotated -20 deg around Y,
    leaf = parent (translation + uniform scale 0.9) -> child (rotation 210 deg around Y)."""
    smp = {"wrapS": 10497, "wrapT": 10497, "magFilter": 9729, "minFilter": 9987}
    base = g.texture(tex_combined_base(), "combined_base", sampler=smp)
    orm = g.texture(tex_combined_orm(), "combined_orm_packed", sampler=smp)
    nrm = g.texture(tex_bumps(), "combined_normal_bumps", sampler=smp)
    emi = g.texture(tex_combined_emissive(), "combined_emissive", sampler=smp)
    hero = g.pbr((1, 0.85, 0.7, 1), 1.0, 1.0,
                 baseColorTexture=base, metallicRoughnessTexture=orm,
                 occlusionTexture=dict(orm, strength=1.0), normalTexture=dict(nrm, scale=0.6),
                 emissiveTexture=emi, emissiveFactor=[0.2, 1.0, 1.0])
    leaf = g.pbr((1, 1, 1, 1), 0.0, 0.5, alphaMode="MASK", alphaCutoff=0.5, doubleSided=True,
                 baseColorTexture=g.texture(tex_alpha_checker(), "combined_alpha_checker", sampler=smp),
                 normalTexture=nrm)

    def add(**n):
        g.d["nodes"].append(n)
        return len(g.d["nodes"]) - 1

    sphere_node = add(mesh=g.mesh(sphere(), hero, tangents=True), translation=[-1.2, 0, 0], rotation=rot_y(200))
    quad_node = add(mesh=g.mesh(quad(), hero, tangents=True), rotation=rot_y(-20))
    leaf_child = add(mesh=g.mesh(quad(), leaf, tangents=True), rotation=rot_y(210))
    leaf_parent = add(children=[leaf_child], translation=[1.2, 0, 0], scale=[0.9, 0.9, 0.9])
    root = add(children=[sphere_node, quad_node, leaf_parent], translation=[0, -0.05, 0])
    g.d["scenes"][0]["nodes"].append(root)


TESTS = [base_color_red_green_blue, metallic_0_05_1, roughness_0_05_1, base_color_texture_quadrants,
         base_color_texture_on_sphere, (normal_map_gl_off_on_tangent_w_flip, normal_map_dx_off_on_tangent_w_flip,
          normal_map_gl_scale_0_1_4, normal_map_gl_1024_plane_vs_height_reference),
         emissive_none_factor_strength4_texture, alpha_opaque_blend_mask,
         double_sided_on_off, node_transforms, axes_x_red_y_green_z_blue,
         anisotropy_strength_0_05_1, anisotropy_rotation_0_45_90, anisotropy_texture_h_v_checker,
         anisotropy_texture_blue_strength, anisotropy_tangents_given_none_funky,
         sampler_wrap_repeat_clamp_mirror_mixed,
         (sampler_filter_mag_nearest_linear_min_nearest_mipmap,
          mipmaps_grazing_point_bilinear_trilinear_default_npot_mipbands),
         metallic_roughness_texture_vs_factors, metallic_roughness_texture_grid,
         occlusion_none_strength_0_05_1, occlusion_and_mr_separate_packed_ao_only_mr_only,
         combined_all_basic_features]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, entry in enumerate(TESTS, 1):
        # A tuple in TESTS = variants of one test, numbered 06a, 06b, ...
        variants = [("", entry)] if callable(entry) else [(chr(ord("a") + k), t) for k, t in enumerate(entry)]
        for suffix, test in variants:
            name = "%02d%s_%s" % (i, suffix, test.__name__)
            g = Gltf(name)
            test(g)
            path = os.path.join(OUT_DIR, name + ".glb")
            g.save(path)
            print("wrote", path)


if __name__ == "__main__":
    main()
