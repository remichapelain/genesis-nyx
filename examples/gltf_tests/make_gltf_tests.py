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


# ---------------------------------------------------------------- PNG textures

def png(w, h, px):
    """Tiny RGB PNG encoder. px(x, y) -> (r, g, b) in 0..255."""
    raw = b"".join(b"\0" + b"".join(bytes(px(x, y)) for x in range(w)) for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def tex_quadrants(n=64):
    """Top-left red, top-right green, bottom-left blue, bottom-right yellow."""
    cols = {(0, 0): (255, 0, 0), (1, 0): (0, 255, 0), (0, 1): (0, 0, 255), (1, 1): (255, 255, 0)}
    return png(n, n, lambda x, y: cols[(x * 2 // n, y * 2 // n)])


def tex_checker(n=128, cells=8):
    return png(n, n, lambda x, y: (230, 230, 230) if (x * cells // n + y * cells // n) % 2 else (40, 40, 40))


def tex_bumps(n=128, cells=4):
    """Tangent-space normal map with round bumps (glTF / OpenGL convention, +Y up)."""
    c = n / cells

    def f(x, y):
        dx, dy = (x % c) / c * 2 - 1, -((y % c) / c * 2 - 1)
        r2 = dx * dx + dy * dy
        nz = math.sqrt(0.64 - r2) if r2 < 0.64 else None
        v = (dx * 0.8, dy * 0.8, nz) if nz is not None else (0, 0, 1)
        ln = math.sqrt(sum(a * a for a in v))
        return tuple(int((a / ln * 0.5 + 0.5) * 255) for a in v)

    return png(n, n, f)


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


def quad(size=1.0):
    h = size / 2
    return dict(pos=[(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], nrm=[(0, 0, 1)] * 4,
                uv=[(0, 1), (1, 1), (1, 0), (0, 0)], tan=[(1, 0, 0, 1)] * 4, idx=[0, 1, 2, 0, 2, 3])


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
    def __init__(self):
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

    def texture(self, png_bytes):
        self.d.setdefault("images", []).append({"bufferView": self._view(png_bytes), "mimeType": "image/png"})
        self.d.setdefault("textures", []).append({"source": len(self.d["images"]) - 1})
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
        self.d["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.d).encode()
        js += b" " * (-len(js) % 4)
        self.bin += b"\0" * (-len(self.bin) % 4)
        body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(self.bin), b"BIN\0") + bytes(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


# ---------------------------------------------------------------- the tests
# Each function fills a Gltf. The function name becomes the file / image name.

def base_color_red_green_blue(g):
    g.row([g.mesh(sphere(), g.pbr(c)) for c in ((0.8, 0.05, 0.05, 1), (0.05, 0.8, 0.05, 1), (0.05, 0.05, 0.8, 1))])


def metallic_0_05_1(g):
    """Same roughness (0.2), metallic 0 -> 0.5 -> 1 left to right."""
    g.row([g.mesh(sphere(), g.pbr((1, 0.77, 0.34, 1), m, 0.2)) for m in (0.0, 0.5, 1.0)])


def roughness_0_05_1(g):
    """Metallic 1, roughness 0 -> 0.5 -> 1 left to right."""
    g.row([g.mesh(sphere(), g.pbr((1, 0.77, 0.34, 1), 1.0, r)) for r in (0.0, 0.5, 1.0)])


def base_color_texture_quadrants(g):
    """Front-facing quad: top-left red, top-right green, bottom-left blue, bottom-right yellow."""
    g.node(g.mesh(quad(), g.pbr((1, 1, 1, 1), baseColorTexture=g.texture(tex_quadrants()))))


def base_color_texture_on_sphere(g):
    """Checker texture on a sphere, plain white factor."""
    g.node(g.mesh(sphere(), g.pbr((1, 1, 1, 1), baseColorTexture=g.texture(tex_checker()))))


def normal_map_off_on(g):
    """Left: no normal map. Right: bump normal map (bumps should look convex)."""
    bumps = g.texture(tex_bumps())
    g.row([g.mesh(sphere(), g.pbr(rough=0.3), tangents=True),
           g.mesh(sphere(), g.pbr(rough=0.3, normalTexture=bumps), tangents=True)])


def emissive_off_on(g):
    """Black spheres: left no emission, right emissiveFactor orange."""
    g.row([g.mesh(sphere(), g.pbr((0, 0, 0, 1))),
           g.mesh(sphere(), g.pbr((0, 0, 0, 1), emissiveFactor=[1, 0.4, 0]))])


def alpha_opaque_blend_mask(g):
    """Alpha 0.3 with alphaMode OPAQUE (solid), BLEND (see-through), MASK cutoff 0.5 (invisible)."""
    col = (1, 0.5, 0, 0.3)
    g.row([g.mesh(sphere(), g.pbr(col, alphaMode=m)) for m in ("OPAQUE", "BLEND", "MASK")])


def double_sided_on_off(g):
    """Two quads seen from the BACK: left doubleSided (visible), right single-sided (culled)."""
    flip = [0, 1, 0, 0]  # 180 deg around Y
    tex = g.texture(tex_quadrants())
    g.node(g.mesh(quad(), g.pbr(baseColorTexture=tex, doubleSided=True)), translation=[-0.6, 0, 0], rotation=flip)
    g.node(g.mesh(quad(), g.pbr(baseColorTexture=tex)), translation=[0.6, 0, 0], rotation=flip)


def node_transforms(g):
    """Left: sphere squashed (scale y 0.5). Middle: cube rotated 45deg on Z. Right: cube translated up 0.3."""
    g.node(g.mesh(sphere(), g.pbr((0.1, 0.6, 0.6, 1))), translation=[-1.2, 0, 0], scale=[1, 0.5, 1])
    s = math.sin(math.radians(22.5))
    g.node(g.mesh(box(0.6, 0.6, 0.6), g.pbr((0.8, 0.4, 0.1, 1))), rotation=[0, 0, s, math.cos(math.radians(22.5))])
    g.node(g.mesh(box(0.6, 0.6, 0.6), g.pbr((0.5, 0.2, 0.8, 1))), translation=[1.2, 0.3, 0])


def axes_x_red_y_green_z_blue(g):
    """glTF axes: red bar +X, green bar +Y (should point up), blue bar +Z (toward camera)."""
    for size, center, col in (((1, 0.1, 0.1), (0.5, 0, 0), (0.9, 0.05, 0.05, 1)),
                              ((0.1, 1, 0.1), (0, 0.5, 0), (0.05, 0.8, 0.05, 1)),
                              ((0.1, 0.1, 1), (0, 0, 0.5), (0.05, 0.1, 0.9, 1))):
        g.node(g.mesh(box(*size, center=center), g.pbr(col)))


TESTS = [base_color_red_green_blue, metallic_0_05_1, roughness_0_05_1, base_color_texture_quadrants,
         base_color_texture_on_sphere, normal_map_off_on, emissive_off_on, alpha_opaque_blend_mask,
         double_sided_on_off, node_transforms, axes_x_red_y_green_z_blue]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, test in enumerate(TESTS, 1):
        g = Gltf()
        test(g)
        path = os.path.join(OUT_DIR, "%02d_%s.glb" % (i, test.__name__))
        g.save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
