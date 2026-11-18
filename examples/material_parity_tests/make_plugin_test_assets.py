"""Generate the assets of the glTF parity test of the Nyx Genesis plugin (tests/test_gltf_parity.py).

For every pair of that test: <pair>.glb, ONE object at the origin with its glTF material.
Plus the same meshes without material (sphere.glb, quad.glb), which the test loads with a
Genesis surface, and the textures those surfaces read.

The Genesis surface of each pair is written in the test itself: keep both in sync.

Usage:
    python examples/material_parity_tests/make_plugin_test_assets.py <plugin repo>/tests/assets
        -> <assets>/models/gltf_parity/*.glb and <assets>/textures/gltf_parity/*.png
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gltf_tests"))
sys.path.insert(0, os.path.join(HERE, "..", "gltf_transparency_tests"))
import make_gltf_tests as mk                 # noqa: E402
import make_gltf_transparency_tests as mkt   # noqa: E402

GOLD   = (1.0, 0.77, 0.34, 1)
WHITE  = (1, 1, 1, 1)
BLUE   = (0.1, 0.3, 0.8, 1)
ORANGE = [1.0, 0.4, 0.0]

TEXTURES = {
    "checker":       mk.tex_checker,
    "quadrants":     mk.tex_quadrants,
    "alpha_checker": mk.tex_alpha_checker,
    "normal_bumps":  mk.tex_bumps,
    "stripes":       mk.tex_stripes,
    # Wall behind the objects: two vertical grey stripes.
    "backdrop":      lambda: mk.png(64, 2, lambda x, y: (165, 165, 165) if x < 32 else (95, 95, 95)),
}


def sphere():
    return mk.sphere(0.5, seg=32, rings=16)   # coarser than the examples, to keep the files small


GEOMETRY = {"sphere": sphere, "quad": mk.quad}


def tex(g, name):
    return g.texture(TEXTURES[name](), name)


# pair name -> (geometry, function(g) -> glTF material index)
PAIRS = {
    "base_color":        ("sphere", lambda g: g.pbr((0.8, 0.05, 0.05, 1), 0.0, 0.5)),
    "metallic_0":        ("sphere", lambda g: g.pbr(GOLD, 0.0, 0.2)),
    "metallic_05":       ("sphere", lambda g: g.pbr(GOLD, 0.5, 0.2)),
    "metallic_1":        ("sphere", lambda g: g.pbr(GOLD, 1.0, 0.2)),
    "roughness_06":      ("sphere", lambda g: g.pbr(GOLD, 1.0, 0.6)),
    "texture":           ("sphere", lambda g: g.pbr(WHITE, 0.0, 0.5, baseColorTexture=tex(g, "checker"))),
    "texture_tinted":    ("sphere", lambda g: g.pbr((1.0, 0.5, 0.2, 1), 0.0, 0.5, baseColorTexture=tex(g, "checker"))),
    "normal_map":        ("sphere", lambda g: g.pbr((0.8, 0.8, 0.8, 1), 0.0, 0.5, normalTexture=tex(g, "normal_bumps"))),
    "opacity_03":        ("sphere", lambda g: g.pbr((1.0, 0.5, 0.0, 0.3), 0.0, 0.5, alphaMode="BLEND")),
    "alpha_texture":     ("sphere", lambda g: g.pbr(WHITE, 0.0, 0.5, alphaMode="BLEND", doubleSided=True,
                                                    baseColorTexture=tex(g, "alpha_checker"))),
    "double_sided_back": ("quad",   lambda g: g.pbr(WHITE, 0.0, 0.5, doubleSided=True,
                                                    baseColorTexture=tex(g, "quadrants"))),
    "glass_clear":       ("sphere", lambda g: mkt.glass(g, ior=1.5, thickness=1.0)),
    "glass_rough":       ("sphere", lambda g: mkt.glass(g, rough=0.3, ior=1.5, thickness=1.0)),
    "glass_tinted":      ("sphere", lambda g: mkt.glass(g, color=(0.2, 0.6, 1.0), ior=1.5, thickness=1.0)),
    "glass_thin":        ("sphere", lambda g: mkt.glass(g, ior=1.5, doubleSided=True)),
    "glass_ior_133":     ("sphere", lambda g: mkt.glass(g, ior=1.33, thickness=1.0)),
    # Emission on top of a blue base: glTF keeps the base, the plugin makes a pure emitter.
    "emissive_color":    ("sphere", lambda g: g.pbr(BLUE, 0.0, 0.5, emissiveFactor=ORANGE)),
    "emissive_texture":  ("sphere", lambda g: g.pbr(BLUE, 0.0, 0.5, emissiveFactor=ORANGE,
                                                    emissiveTexture=tex(g, "stripes"))),
}


def save(g, path):
    g.pngs.clear()   # no copy of the embedded textures
    g.save(path)
    print("wrote", path)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    models = os.path.join(sys.argv[1], "models", "gltf_parity")
    textures = os.path.join(sys.argv[1], "textures", "gltf_parity")
    os.makedirs(models, exist_ok=True)
    os.makedirs(textures, exist_ok=True)

    for name, (geo, material) in PAIRS.items():
        g = mk.Gltf(name)
        g.node(g.mesh(GEOMETRY[geo](), material(g), tangents=True))
        save(g, os.path.join(models, name + ".glb"))
    for geo, make in GEOMETRY.items():
        g = mk.Gltf(geo)
        g.node(g.mesh(make(), tangents=True))
        save(g, os.path.join(models, geo + ".glb"))
    for name, make in TEXTURES.items():
        path = os.path.join(textures, name + ".png")
        with open(path, "wb") as f:
            f.write(make())
        print("wrote", path)


if __name__ == "__main__":
    main()
