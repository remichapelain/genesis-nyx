"""Same tests as gltf_tests, but the materials are built with gs.surfaces instead of a glTF file.

This tests the Genesis -> Nyx material conversion (gs_nyx_plugin's nyx_scene_exporter),
not Nyx's glTF importer. Names, numbers, sphere layout and camera match gltf_tests so
out/<test>_<n>_<mode>.png can be compared side by side with gltf_tests/out/.
Tests 30+ cover Genesis surface features that have no glTF equivalent.

Usage:
    python examples/surface_tests/render_surface_tests.py            # all tests
    python examples/surface_tests/render_surface_tests.py 02 07      # only tests whose name contains "02" or "07"
    python examples/surface_tests/render_surface_tests.py metal

Textures (PNG) and the quad mesh (OBJ) are written to textures/ before rendering.
Each test is rendered in its own Python process, so a crash on one doesn't stop the others.
"""

import os
import subprocess
import sys

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
TEX_DIR = os.path.join(HERE, "textures")
ENV_MAP = os.path.join(HERE, "..", "assets", "green_sanctuary_4k.hdr")

# Reuse the PNG encoder and textures of the glTF tests, so both suites use the same images.
sys.path.insert(0, os.path.join(HERE, "..", "gltf_tests"))
from make_gltf_tests import (png, tex_alpha_checker, tex_bumps, tex_checker,  # noqa: E402
                             tex_quadrants, tex_solid, tex_stripes)

# Same render setup as gltf_tests/render_gltf_tests.py.
MODES      = ("Forward", "FastPathTracer", "RefPathTracer")
CAM_POS    = (0.0, -5.0, 1.5)
CAM_LOOKAT = (0.0, 0.0, 0.6)
CAM_RES    = (1280, 640)
CAM_FOV    = 30.0

# Same layout as the glTF files: objects 1 unit wide, 1.2 apart, centre at height 0.6.
RADIUS  = 0.5
SPACING = 1.2
HEIGHT  = 0.6

GOLD   = (1.0, 0.77, 0.34)
ORANGE = (1.0, 0.4, 0.0)


# ---------------------------------------------------------------------------
# Texture files
# ---------------------------------------------------------------------------

def tex_vertical_bands(lo, hi, n=128, bands=4):
    """Grey vertical bands alternating lo / hi (0..255), used as roughness / metallic maps."""
    return png(n, n, lambda x, y: (hi,) * 3 if (x * bands // n) % 2 else (lo,) * 3)


TEXTURES = {
    "quadrants.png":       tex_quadrants,
    "checker.png":         tex_checker,
    "alpha_checker.png":   tex_alpha_checker,
    "stripes.png":         tex_stripes,
    "normal_bumps.png":    tex_bumps,
    "solid_red.png":       lambda: tex_solid((204, 13, 13)),
    "solid_white.png":     lambda: tex_solid((255, 255, 255)),
    "rough_bands.png":     lambda: tex_vertical_bands(25, 204),   # roughness 0.1 / 0.8
    "metal_bands.png":     lambda: tex_vertical_bands(0, 255),    # metallic 0 / 1
}

# 1x1 quad standing in the XZ plane (Genesis is Z-up), facing the camera (-Y).
# OBJ UVs: v = 1 at the top, so the top-left of the image is on the top-left corner.
QUAD_OBJ = """\
v -0.5 0 -0.5
v  0.5 0 -0.5
v  0.5 0  0.5
v -0.5 0  0.5
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 -1 0
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
"""


def write_assets():
    os.makedirs(TEX_DIR, exist_ok=True)
    for name, make in TEXTURES.items():
        with open(os.path.join(TEX_DIR, name), "wb") as f:
            f.write(make())
    with open(os.path.join(TEX_DIR, "quad.obj"), "w") as f:
        f.write(QUAD_OBJ)


def tex(name):
    return os.path.join(TEX_DIR, name)


# ---------------------------------------------------------------------------
# Scene objects. A test returns a list of them, placed left to right.
# ---------------------------------------------------------------------------

def S(surface):
    """Sphere, radius 0.5 (Nyx built-in sphere primitive)."""
    return ("sphere", surface, 0.0)


def Q(surface, turn=0.0):
    """1x1 quad mesh (OBJ) facing the camera, turned by `turn` degrees around the vertical axis."""
    return ("quad", surface, turn)


# ---------------------------------------------------------------------------
# Tests converted from gltf_tests (same number and as close as possible)
# gs.surfaces.Default is gs.surfaces.BSDF (color / metallic / roughness / opacity / emissive).
# ---------------------------------------------------------------------------

def base_color_red_green_blue(gs):
    """glTF 01. Metallic 0, roughness 0.5."""
    return [S(gs.surfaces.Default(color=c, metallic=0.0, roughness=0.5))
            for c in ((0.8, 0.05, 0.05), (0.05, 0.8, 0.05), (0.05, 0.05, 0.8))]


def metallic_0_05_1(gs):
    """glTF 02. Gold, roughness 0.2, metallic 0 -> 0.5 -> 1."""
    return [S(gs.surfaces.Default(color=GOLD, metallic=m, roughness=0.2)) for m in (0.0, 0.5, 1.0)]


def roughness_0_05_1(gs):
    """glTF 03. Gold, metallic 1, roughness 0 -> 0.5 -> 1."""
    return [S(gs.surfaces.Default(color=GOLD, metallic=1.0, roughness=r)) for r in (0.0, 0.5, 1.0)]


def base_color_texture_quadrants(gs):
    """glTF 04. Quad facing the camera: top-left red, top-right green, bottom-left blue,
    bottom-right yellow."""
    return [Q(gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=tex("quadrants.png")),
                                  roughness=0.5))]


def base_color_texture_on_sphere(gs):
    """glTF 05. Checker texture on a sphere. The sphere is Nyx's built-in primitive, so its UV
    layout (and so the checker layout) may differ from the glTF sphere."""
    return [S(gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=tex("checker.png")),
                                  roughness=0.5))]


def normal_map_off_on(gs):
    """glTF 06. Left: no normal map. Right: bump normal map (bumps should look convex)."""
    bumps = gs.textures.ImageTexture(image_path=tex("normal_bumps.png"), encoding="linear")
    return [S(gs.surfaces.Default(color=(0.8, 0.8, 0.8), roughness=0.3)),
            S(gs.surfaces.Default(color=(0.8, 0.8, 0.8), roughness=0.3, normal_texture=bumps))]


def emissive_none_orange_orange_x4_texture(gs):
    """glTF 07. Black base:
    1. no emission  2. emissive orange  3. emissive orange x 4 (Genesis has no strength parameter,
    values above 1 play that role, must be clearly brighter than 2)  4. emissive_texture of white /
    black stripes tinted orange with image_color -> orange glowing stripes."""
    stripes = gs.textures.ImageTexture(image_path=tex("stripes.png"), image_color=ORANGE)
    return [S(gs.surfaces.Default(color=(0, 0, 0))),
            S(gs.surfaces.Default(color=(0, 0, 0), emissive=ORANGE)),
            S(gs.surfaces.Default(color=(0, 0, 0), emissive=tuple(4 * c for c in ORANGE))),
            S(gs.surfaces.Default(color=(0, 0, 0), emissive_texture=stripes))]


def opacity_1_06_03_alpha_texture(gs):
    """glTF 08 equivalent. Genesis has no alpha mode (OPAQUE / BLEND / MASK), only opacity.
    1-3: orange, opacity 1 -> 0.6 -> 0.3 (should get more and more see-through).
    4: diffuse texture with alpha 0.75 / 0.25 checker (Genesis takes opacity from the texture
       alpha), double-sided: 0.25 cells should be much more see-through than 0.75 cells."""
    return [*(S(gs.surfaces.Default(color=ORANGE, opacity=a, roughness=0.5)) for a in (1.0, 0.6, 0.3)),
            S(gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=tex("alpha_checker.png")),
                                  roughness=0.5, double_sided=True))]


def double_sided_on_off(gs):
    """glTF 09. Two quads seen from the BACK: left double_sided=True (visible),
    right double_sided=False (culled, you should only see the ground behind)."""
    quad = gs.textures.ImageTexture(image_path=tex("quadrants.png"))
    return [Q(gs.surfaces.Default(diffuse_texture=quad, double_sided=True), turn=180),
            Q(gs.surfaces.Default(diffuse_texture=quad, double_sided=False), turn=180)]


def rough_metal_values_colortextures_roughtex_metaltex(gs):
    """glTF 19/20 equivalent. Genesis has separate roughness_texture and metallic_texture
    (one channel each) instead of glTF's packed metallicRoughness texture.
    1. gold, metallic 1, roughness 0.2 as values
    2. same, given as ColorTexture (must look identical to 1)
    3. metallic 1, roughness_texture: vertical bands roughness 0.1 / 0.8
    4. roughness 0.2, metallic_texture: vertical bands metallic 0 / 1"""
    T = gs.textures
    return [S(gs.surfaces.Default(color=GOLD, metallic=1.0, roughness=0.2)),
            S(gs.surfaces.Default(color=GOLD, metallic_texture=T.ColorTexture(color=(1.0,)),
                                  roughness_texture=T.ColorTexture(color=(0.2,)))),
            S(gs.surfaces.Default(color=GOLD, metallic=1.0,
                                  roughness_texture=T.ImageTexture(image_path=tex("rough_bands.png"),
                                                                   encoding="linear"))),
            S(gs.surfaces.Default(color=GOLD, roughness=0.2,
                                  metallic_texture=T.ImageTexture(image_path=tex("metal_bands.png"),
                                                                  encoding="linear")))]


def combined_all_basic_features(gs):
    """glTF 23 equivalent, using only what Genesis surfaces have.
    Left sphere and middle quad, same surface: checker colour texture, bump normal map,
    roughness 0.3, metallic 0, orange glowing stripes (emissive_texture).
    Right: alpha checker quad seen from the back, double-sided -> visible, with see-through cells."""
    T = gs.textures
    hero = gs.surfaces.Default(
        diffuse_texture=T.ImageTexture(image_path=tex("checker.png")),
        normal_texture=T.ImageTexture(image_path=tex("normal_bumps.png"), encoding="linear"),
        emissive_texture=T.ImageTexture(image_path=tex("stripes.png"), image_color=ORANGE),
        roughness=0.3, metallic=0.0)
    leaf = gs.surfaces.Default(diffuse_texture=T.ImageTexture(image_path=tex("alpha_checker.png")),
                               roughness=0.5, double_sided=True)
    return [S(hero), Q(hero, turn=-20), Q(leaf, turn=200)]


# ---------------------------------------------------------------------------
# Genesis-only features (no glTF equivalent)
# ---------------------------------------------------------------------------

def presets_plastic_rough_smooth_reflective(gs):
    """Same blue colour. Plastic (no roughness set), Rough (roughness 1, ior 1.5),
    Smooth (roughness 0.1, ior 1.5), Reflective (roughness 0.01, ior 2):
    should go from matte to mirror-like, left to right (except Plastic, which uses the default)."""
    c = (0.1, 0.3, 0.8)
    return [S(cls(color=c)) for cls in (gs.surfaces.Plastic, gs.surfaces.Rough,
                                        gs.surfaces.Smooth, gs.surfaces.Reflective)]


def metals_iron_aluminium_copper_gold(gs):
    """Metal presets (roughness 0.1 by default): Iron, Aluminium, Copper, Gold."""
    return [S(cls()) for cls in (gs.surfaces.Iron, gs.surfaces.Aluminium, gs.surfaces.Copper, gs.surfaces.Gold)]


def metals_brass_titanium_vanadium_lithium(gs):
    """The other metal_type values, through gs.surfaces.Metal(metal_type=...)."""
    return [S(gs.surfaces.Metal(metal_type=t)) for t in ("brass", "titanium", "vanadium", "lithium")]


def metal_gold_red_color_rough05(gs):
    """1. Gold()  2. Gold(color=red): the colour should win -> red metal
    3. Gold(roughness=0.5): blurrier gold."""
    return [S(gs.surfaces.Gold()),
            S(gs.surfaces.Gold(color=(0.9, 0.1, 0.1))),
            S(gs.surfaces.Gold(roughness=0.5))]


def glass_clear_rough_ior12_water(gs):
    """Glass(white), Glass(white, roughness 0.3), Glass(white, ior 1.2), Water().
    In Genesis these are transparent (refractive). Nyx has transmission / IOR, but the plugin
    doesn't pass them: expect opaque spheres if it is not converted."""
    w = (1.0, 1.0, 1.0)
    return [S(gs.surfaces.Glass(color=w)),
            S(gs.surfaces.Glass(color=w, roughness=0.3)),
            S(gs.surfaces.Glass(color=w, ior=1.2)),
            S(gs.surfaces.Water())]


def bsdf_trans_none_specular_diffuse(gs):
    """BSDF with specular_trans 0 / 1 / diffuse_trans 1 (same white-ish colour).
    2 should look like clear glass, 3 like frosted / translucent."""
    c = (0.9, 0.9, 0.9)
    return [S(gs.surfaces.BSDF(color=c, roughness=0.1)),
            S(gs.surfaces.BSDF(color=c, roughness=0.1, specular_trans=1.0)),
            S(gs.surfaces.BSDF(color=c, roughness=0.1, diffuse_trans=1.0))]


def emission_surface_color_x4_texture(gs):
    """gs.surfaces.Emission (a light-emitting surface):
    1. Emission(color=orange)  2. Emission(emissive=orange x 4), brighter than 1
    3. Emission(emissive_texture=white / black stripes tinted orange)."""
    return [S(gs.surfaces.Emission(color=ORANGE)),
            S(gs.surfaces.Emission(emissive=tuple(4 * c for c in ORANGE))),
            S(gs.surfaces.Emission(emissive_texture=gs.textures.ImageTexture(
                image_path=tex("stripes.png"), image_color=ORANGE)))]


def color_value_colortex_imagetex_tinted(gs):
    """4 ways to say the same red in Genesis, all four should look identical:
    1. color=red  2. diffuse_texture=ColorTexture(red)  3. diffuse_texture = red 1x1 PNG
    4. diffuse_texture = white 1x1 PNG with image_color=red (tint)."""
    red = (0.8, 0.05, 0.05)
    T = gs.textures
    return [S(gs.surfaces.Default(color=red, roughness=0.5)),
            S(gs.surfaces.Default(diffuse_texture=T.ColorTexture(color=red), roughness=0.5)),
            S(gs.surfaces.Default(diffuse_texture=T.ImageTexture(image_path=tex("solid_red.png")), roughness=0.5)),
            S(gs.surfaces.Default(diffuse_texture=T.ImageTexture(image_path=tex("solid_white.png"), image_color=red),
                                  roughness=0.5))]


def texture_from_file_vs_array(gs):
    """Same checker texture, left loaded from a PNG file (image_path), right given as a numpy
    array (image_array, no file). Both should look identical."""
    import numpy as np
    from PIL import Image
    arr = np.array(Image.open(tex("checker.png")))
    T = gs.textures
    return [S(gs.surfaces.Default(diffuse_texture=T.ImageTexture(image_path=tex("checker.png")), roughness=0.5)),
            S(gs.surfaces.Default(diffuse_texture=T.ImageTexture(image_array=arr), roughness=0.5))]


def opacity_texture_stripes(gs):
    """Orange plastic with an opacity_texture of horizontal stripes (opaque / fully transparent),
    double-sided: you should see through every other stripe. Left: same without opacity_texture."""
    import numpy as np
    stripes = np.array([[255 if (y * 8 // 128) % 2 == 0 else 0] * 128 for y in range(128)], dtype=np.uint8)
    T = gs.textures
    return [S(gs.surfaces.Plastic(color=ORANGE, roughness=0.5, double_sided=True)),
            S(gs.surfaces.Plastic(color=ORANGE, roughness=0.5, double_sided=True,
                                  opacity_texture=T.ImageTexture(image_array=stripes, encoding="linear")))]


TESTS = [
    # converted from gltf_tests
    ("01_base_color_red_green_blue",                         base_color_red_green_blue),
    ("02_metallic_0_05_1",                                   metallic_0_05_1),
    ("03_roughness_0_05_1",                                  roughness_0_05_1),
    ("04_base_color_texture_quadrants",                      base_color_texture_quadrants),
    ("05_base_color_texture_on_sphere",                      base_color_texture_on_sphere),
    ("06_normal_map_off_on",                                 normal_map_off_on),
    ("07_emissive_none_orange_orange_x4_texture",            emissive_none_orange_orange_x4_texture),
    ("08_opacity_1_06_03_alpha_texture",                     opacity_1_06_03_alpha_texture),
    ("09_double_sided_on_off",                               double_sided_on_off),
    ("19_rough_metal_values_colortextures_roughtex_metaltex", rough_metal_values_colortextures_roughtex_metaltex),
    ("23_combined_all_basic_features",                       combined_all_basic_features),
    # Genesis only
    ("30_presets_plastic_rough_smooth_reflective",           presets_plastic_rough_smooth_reflective),
    ("31_metals_iron_aluminium_copper_gold",                 metals_iron_aluminium_copper_gold),
    ("32_metals_brass_titanium_vanadium_lithium",            metals_brass_titanium_vanadium_lithium),
    ("33_metal_gold_red_color_rough05",                      metal_gold_red_color_rough05),
    ("34_glass_clear_rough_ior12_water",                     glass_clear_rough_ior12_water),
    ("35_bsdf_trans_none_specular_diffuse",                  bsdf_trans_none_specular_diffuse),
    ("36_emission_surface_color_x4_texture",                 emission_surface_color_x4_texture),
    ("37_color_value_colortex_imagetex_tinted",              color_value_colortex_imagetex_tinted),
    ("38_texture_from_file_vs_array",                        texture_from_file_vs_array),
    ("39_opacity_texture_stripes",                           opacity_texture_stripes),
]


# ---------------------------------------------------------------------------

def add_object(gs, scene, kind, surface, turn, x):
    if kind == "sphere":
        morph = gs.morphs.Sphere(pos=(x, 0.0, HEIGHT), radius=RADIUS, fixed=True, collision=False)
    else:
        morph = gs.morphs.Mesh(file=tex("quad.obj"), pos=(x, 0.0, HEIGHT), euler=(0.0, 0.0, turn),
                               fixed=True, collision=False)
    scene.add_entity(morph=morph, surface=surface)


def render(name, make_objects):
    import genesis as gs
    import gs_nyx.nyx_py_renderer as npr
    import gs_nyx.nyx_py_sdk as nps
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from PIL import Image

    gs.init()
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane(plane_size=(20.0, 20.0)))

    objects = make_objects(gs)
    for i, (kind, surface, turn) in enumerate(objects):
        add_object(gs, scene, kind, surface, turn, (i - (len(objects) - 1) / 2) * SPACING)

    env_map            = nps.EnvironmentMapAsset()
    env_map.texture    = ENV_MAP
    env_map.layout     = nps.EEnvMapLayout.LongLat
    env_map.multiplier = 2

    cam = scene.add_sensor(NyxCameraOptions(
        res=CAM_RES, pos=CAM_POS, lookat=CAM_LOOKAT, fov=CAM_FOV, spp=64,
        render_mode=getattr(npr.ERenderMode, MODES[0]), env_maps=(env_map,)))
    scene.build(n_envs=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, mode in enumerate(MODES):
        if i > 0:
            cam._shared_metadata.renderer._renderer.set_render_mode(getattr(npr.ERenderMode, mode))
        scene.step()  # the camera renders during step
        out = os.path.join(OUT_DIR, "%s_%d_%s.png" % (name, i + 1, mode))
        Image.fromarray(cam.read().rgb[0].cpu().numpy()).save(out)
        print("Saved", out)


def main():
    tests = dict(TESTS)
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":  # internal: render one test in this process
        render(sys.argv[2], tests[sys.argv[2]])
        return
    filters = sys.argv[1:]
    names = [n for n, _ in TESTS if not filters or any(p in n for p in filters)]
    if not names:
        sys.exit("No test matches %s. Available:\n  %s" % (filters, "\n  ".join(n for n, _ in TESTS)))
    write_assets()
    for n in names:
        print("==>", n)
        if subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", n]).returncode != 0:
            print("    FAILED:", n)


if __name__ == "__main__":
    main()
