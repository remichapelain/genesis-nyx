"""Parity tests between the two ways a material reaches Nyx: a glTF material and a Genesis surface.

Every test renders the SAME objects twice in one image:

    top row     glTF material      the material is written in the .glb, read by Nyx's glTF importer
    bottom row  Genesis surface    the same .glb geometry WITHOUT material, with a gs.surfaces object,
                                   converted by gs_nyx_plugin (nyx_material_utils.build_override)

Both rows use the same meshes (same UVs, normals and tangents), so any difference between a
top object and the one right below it comes from the material path only.

Only features that BOTH paths are supposed to support are here. glTF-only features (samplers,
anisotropy, occlusion, metallic-roughness textures, alpha MASK, normal scale...) stay in
gltf_tests / gltf_transparency_tests, Genesis-only ones (presets, metals...) in surface_tests.
Test numbers follow gltf_tests where a test was taken from it; 24+ are transmission tests.

The scene is symmetrical top / bottom: no ground, the camera is level and aims at the middle,
and a striped wall stands behind (vertical stripes, so both rows have the same background).

Usage:
    python examples/material_parity_tests/render_material_parity_tests.py            # all tests
    python examples/material_parity_tests/render_material_parity_tests.py 02 07      # names containing "02" or "07"
    python examples/material_parity_tests/render_material_parity_tests.py normal glass

Output: out/<test>_<n>_<mode>.png. The .glb files go to generated/, the textures to textures/.
Each test is rendered in its own Python process, so a crash on one doesn't stop the others.
"""

import functools
import os
import subprocess
import sys

HERE    = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(HERE, "generated")
TEX_DIR = os.path.join(HERE, "textures")
OUT_DIR = os.path.join(HERE, "out")
ENV_MAP = os.path.join(HERE, "..", "assets", "green_sanctuary_4k.hdr")

# The glTF builder, PNG encoder, geometry and textures come from the glTF tests.
sys.path.insert(0, os.path.join(HERE, "..", "gltf_tests"))
sys.path.insert(0, os.path.join(HERE, "..", "gltf_transparency_tests"))
import make_gltf_tests as mk                 # noqa: E402
import make_gltf_transparency_tests as mkt   # noqa: E402

mk.TEX_DIR = TEX_DIR  # (the import above pointed it at the transparency tests)

MODES   = ("Forward", "FastPathTracer", "RefPathTracer")
CAM_RES = (1280, 720)
# Far away with a narrow lens: both rows are seen almost straight on (+-3 degrees), so an
# object and its twin on the other row show nearly the same reflections.
CAM_POS    = (0.0, -11.0, 0.0)   # glTF +Z (front) ends up facing Genesis -Y
CAM_LOOKAT = (0.0, 0.0, 0.0)
CAM_FOV    = 15.0

# Objects are 1 unit wide, 1.2 apart, the rows 1.2 apart too (centres at +-ROW_Z).
SPACING    = 1.2
ROW_Z      = 0.6
BACKDROP_Y = 1.5    # the wall, behind the objects
LABELS     = True   # write the title, the row names and the column captions on the saved images

GOLD   = (1.0, 0.77, 0.34)
GREY   = (0.8, 0.8, 0.8)
WHITE  = (1.0, 1.0, 1.0)
ORANGE = (1.0, 0.4, 0.0)


# ---------------------------------------------------------------------------
# Textures: one PNG per name, embedded in the .glb (top row) and read from textures/ (bottom row)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def height_field():
    return mk.height_field_regions()


TEXTURES = {
    "quadrants":         mk.tex_quadrants,
    "checker":           mk.tex_checker,
    "alpha_checker":     mk.tex_alpha_checker,       # orange, alpha 0.75 / 0.25
    "stripes":           mk.tex_stripes,             # white / black, emissive mask
    "normal_bumps_gl":   mk.tex_bumps,
    "normal_bumps_dx":   lambda: mk.tex_bumps(directx=True),
    "normal_hires_gl":   lambda: mk.tex_normal_from_height(height_field()),
    "height_reference":  lambda: mk.tex_height_preview(height_field()),
}


@functools.lru_cache(maxsize=None)
def png_bytes(name):
    return TEXTURES[name]()


def tex_path(name, rewrite=False):
    """Path of textures/<name>.png, written if missing (or always with rewrite)."""
    path = os.path.join(TEX_DIR, name + ".png")
    if rewrite or not os.path.exists(path):
        os.makedirs(TEX_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(png_bytes(name))
    return path


def gtex(g, name):
    """glTF side: the texture embedded in the Gltf `g` (once per file). Returns {"index": i}."""
    cache = g.__dict__.setdefault("parity_textures", {})
    if name not in cache:
        tex_path(name, rewrite=True)
        cache[name] = g.texture(png_bytes(name), name)
    return cache[name]


def img(gs, name, linear=False, tint=None):
    """Genesis side: the same texture as a gs.textures.ImageTexture read from textures/."""
    kwargs = {"image_path": tex_path(name)}
    if linear:
        kwargs["encoding"] = "linear"
    if tint is not None:
        kwargs["image_color"] = tint
    return gs.textures.ImageTexture(**kwargs)


# ---------------------------------------------------------------------------
# Geometry shared by both rows
# ---------------------------------------------------------------------------

GEOMETRY = {
    "sphere":                  mk.sphere,
    "sphere_tangent_w_flipped": lambda: mk.flip_tangent_w(mk.sphere()),
    "quad":                    mk.quad,
    "quad_11":                 lambda: mk.quad(1.1),
    "quad_09":                 lambda: mk.quad(0.9),
    # 0.3 thick, its FRONT face where quad_09 is, so a slab of 0.3 and this box can be compared.
    "box_09_thick_03":         lambda: mk.box(0.9, 0.9, 0.3, center=(0, 0, -0.15)),
}


def col(geo, gltf, surface, turn=0.0):
    """One column of a test: the same object on both rows.
    geo:     key of GEOMETRY
    gltf:    function(g) -> glTF material index, for the top row (g is a mk.Gltf)
    surface: function(gs) -> gs.surfaces object, for the bottom row
    turn:    rotation around the vertical axis, in degrees (180 = seen from the back)"""
    return dict(geo=geo, gltf=gltf, surface=surface, turn=turn)


# ---------------------------------------------------------------------------
# The tests. Each returns its columns, left to right. In every column the first function is
# the glTF material and the second one the Genesis surface that should look the same.
# The Genesis surfaces always set metallic and roughness: what is left unset falls back to
# the material of the mesh file, which has none here.
# ---------------------------------------------------------------------------

#   01   Base colour
#   02   Metallic ramp
#   03   Roughness ramp
#   04   Base colour texture: UV orientation
#   05   Base colour texture on a sphere, plain and tinted
#   06a  Normal map (OpenGL) off / on / tangent w flipped
#   06b  Normal map (DirectX) off / on / tangent w flipped
#   06d  Normal map: high-res 1024 on a plane, vs height reference
#   07   Emission: none / colour / colour x4 / texture
#   08   Opacity: opaque / 0.3 / alpha texture
#   09   Double-sided vs single-sided (back faces)
#   24   Transmission 0 / 0.5 / 1
#   25   Glass: clear / rough / tinted / thin-walled
#   26   Glass IOR 1 / 1.33 / 1.5 / 2.4
#   27a  Refraction mode, quad: sidedness x thickness
#   27b  Refraction mode, box: sidedness x thickness
#   27c  Refraction mode, sphere: sidedness x thickness

# ================================================================================================
# 01  | Base colour
# ================================================================================================
def base_color_red_green_blue():
    """Plain colours, metallic 0, roughness 0.5."""
    return [col("sphere",
                lambda g, c=c: g.pbr(c + (1,), 0.0, 0.5),
                lambda gs, c=c: gs.surfaces.Default(color=c, metallic=0.0, roughness=0.5))
            for c in ((0.8, 0.05, 0.05), (0.05, 0.8, 0.05), (0.05, 0.05, 0.8))]


# ================================================================================================
# 02  | Metallic ramp
# ================================================================================================
def metallic_0_05_1():
    """Gold, roughness 0.2, metallic 0 -> 0.5 -> 1 left to right."""
    return [col("sphere",
                lambda g, m=m: g.pbr(GOLD + (1,), m, 0.2),
                lambda gs, m=m: gs.surfaces.Default(color=GOLD, metallic=m, roughness=0.2))
            for m in (0.0, 0.5, 1.0)]


# ================================================================================================
# 03  | Roughness ramp
# ================================================================================================
def roughness_0_05_1():
    """Gold, metallic 1, roughness 0 -> 0.5 -> 1 left to right."""
    return [col("sphere",
                lambda g, r=r: g.pbr(GOLD + (1,), 1.0, r),
                lambda gs, r=r: gs.surfaces.Default(color=GOLD, metallic=1.0, roughness=r))
            for r in (0.0, 0.5, 1.0)]


# ================================================================================================
# 04  | Base colour texture: UV orientation
# ================================================================================================
def base_color_texture_quadrants():
    """Quad facing the camera: top-left red, top-right green, bottom-left blue, bottom-right yellow."""
    return [col("quad",
                lambda g: g.pbr(WHITE + (1,), 0.0, 0.5, baseColorTexture=gtex(g, "quadrants")),
                lambda gs: gs.surfaces.Default(diffuse_texture=img(gs, "quadrants"), metallic=0.0, roughness=0.5))]


# ================================================================================================
# 05  | Base colour texture on a sphere, plain and tinted
# ================================================================================================
def base_color_texture_on_sphere_plain_tinted():
    """Checker texture on a sphere. Left: as is. Right: multiplied by an orange tint
    (glTF baseColorFactor, Genesis image_color)."""
    tint = (1.0, 0.5, 0.2)
    return [col("sphere",
                lambda g: g.pbr(WHITE + (1,), 0.0, 0.5, baseColorTexture=gtex(g, "checker")),
                lambda gs: gs.surfaces.Default(diffuse_texture=img(gs, "checker"), metallic=0.0, roughness=0.5)),
            col("sphere",
                lambda g: g.pbr(tint + (1,), 0.0, 0.5, baseColorTexture=gtex(g, "checker")),
                lambda gs: gs.surfaces.Default(diffuse_texture=img(gs, "checker", tint=tint),
                                               metallic=0.0, roughness=0.5))]


def normal_map_columns(texture):
    """No normal map / normal map / normal map on a sphere whose TANGENT.w is -1."""
    def plain_gltf(g):
        return g.pbr(GREY + (1,), 0.0, 0.8)

    def plain_surface(gs):
        return gs.surfaces.Default(color=GREY, metallic=0.0, roughness=0.8)

    def bumps_gltf(g):
        return g.pbr(GREY + (1,), 0.0, 0.8, normalTexture=gtex(g, texture))

    def bumps_surface(gs):
        return gs.surfaces.Default(color=GREY, metallic=0.0, roughness=0.8,
                                   normal_texture=img(gs, texture, linear=True))

    return [col("sphere", plain_gltf, plain_surface),
            col("sphere", bumps_gltf, bumps_surface),
            col("sphere_tangent_w_flipped", bumps_gltf, bumps_surface)]


# ================================================================================================
# 06a | Normal map (OpenGL) off / on / tangent w flipped
# ================================================================================================
def normal_map_gl_off_on_tangent_w_flip():
    """OpenGL normal map (green = up), the glTF convention.
    Left: no normal map. Middle: round CONVEX bumps, lit from the same side as the sphere itself.
    Right: same map, TANGENT.w = -1 (bitangent flipped): bumps lit from the wrong side up / down."""
    return normal_map_columns("normal_bumps_gl")


# ================================================================================================
# 06b | Normal map (DirectX) off / on / tangent w flipped
# ================================================================================================
def normal_map_dx_off_on_tangent_w_flip():
    """DirectX normal map (green inverted). Neither glTF nor Genesis can declare it, so both read
    it as OpenGL: middle = bumps lit from the wrong side up / down (same as 06a right),
    right = correct again (same as 06a middle)."""
    return normal_map_columns("normal_bumps_dx")


# ================================================================================================
# 06d | Normal map: high-res 1024 on a plane, vs height reference
# ================================================================================================
def normal_map_gl_1024_plane_vs_height_reference():
    """Left: grey plane with ONLY a 1024x1024 normal map. Right: the same height field as a colour
    texture (light = high): the shape the left plane must show. Regions: top-left raised domes,
    top-right sunken dents, bottom-left pyramids, bottom-right fine ripples."""
    return [col("quad_11",
                lambda g: g.pbr(GREY + (1,), 0.0, 0.35, normalTexture=gtex(g, "normal_hires_gl")),
                lambda gs: gs.surfaces.Default(color=GREY, metallic=0.0, roughness=0.35,
                                               normal_texture=img(gs, "normal_hires_gl", linear=True))),
            col("quad_11",
                lambda g: g.pbr(WHITE + (1,), 0.0, 0.9, baseColorTexture=gtex(g, "height_reference")),
                lambda gs: gs.surfaces.Default(diffuse_texture=img(gs, "height_reference"),
                                               metallic=0.0, roughness=0.9))]


# ================================================================================================
# 07  | Emission: none / colour / colour x4 / texture
# ================================================================================================
def emissive_none_orange_orange_x4_texture():
    """Black base, metallic 0, roughness 0.5. Left to right:
    1. no emission
    2. orange emission
    3. orange x 4: glTF KHR_materials_emissive_strength 4, Genesis emissive values above 1
    4. white / black stripes texture tinted orange (glTF emissiveFactor, Genesis image_color):
       orange glowing stripes, black ones in between.
    The plugin turns an emissive Genesis surface into a pure emitter (black, metalness 1), so the
    bottom row is expected to lose the faint reflections the top row keeps."""
    black = (0.0, 0.0, 0.0)
    x4 = tuple(4 * c for c in ORANGE)
    strength = {"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}}

    def gltf_x4(g):
        mkt.use_ext(g, "KHR_materials_emissive_strength")
        return g.pbr(black + (1,), 0.0, 0.5, emissiveFactor=list(ORANGE), extensions=strength)

    return [col("sphere",
                lambda g: g.pbr(black + (1,), 0.0, 0.5),
                lambda gs: gs.surfaces.Default(color=black, metallic=0.0, roughness=0.5)),
            col("sphere",
                lambda g: g.pbr(black + (1,), 0.0, 0.5, emissiveFactor=list(ORANGE)),
                lambda gs: gs.surfaces.Default(color=black, metallic=0.0, roughness=0.5, emissive=ORANGE)),
            col("sphere",
                gltf_x4,
                lambda gs: gs.surfaces.Default(color=black, metallic=0.0, roughness=0.5, emissive=x4)),
            col("sphere",
                lambda g: g.pbr(black + (1,), 0.0, 0.5, emissiveFactor=list(ORANGE),
                                emissiveTexture=gtex(g, "stripes")),
                lambda gs: gs.surfaces.Default(color=black, metallic=0.0, roughness=0.5,
                                               emissive_texture=img(gs, "stripes", tint=ORANGE)))]


# ================================================================================================
# 08  | Opacity: opaque / 0.3 / alpha texture
# ================================================================================================
def opacity_1_03_alpha_checker():
    """Orange, roughness 0.5. Left to right:
    1. fully solid (glTF: alpha 0.3 but alphaMode OPAQUE, so the alpha is ignored; Genesis: opacity 1)
    2. see-through, 30 % (glTF alphaMode BLEND with alpha 0.3; Genesis opacity 0.3)
    3. checker texture whose ALPHA alternates 0.75 / 0.25, blended, double-sided: cells slightly /
       very see-through, the inside of the far half visible through them.
    Genesis has no alpha cutoff, so glTF's alphaMode MASK is not compared here."""
    orange = (1.0, 0.5, 0.0)
    return [col("sphere",
                lambda g: g.pbr(orange + (0.3,), 0.0, 0.5, alphaMode="OPAQUE"),
                lambda gs: gs.surfaces.Default(color=orange, opacity=1.0, metallic=0.0, roughness=0.5)),
            col("sphere",
                lambda g: g.pbr(orange + (0.3,), 0.0, 0.5, alphaMode="BLEND"),
                lambda gs: gs.surfaces.Default(color=orange, opacity=0.3, metallic=0.0, roughness=0.5)),
            col("sphere",
                lambda g: g.pbr(WHITE + (1,), 0.0, 0.5, alphaMode="BLEND", doubleSided=True,
                                baseColorTexture=gtex(g, "alpha_checker")),
                lambda gs: gs.surfaces.Default(diffuse_texture=img(gs, "alpha_checker"), double_sided=True,
                                               metallic=0.0, roughness=0.5))]


# ================================================================================================
# 09  | Double-sided vs single-sided (back faces)
# ================================================================================================
def double_sided_on_off():
    """Two quads seen from the BACK. Left: double-sided, visible (texture mirrored: green top-left,
    red top-right). Right: single-sided, culled: only the wall behind."""
    return [col("quad",
                lambda g: g.pbr(WHITE + (1,), 0.0, 0.5, baseColorTexture=gtex(g, "quadrants"), doubleSided=True),
                lambda gs: gs.surfaces.Default(diffuse_texture=img(gs, "quadrants"), double_sided=True,
                                               metallic=0.0, roughness=0.5),
                turn=180),
            col("quad",
                lambda g: g.pbr(WHITE + (1,), 0.0, 0.5, baseColorTexture=gtex(g, "quadrants")),
                lambda gs: gs.surfaces.Default(diffuse_texture=img(gs, "quadrants"), double_sided=False,
                                               metallic=0.0, roughness=0.5),
                turn=180)]


# ---- transmission ------------------------------------------------------------------
# glTF: KHR_materials_transmission + _ior, and _volume (thicknessFactor) for a solid object.
# Genesis: BSDF(specular_trans=...) or Glass; the plugin makes a closed single-sided mesh a
# solid volume and a double_sided one a thin wall.

# ================================================================================================
# 24  | Transmission 0 / 0.5 / 1
# ================================================================================================
def transmission_0_05_1():
    """White smooth solid spheres, IOR 1.5, transmission 0 (opaque white) -> 0.5 -> 1 (clear
    glass ball: the stripes of the wall seen through it are flipped and distorted)."""
    return [col("sphere",
                lambda g, t=t: mkt.glass(g, transmission=t, ior=1.5, thickness=1.0),
                lambda gs, t=t: gs.surfaces.BSDF(color=WHITE, metallic=0.0, roughness=0.0,
                                                 specular_trans=t, ior=1.5))
            for t in (0.0, 0.5, 1.0)]


# ================================================================================================
# 25  | Glass: clear / rough / tinted / thin-walled
# ================================================================================================
def glass_clear_rough_tinted_thin():
    """IOR 1.5. Left to right:
    1. clear solid glass ball
    2. roughness 0.3: frosted, the wall behind is blurred
    3. blue tint (base colour): the wall seen through it is tinted
    4. thin-walled (double-sided, no volume): a soap bubble, the wall seen straight through,
       not flipped."""
    blue = (0.2, 0.6, 1.0)
    return [col("sphere",
                lambda g: mkt.glass(g, ior=1.5, thickness=1.0),
                lambda gs: gs.surfaces.Glass(color=WHITE, roughness=0.0, ior=1.5)),
            col("sphere",
                lambda g: mkt.glass(g, rough=0.3, ior=1.5, thickness=1.0),
                lambda gs: gs.surfaces.Glass(color=WHITE, roughness=0.3, ior=1.5)),
            col("sphere",
                lambda g: mkt.glass(g, color=blue, ior=1.5, thickness=1.0),
                lambda gs: gs.surfaces.Glass(color=blue, roughness=0.0, ior=1.5)),
            col("sphere",
                lambda g: mkt.glass(g, ior=1.5, doubleSided=True),
                lambda gs: gs.surfaces.Glass(color=WHITE, roughness=0.0, ior=1.5, double_sided=True))]


# ================================================================================================
# 26  | Glass IOR 1 / 1.33 / 1.5 / 2.4
# ================================================================================================
def glass_ior_1_133_15_24():
    """Clear solid glass balls, IOR 1.0 (no bending, no reflection: almost invisible), 1.33
    (water), 1.5 (glass), 2.4 (diamond: strongest bending and reflections)."""
    return [col("sphere",
                lambda g, i=i: mkt.glass(g, ior=i, thickness=1.0),
                lambda gs, i=i: gs.surfaces.Glass(color=WHITE, roughness=0.0, ior=i))
            for i in (1.0, 1.33, 1.5, 2.4)]


# ---- refraction mode: sidedness x thickness ------------------------------------------
# A transmissive surface refracts in one of three ways:
#   THIN-WALLED    a sheet with no inside: the wall is seen straight through, where it really is.
#   SLAB           a sheet declared to have a thickness: the wall is SHIFTED sideways when the
#                  sheet is tilted (in and out through two parallel faces).
#   CLOSED VOLUME  a solid: light enters by one face and leaves by the real exit face of the
#                  mesh. A sphere flips the wall (lens), a box shifts it like a slab.
# What picks the mode:
#   glTF spec   the thickness only (KHR_materials_volume): none = thin-walled, set = volume.
#   Nyx         the sidedness too: single-sided + thickness = closed volume (falls back to
#               thin-walled when the mesh has no exit face, e.g. a quad), doubleSided +
#               thickness = slab, no thickness = thin-walled.
#   Genesis     has no volume thickness (Glass.thickness is the subsurface layer, the plugin
#               drops it), only double_sided: the plugin makes a single-sided surface a closed
#               volume and a double_sided one thin-walled.
# So the bottom row can only show two of the modes, and each test has the same four columns:
#               glTF (top row)                           Genesis (bottom row)
#   1. single-sided, no thickness   thin-walled          closed volume  <- differs by design
#   2. single-sided, thickness      closed volume        closed volume
#   3. double-sided, no thickness   thin-walled          thin-walled
#   4. double-sided, thickness      slab                 thin-walled    <- differs by design
# The Genesis surfaces of 2 and 4 are still given the thickness, so the bottom row changes by
# itself the day the plugin maps it. Clear glass, IOR 1.5. Quads and boxes are turned 45
# degrees, to show the sideways shift of the wall stripes (about a third of a stripe for a
# thickness of 0.3).

def refraction_mode_columns(geo, thickness, turn=0.0):
    """The four sidedness x thickness combinations of one shape."""
    def column(double_sided, thick):
        def gltf(g):
            extra = {"doubleSided": True} if double_sided else {}
            return mkt.glass(g, ior=1.5, thickness=thickness if thick else None, **extra)

        def surface(gs):
            extra = {"thickness": thickness} if thick else {}
            return gs.surfaces.Glass(color=WHITE, roughness=0.0, ior=1.5, double_sided=double_sided, **extra)

        return col(geo, gltf, surface, turn=turn)

    return [column(False, False), column(False, True), column(True, False), column(True, True)]


REFRACTION_MODE_CAPTIONS_QUAD = ["single-sided - no thickness\nthin", "single-sided - thickness\nthin", "double-sided - no thickness\nthin", "double-sided - thickness\nthick"]
REFRACTION_MODE_CAPTIONS_BOX = ["single-sided - no thickness\nthin", "single-sided - thickness\nmesh", "double-sided - no thickness\nthin", "double-sided\nthickness\nthick"]
REFRACTION_MODE_CAPTIONS_SPHERE = ["single-sided - no thickness\nthin", "single-sided - thickness\nmesh", "double-sided - no thickness\nthin", "double-sided\nthickness\nthick"]


# ================================================================================================
# 27a | Refraction mode, quad: sidedness x thickness
# ================================================================================================
def refraction_mode_quad():
    """Flat quad (an open sheet, no inside), turned 45 degrees, thickness 0.3 where set.
    glTF row:    1. thin-walled, stripes straight through.
                 2. asks for a closed volume but a quad has no exit face: falls back to
                    thin-walled, same as 1.
                 3. thin-walled, same as 1.
                 4. SLAB: the stripes are shifted sideways. Must match the box of 27b column 2.
    Genesis row: 1-2. single-sided, exported as a closed volume: same fallback, thin-walled.
                 3-4. thin-walled.
    So only column 4 is expected to differ between the rows."""
    return refraction_mode_columns("quad_09", 0.3, turn=45)


# ================================================================================================
# 27b | Refraction mode, box: sidedness x thickness
# ================================================================================================
def refraction_mode_box():
    """Real box 0.3 deep (a slab modelled with geometry), turned 45 degrees, thickness 0.3 where
    set. Its front face is where the quad of 27a is.
    glTF row:    1. thin-walled: every face is a sheet, stripes straight through (the far and
                    side faces only add reflections).
                 2. CLOSED VOLUME: enters the front face, leaves by the back one, stripes
                    shifted sideways. The reference for the slab of 27a column 4.
                 3. thin-walled, like 1 (back faces are drawn too).
                 4. SLAB on each face: a shift at the front face and another one at the back
                    face, so about twice the shift of 2. A solid must be single-sided.
    Genesis row: 1-2. closed volume, like glTF 2.
                 3-4. thin-walled, like glTF 3."""
    return refraction_mode_columns("box_09_thick_03", 0.3, turn=45)


# ================================================================================================
# 27c | Refraction mode, sphere: sidedness x thickness
# ================================================================================================
def refraction_mode_sphere():
    """Solid sphere, thickness 1.0 (its diameter) where set.
    glTF row:    1. thin-walled: a soap bubble, stripes straight through.
                 2. CLOSED VOLUME: a glass ball, the stripes are FLIPPED left / right (lens).
                 3. thin-walled, like 1.
                 4. SLAB at every point: not the lens image of 2, a magnified and distorted wall.
    Genesis row: 1-2. closed volume, like glTF 2.
                 3-4. thin-walled, like glTF 3."""
    return refraction_mode_columns("sphere", 1.0)


# name, test, title written at the top of the image, caption of each column (written between the
# two rows, "\n" starts a second line). Keep the captions in the order of the columns.
TESTS = [
    ("01_base_color_red_green_blue", base_color_red_green_blue,
     "Base colour", ["red", "green", "blue"]),
    ("02_metallic_0_05_1", metallic_0_05_1,
     "Metallic (gold, roughness 0.2)", ["metallic 0", "metallic 0.5", "metallic 1"]),
    ("03_roughness_0_05_1", roughness_0_05_1,
     "Roughness (gold, metallic 1)", ["roughness 0", "roughness 0.5", "roughness 1"]),
    ("04_base_color_texture_quadrants", base_color_texture_quadrants,
     "Base colour texture: UV orientation", ["red green / blue yellow"]),
    ("05_base_color_texture_on_sphere_plain_tinted", base_color_texture_on_sphere_plain_tinted,
     "Base colour texture on a sphere", ["texture as is", "texture x orange tint"]),
    ("06a_normal_map_gl_off_on_tangent_w_flip", normal_map_gl_off_on_tangent_w_flip,
     "Normal map, OpenGL convention", ["no normal map", "normal map: convex bumps", "tangent w = -1: flipped up / down"]),
    ("06b_normal_map_dx_off_on_tangent_w_flip", normal_map_dx_off_on_tangent_w_flip,
     "Normal map, DirectX convention (read as OpenGL)",
     ["no normal map", "normal map: flipped up / down", "tangent w = -1: convex bumps"]),
    ("06d_normal_map_gl_1024_plane_vs_height_reference", normal_map_gl_1024_plane_vs_height_reference,
     "Normal map 1024 x 1024 on a plane", ["normal map only", "height reference (light = high)"]),
    ("07_emissive_none_orange_orange_x4_texture", emissive_none_orange_orange_x4_texture,
     "Emission (black base)", ["no emission", "orange", "orange x 4", "stripes texture x orange"]),
    ("08_opacity_1_03_alpha_checker", opacity_1_03_alpha_checker,
     "Opacity", ["opaque", "opacity 0.3", "alpha texture 0.75 / 0.25\ndouble-sided"]),
    ("09_double_sided_on_off", double_sided_on_off,
     "Double-sided (quads seen from the back)", ["double-sided: visible", "single-sided: culled"]),
    ("24_transmission_0_05_1", transmission_0_05_1,
     "Transmission (solid, IOR 1.5)", ["transmission 0", "transmission 0.5", "transmission 1"]),
    ("25_glass_clear_rough_tinted_thin", glass_clear_rough_tinted_thin,
     "Glass (IOR 1.5)", ["clear, solid", "roughness 0.3", "blue tint", "thin-walled"]),
    ("26_glass_ior_1_133_15_24", glass_ior_1_133_15_24,
     "Glass IOR (solid)", ["IOR 1.0", "IOR 1.33", "IOR 1.5", "IOR 2.4"]),
    ("27a_refraction_mode_quad", refraction_mode_quad,
     "Refraction mode: quad turned 45 deg (thickness 0.3)", REFRACTION_MODE_CAPTIONS_QUAD),
    ("27b_refraction_mode_box", refraction_mode_box,
     "Refraction mode: box 0.3 deep turned 45 deg (thickness 0.3)", REFRACTION_MODE_CAPTIONS_BOX),
    ("27c_refraction_mode_sphere", refraction_mode_sphere,
     "Refraction mode: sphere (thickness 1.0)", REFRACTION_MODE_CAPTIONS_SPHERE),
]


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def column_x(i, n):
    return (i - (n - 1) / 2) * SPACING


def gltf_row_path(name):
    return os.path.join(GEN_DIR, name + "_gltf.glb")


def geometry_path(geo):
    return os.path.join(GEN_DIR, "geometry_%s.glb" % geo)


def backdrop_path():
    return os.path.join(GEN_DIR, "backdrop.glb")


def save(g, path):
    g.pngs.clear()  # the textures are already in textures/, under their own name
    g.save(path)


def write_gltf_row(name, columns):
    """Top row: one .glb holding every object of the test with its glTF material."""
    g = mk.Gltf(name)
    for i, c in enumerate(columns):
        node = {"translation": [column_x(i, len(columns)), 0, 0]}
        if c["turn"]:
            node["rotation"] = mk.rot_y(c["turn"])
        geo = GEOMETRY[c["geo"]]()
        g.node(g.mesh(geo, c["gltf"](g), tangents="tan" in geo), **node)   # the box has no tangents
    save(g, gltf_row_path(name))


def write_geometry(geo):
    """Bottom row: the bare mesh, no material at all (the Genesis surface provides it)."""
    g = mk.Gltf(geo)
    mesh = GEOMETRY[geo]()
    g.node(g.mesh(mesh, tangents="tan" in mesh))
    save(g, geometry_path(geo))


def write_backdrop(size=18.0, stripe=0.3):
    """Matte wall of vertical grey stripes. Vertical, so the background is the same behind both
    rows whatever the height; stripes, so see-through and refracting objects have something to show."""
    g = mk.Gltf("backdrop")
    stripes = mk.png(64, 2, lambda x, y: (165, 165, 165) if x < 32 else (95, 95, 95))
    tex = g.texture(stripes, "stripes", sampler={"wrapS": 10497, "wrapT": 10497})
    g.node(g.mesh(mk.quad(size, uv_scale=size / (2 * stripe)),
                  g.pbr((1, 1, 1, 1), 0.0, 1.0, baseColorTexture=tex)))
    save(g, backdrop_path())


def write_files(name, columns):
    os.makedirs(GEN_DIR, exist_ok=True)
    write_backdrop()
    write_gltf_row(name, columns)
    for geo in {c["geo"] for c in columns}:
        write_geometry(geo)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)   # scalable since Pillow 10.1
    except TypeError:
        return ImageFont.load_default()            # older Pillow: small fixed-size font


def write(draw, xy, text, size, anchor):
    """Text on a black box. anchor: "l" (xy = top-left corner) or "m" (xy = centre)."""
    kwargs = dict(font=font(size), align="center", spacing=2)
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, **kwargs)
    x, y = xy
    if anchor == "m":
        x, y = x - (left + right) / 2, y - (top + bottom) / 2
    draw.rectangle((x + left - 5, y + top - 3, x + right + 5, y + bottom + 3), fill=(0, 0, 0))
    draw.multiline_text((x, y), text, fill=(255, 255, 255), **kwargs)


def label(image, title, captions):
    """Annotate a render. This is 2D text drawn on the saved image with Pillow, nothing is added
    to the 3D scene:
    - the title of the test, top centre;
    - which row is which, on the left of each row;
    - the caption of each column, in the gap between the two rows, under / above its objects."""
    import math
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    w, h = image.size
    # Pixels per scene unit at the depth of the objects, to find the columns on the image.
    scale = (h / 2) / math.tan(math.radians(CAM_FOV) / 2) / abs(CAM_POS[1])
    write(draw, (w / 2, 22), title, 22, "m")
    write(draw, (10, h / 2 - (ROW_Z + 0.5) * scale - 40), "glTF material", 15, "l")
    write(draw, (10, h / 2 + (ROW_Z + 0.5) * scale + 22), "Genesis surface", 15, "l")
    for i, caption in enumerate(captions):
        write(draw, (w / 2 + column_x(i, len(captions)) * scale, h / 2), caption, 14, "m")


def render(name, columns, title, captions):
    import genesis as gs
    import gs_nyx.nyx_py_renderer as npr
    import gs_nyx.nyx_py_sdk as nps
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from PIL import Image

    gs.init()
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
    scene.add_entity(morph=gs.morphs.Mesh(file=backdrop_path(), pos=(0.0, BACKDROP_Y, 0.0),
                                          fixed=True, collision=False))

    # Top row: glTF materials, the whole row is one file.
    scene.add_entity(morph=gs.morphs.Mesh(file=gltf_row_path(name), pos=(0.0, 0.0, ROW_Z),
                                          fixed=True, collision=False))

    # Bottom row: the same meshes without material, one entity per object, with a Genesis surface.
    for i, c in enumerate(columns):
        scene.add_entity(
            morph=gs.morphs.Mesh(file=geometry_path(c["geo"]), pos=(column_x(i, len(columns)), 0.0, -ROW_Z),
                                 euler=(0.0, 0.0, c["turn"]), fixed=True, collision=False),
            surface=c["surface"](gs))

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
        # Number the modes so a file browser lists them in quality order.
        out = os.path.join(OUT_DIR, "%s_%d_%s.png" % (name, i + 1, mode))
        image = Image.fromarray(cam.read().rgb[0].cpu().numpy())
        if LABELS:
            label(image, title, captions)
        image.save(out)
        print("Saved", out)


def main():
    tests = {name: (make, title, captions) for name, make, title, captions in TESTS}
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":  # internal: render one test in this process
        make, title, captions = tests[sys.argv[2]]
        render(sys.argv[2], make(), title, captions)
        return
    filters = sys.argv[1:]
    names = [n for n in tests if not filters or any(p in n for p in filters)]
    if not names:
        sys.exit("No test matches %s. Available:\n  %s" % (filters, "\n  ".join(tests)))
    for n in names:
        make, _, captions = tests[n]
        columns = make()
        if len(captions) != len(columns):
            sys.exit("%s: %d captions for %d columns, fix TESTS." % (n, len(captions), len(columns)))
        print("==>", n)
        write_files(n, columns)
        if subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", n]).returncode != 0:
            print("    FAILED:", n)


if __name__ == "__main__":
    main()
