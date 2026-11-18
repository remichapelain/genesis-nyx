"""Generate glTF test files for transparency: alpha modes, transmission, IOR, volume, stacking.

Same idea as examples/gltf_tests (whose glTF builder is reused): each file tests one thing,
objects sit in a row along X facing +Z (glTF is Y-up). Every file also gets a patterned BACKDROP
quad behind the objects, because transparency can only be judged against something:
  - light / dark checker, 0.1 unit cells -> straight lines show bending, cell edges show blur;
  - dark cells are RED in the top third, GREY in the middle, BLUE in the bottom third -> a
    refracting solid flips the image, so red and blue swap places when seen through it.

Usage:
    python examples/gltf_transparency_tests/make_gltf_transparency_tests.py   # -> generated/*.glb

How glTF defines transparency (the tests are built around these rules):
  ALPHA (core spec) is COVERAGE: how much of the surface exists.
    alpha = baseColorFactor.a x baseColorTexture.a x COLOR_0.a
    OPAQUE ignores it, MASK keeps the surface where alpha >= alphaCutoff (default 0.5),
    BLEND composites it. At alpha 0 the surface is gone, highlights and reflections included.
  TRANSMISSION (KHR_materials_transmission) is light going THROUGH a surface that is fully there.
    transmissionFactor (x transmissionTexture.R) replaces the diffuse lobe by a transmission lobe.
    The specular reflection stays (highlights at transmission 1), transmitted light is tinted by
    the base colour, roughness blurs it, and metals do not transmit: transmission x (1 - metallic).
  IOR (KHR_materials_ior, default 1.5) sets the reflectance F0 = ((ior - 1) / (ior + 1))^2.
    Whether it also BENDS light depends on the volume:
  VOLUME (KHR_materials_volume) picks the refraction mode.
    thicknessFactor 0 or no extension: THIN-WALLED. Light exits where it entered, so no bending;
      the IOR only changes how strong the reflections are. No absorption.
    thicknessFactor > 0: VOLUMETRIC. The mesh is the boundary of a solid, the IOR bends light,
      and it is absorbed along the way: T = attenuationColor ^ (distance / attenuationDistance).
      thicknessFactor (x thicknessTexture.G) is in mesh units and scales with the node; it is a
      hint for rasterizers, a ray tracer uses the real geometry. attenuationDistance defaults to
      infinity (no absorption).
"""

import math
import os
import sys

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "generated")
TEX_DIR = os.path.join(OUT_DIR, "textures")

# The glTF builder, PNG encoder and geometry come from the basic glTF tests.
sys.path.insert(0, os.path.join(HERE, "..", "gltf_tests"))
import make_gltf_tests as mk  # noqa: E402

mk.TEX_DIR = TEX_DIR  # Gltf.save() writes texture copies there

# Linear colours (base colour FACTORS are linear, base colour TEXTURES are sRGB).
ORANGE      = (1.0, 0.5, 0.0)
ORANGE_SRGB = (255, 188, 0)       # the same orange as an sRGB texel
RED         = (0.9, 0.05, 0.05)
BLUE        = (0.05, 0.1, 0.9)
ABSORB      = (1.0, 0.5, 0.25)    # attenuationColor used by the absorption tests


# ---------------------------------------------------------------- textures

def tex_backdrop(w=512, h=128, cell=8):
    """Checker; dark cells red in the top third, grey in the middle, blue in the bottom third."""
    def f(x, y):
        if (x // cell + y // cell) % 2:
            return (225, 225, 225)
        return (190, 40, 40) if y < h // 3 else (40, 40, 40) if y < 2 * h // 3 else (40, 70, 190)
    return mk.png(w, h, f)


def tex_solid_rgba(rgba):
    return mk.png(1, 1, lambda x, y: rgba, rgba=True)


def tex_alpha_gradient_bands(n=64):
    """Alpha goes 0 (left) -> 1 (right). Colour: 4 vertical bands orange / blue / orange / blue,
    so the band borders mark alpha 0.25, 0.5 and 0.75."""
    return mk.png(n, n, lambda x, y: ((255, 188, 0) if (x * 4 // n) % 2 == 0 else (0, 120, 255))
                  + (round(255 * x / (n - 1)),), rgba=True)


def tex_transmission_stripes(n=64, stripes=8):
    """Horizontal stripes, R = 255 / 0 (transmission is read from RED). G and B stay 0."""
    return mk.png(n, n, lambda x, y: (255 if (y * stripes // n) % 2 == 0 else 0, 0, 0))


def tex_thickness_gradient(n=64):
    """G goes 0 (left) -> 1 (right) (thickness is read from GREEN). R and B stay 255."""
    return mk.png(n, n, lambda x, y: (255, round(255 * x / (n - 1)), 255))


# ---------------------------------------------------------------- helpers

def rect(w, h):
    """Quad of w x h facing +Z, texture upright."""
    a, b = w / 2, h / 2
    return dict(pos=[(-a, -b, 0), (a, -b, 0), (a, b, 0), (-a, b, 0)], nrm=[(0, 0, 1)] * 4,
                uv=[(0, 1), (1, 1), (1, 0), (0, 0)], tan=[(1, 0, 0, 1)] * 4, idx=[0, 1, 2, 0, 2, 3])


def use_ext(g, name):
    used = g.d.setdefault("extensionsUsed", [])
    if name not in used:
        used.append(name)


def glass(g, color=(1, 1, 1), rough=0.0, metal=0.0, transmission=1.0, ior=None, thickness=None,
          att_color=None, att_dist=None, transmission_tex=None, thickness_tex=None, alpha=1.0, **extra):
    """Material with KHR_materials_transmission, plus _ior / _volume when their values are given.
    No thickness = thin-walled. `extra` goes to the material (alphaMode, doubleSided, ...)."""
    trans = {"transmissionFactor": transmission}
    if transmission_tex is not None:
        trans["transmissionTexture"] = transmission_tex
    exts = {"KHR_materials_transmission": trans}
    if ior is not None:
        exts["KHR_materials_ior"] = {"ior": ior}
    vol = {}
    if thickness is not None:
        vol["thicknessFactor"] = thickness
    if thickness_tex is not None:
        vol["thicknessTexture"] = thickness_tex
    if att_color is not None:
        vol["attenuationColor"] = list(att_color)
    if att_dist is not None:
        vol["attenuationDistance"] = att_dist
    if vol:
        exts["KHR_materials_volume"] = vol
    for name in exts:
        use_ext(g, name)
    return g.pbr(tuple(color) + (alpha,), metal, rough, extensions=exts, **extra)


def blend(g, rgb, alpha, rough=0.5, **extra):
    return g.pbr(tuple(rgb) + (alpha,), 0.0, rough, alphaMode="BLEND", **extra)


def leaf(g):
    """Double-sided MASK material: orange checker whose alpha is 0.75 / 0.25 -> half the cells cut out."""
    return g.pbr((1, 1, 1, 1), 0.0, 0.5, alphaMode="MASK", doubleSided=True,
                 baseColorTexture=g.texture(mk.tex_alpha_checker(), "alpha_checker"))


def put(g, mesh, x=0.0, y=0.0, z=0.0, **trs):
    g.node(mesh, translation=[x, y, z], **trs)


def add_backdrop(g):
    """6.4 x 1.6 quad behind the objects, from the ground (y = -0.6) up."""
    tex = g.texture(tex_backdrop(), "backdrop", sampler={"magFilter": 9728, "minFilter": 9987})
    put(g, g.mesh(rect(6.4, 1.6), g.pbr((1, 1, 1, 1), 0.0, 1.0, baseColorTexture=tex)), 0, 0.2, -1.0)


def vertex_alpha(g, mesh, alpha):
    """Add a COLOR_0 attribute (white, given alpha) to a mesh."""
    prim = g.d["meshes"][mesh]["primitives"][0]
    count = g.d["accessors"][prim["attributes"]["POSITION"]]["count"]
    prim["attributes"]["COLOR_0"] = g._accessor([(1.0, 1.0, 1.0, alpha)] * count, "VEC4")
    return mesh


# Rows of 4 use spheres of radius 0.5 spaced 1.2 (like the basic tests); rows of 5 use these.
R5, SPACING5 = 0.4, 0.95


# ---------------------------------------------------------------- the tests
#
# Test index (numbers come from the order of the TESTS list at the bottom of the file).
#
#   Alpha (coverage)
#   01   Alpha BLEND ramp
#   02   Alpha sources: factor / texture / factor x texture / vertex colour
#   03   Alpha MASK: cutoff values and edge cases
#   Transmission
#   04   Transmission ramp vs alpha (transmission is not alpha)
#   05   Transmission: base colour tint and metallic
#   06a  Transmission roughness, thin-walled
#   06b  Transmission roughness, volume
#   06c  Transmission roughness in 0.1 steps, thin-walled
#   06d  Transmission roughness in 0.1 steps, volume
#   IOR and refraction modes
#   07a  IOR, thin-walled (no bending, reflection strength only)
#   07b  IOR, volume (bending)
#   08a  Refraction modes: thin / closed volume / double-sided slab / real box / node scale
#   08b  Sidedness x thickness: doubleSided selects the slab (Nyx rule)
#   09   Absorption vs thickness (Beer's law, measurable)
#   10   Absorption edge cases
#   Interactions
#   11   Alpha mode x transmission
#   12   Transmission and thickness TEXTURES
#   Stacking
#   13   Blend on blend
#   14   Glass on glass
#   15a  Blend and glass
#   15b  Mask and glass
#   16   Opaque / emissive content inside and behind glass
#   17   Everything together

# ================================================================================================
# 01  | Alpha BLEND ramp
# ================================================================================================
def alpha_blend_0_025_05_075_1(g):
    """Orange spheres, alphaMode BLEND, alpha 0 / 0.25 / 0.5 / 0.75 / 1.
    - alpha 0: completely INVISIBLE, no highlight, no reflection, no shadow (alpha is coverage).
    - in between: the backdrop shows through more or less, with NO bending and NO blur.
    - alpha 1: must look IDENTICAL to a plain opaque sphere."""
    g.row([g.mesh(mk.sphere(R5), blend(g, ORANGE, a, rough=0.2)) for a in (0.0, 0.25, 0.5, 0.75, 1.0)],
          spacing=SPACING5)


# ================================================================================================
# 02  | Alpha sources: factor / texture / factor x texture / vertex colour
# ================================================================================================
def alpha_blend_factor_texture_product_vertex(g):
    """Four BLEND spheres that must ALL LOOK IDENTICAL (orange, alpha 0.5):
    1. baseColorFactor alpha 0.5
    2. baseColorTexture alpha 0.5, factor alpha 1
    3. texture alpha 0.75 x factor alpha 0.667: the two must MULTIPLY
    4. factor alpha 1, vertex colour COLOR_0 alpha 0.5 (opaque sphere => COLOR_0 is ignored)"""
    half = g.texture(tex_solid_rgba(ORANGE_SRGB + (128,)), "orange_alpha_050")
    three_q = g.texture(tex_solid_rgba(ORANGE_SRGB + (191,)), "orange_alpha_075")
    g.row([g.mesh(mk.sphere(), blend(g, ORANGE, 0.5)),
           g.mesh(mk.sphere(), blend(g, (1, 1, 1), 1.0, baseColorTexture=half)),
           g.mesh(mk.sphere(), blend(g, (1, 1, 1), 0.667, baseColorTexture=three_q)),
           vertex_alpha(g, g.mesh(mk.sphere(), blend(g, ORANGE, 1.0)), 0.5)])


# ================================================================================================
# 03  | Alpha MASK: cutoff values and edge cases
# ================================================================================================
def alpha_mask_cutoff_025_default_075_below_equal(g):
    """Quads 1-3 share a texture whose alpha goes 0 (left) -> 1 (right), in 4 colour bands
    (orange / blue / orange / blue) whose borders mark alpha 0.25 / 0.5 / 0.75:
    1. alphaCutoff 0.25: the left band is cut, 3 bands remain.
    2. NO alphaCutoff (default 0.5): exactly the right half remains (2 bands).
    3. alphaCutoff 0.75: only the right band remains.
    What remains is fully SOLID with a hard edge, never semi-transparent.
    4. no texture, factor alpha 0.3 < cutoff 0.5: completely INVISIBLE.
    5. no texture, factor alpha 0.5 = cutoff 0.5: fully VISIBLE (the test is alpha >= cutoff)."""
    # Clamped, so the alpha 1 edge does not bleed into the alpha 0 edge through the filtering.
    grad = g.texture(tex_alpha_gradient_bands(), "alpha_gradient_bands", sampler={"wrapS": 33071, "wrapT": 33071})

    def mask(color=(1, 1, 1, 1), **extra):
        return g.pbr(color, 0.0, 0.5, alphaMode="MASK", **extra)
    g.row([g.mesh(mk.quad(0.8), mask(baseColorTexture=grad, alphaCutoff=0.25)),
           g.mesh(mk.quad(0.8), mask(baseColorTexture=grad)),
           g.mesh(mk.quad(0.8), mask(baseColorTexture=grad, alphaCutoff=0.75)),
           g.mesh(mk.quad(0.8), mask(ORANGE + (0.3,))),
           g.mesh(mk.quad(0.8), mask(ORANGE + (0.5,), alphaCutoff=0.5))], spacing=SPACING5)


# ================================================================================================
# 04  | Transmission ramp vs alpha (transmission is not alpha)
# ================================================================================================
def transmission_0_05_1_vs_alpha_0(g):
    """White smooth spheres, thin-walled (no volume). Left to right:
    1. transmissionFactor 0: must look IDENTICAL to a plain opaque white sphere.
    2. transmissionFactor 0.5: half white diffuse, half see-through.
    3. transmissionFactor 1: clear, but the sphere is STILL THERE: highlight and reflections of
       the environment stay, strongest on the rim (Fresnel).
    4. alphaMode BLEND, alpha 0 (no transmission): NOTHING there at all.
    3 and 4 looking the same = transmission was converted to alpha."""
    g.row([g.mesh(mk.sphere(), glass(g, transmission=t)) for t in (0.0, 0.5, 1.0)] +
          [g.mesh(mk.sphere(), blend(g, (1, 1, 1), 0.0, rough=0.0))])


# ================================================================================================
# 05  | Transmission: base colour tint and metallic
# ================================================================================================
def transmission_tint_and_metallic_0_05_1(g):
    """transmissionFactor 1 everywhere, thin-walled. Left to right:
    1. white base colour: clear glass.
    2. orange base colour: the backdrop is TINTED orange (transmitted light x base colour),
       the reflections stay white (dielectric).
    3. same, metallic 0.5: half as transparent, reflections half tinted.
    4. same, metallic 1: fully OPAQUE metal (transmission applies to the dielectric part only).
    5. reference: metallic 1 without any transmission. Must look IDENTICAL to 4."""
    g.row([g.mesh(mk.sphere(R5), glass(g)),
           g.mesh(mk.sphere(R5), glass(g, ORANGE)),
           g.mesh(mk.sphere(R5), glass(g, ORANGE, metal=0.5)),
           g.mesh(mk.sphere(R5), glass(g, ORANGE, metal=1.0)),
           g.mesh(mk.sphere(R5), g.pbr(ORANGE + (1,), 1.0, 0.0))], spacing=SPACING5)


# ================================================================================================
# 06a | Transmission roughness, thin-walled
# ================================================================================================
def transmission_roughness_0_025_05_1_thin(g):
    """Clear thin-walled glass, roughness 0 / 0.25 / 0.5 / 1. The backdrop seen THROUGH the sphere
    gets more and more blurred (frosted glass) but is not bent or flipped; the reflections ON the
    sphere blur the same way as on an opaque sphere of the same roughness."""
    g.row([g.mesh(mk.sphere(), glass(g, rough=r)) for r in (0.0, 0.25, 0.5, 1.0)])


# ================================================================================================
# 06b | Transmission roughness, volume
# ================================================================================================
def transmission_roughness_0_025_05_1_volume(g):
    """Same as 06a with a volume (thicknessFactor = the sphere diameter): the backdrop is now
    flipped by the solid sphere AND blurred. Roughness 1 must be fully frosted (no pattern)."""
    g.row([g.mesh(mk.sphere(), glass(g, rough=r, thickness=1.0)) for r in (0.0, 0.25, 0.5, 1.0)])


def roughness_steps(g, thickness):
    """Two rows of small spheres: roughness 0.0 -> 0.5 on top, 0.6 -> 1.0 below, left to right."""
    for i in range(11):
        row, col, count = (0, i, 6) if i < 6 else (1, i - 6, 5)
        mat = glass(g, rough=i / 10, thickness=thickness)
        put(g, g.mesh(mk.sphere(0.3), mat), (col - (count - 1) / 2) * 0.8, 0.45 - row * 0.7)


# ================================================================================================
# 06c | Transmission roughness in 0.1 steps, thin-walled
# ================================================================================================
def transmission_roughness_step_01_thin(g):
    """Clear thin-walled glass, roughness in steps of 0.1.
    Top row: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5. Bottom row: 0.6, 0.7, 0.8, 0.9, 1.0.
    The blur of the backdrop must grow SMOOTHLY from one sphere to the next: no jump, no step
    where two neighbours look the same, and it must not go back. 0.0 is perfectly sharp."""
    roughness_steps(g, None)


# ================================================================================================
# 06d | Transmission roughness in 0.1 steps, volume
# ================================================================================================
def transmission_roughness_step_01_volume(g):
    """Same as 06c on solid spheres (thicknessFactor = the diameter): flipped backdrop, blurred
    more and more, same layout. The blur must grow smoothly here too."""
    roughness_steps(g, 0.6)


IORS = (None, 1.0, 1.33, 1.5, 2.4)  # None = no KHR_materials_ior at all


# ================================================================================================
# 07a | IOR, thin-walled (no bending, reflection strength only)
# ================================================================================================
def ior_none_1_133_15_24_thin(g):
    """Clear thin-walled glass. IOR: extension absent / 1.0 / 1.33 (water) / 1.5 / 2.4 (diamond).
    - NO bending on any sphere: the backdrop lines stay straight through all of them.
    - only the reflections change: none at 1.0 (F0 = 0, the sphere is INVISIBLE), then
      F0 = 2 % at 1.33, 4 % at 1.5, 17 % at 2.4 (clearly mirror-like).
    - sphere 1 (no extension) must look IDENTICAL to sphere 4 (the default IOR is 1.5)."""
    g.row([g.mesh(mk.sphere(R5), glass(g, ior=i)) for i in IORS], spacing=SPACING5)


# ================================================================================================
# 07b | IOR, volume (bending)
# ================================================================================================
def ior_none_1_133_15_24_volume(g):
    """Same IORs as 07a on SOLID spheres (thicknessFactor = the diameter).
    - IOR 1.0: still invisible (no reflection, no bending).
    - above 1 the sphere acts as a lens: the backdrop appears upside down (red at the bottom,
      blue at the top) and mirrored, more and more compressed toward the centre as the IOR grows.
    - sphere 1 (no extension) must look IDENTICAL to sphere 4."""
    g.row([g.mesh(mk.sphere(R5), glass(g, ior=i, thickness=2 * R5)) for i in IORS], spacing=SPACING5)


# ================================================================================================
# 08a | Refraction modes: thin / closed volume / double-sided slab / real box / node scale
# ================================================================================================
def refraction_modes_thin_volume_slab_box_scaled(g):
    """Clear glass, IOR 1.5. Left to right:
    1. sphere, no volume: THIN-WALLED, backdrop straight through.
    2. sphere, thicknessFactor 0.8 (its diameter), single-sided: closed VOLUME, backdrop flipped.
    3. flat QUAD, doubleSided, thicknessFactor 0.2, turned 30 deg: a single sheet declared as a
       0.2 thick SLAB. The backdrop is shifted sideways (parallel offset), not flipped.
    4. real BOX, 0.2 thick, single-sided, thicknessFactor 0.2, turned 30 deg, its front face where
       quad 3 is. Its backdrop shift must match 3: that is what the slab approximates.
    5. sphere MESH of radius 0.2 with thicknessFactor 0.4 under a node scale of 2: thickness is in
       mesh units and scales with the node, so it must look IDENTICAL to sphere 2."""
    tilt = mk.rot_y(30)
    vol = glass(g, thickness=2 * R5)
    meshes = [g.mesh(mk.sphere(R5), glass(g)),
              g.mesh(mk.sphere(R5), vol),
              g.mesh(mk.quad(0.75), glass(g, thickness=0.2, doubleSided=True)),
              g.mesh(mk.box(0.75, 0.75, 0.2, center=(0, 0, -0.1)), glass(g, thickness=0.2)),
              g.mesh(mk.sphere(R5 / 2), glass(g, thickness=R5))]
    extra = [{}, {}, {"rotation": tilt}, {"rotation": tilt}, {"scale": [2, 2, 2]}]
    for i, (m, trs) in enumerate(zip(meshes, extra)):
        put(g, m, (i - 2) * SPACING5, **trs)


# ================================================================================================
# 08b | Sidedness x thickness: doubleSided selects the slab (Nyx rule)
# ================================================================================================
def refraction_sidedness_sphere_single_double_quad_double_single(g):
    """Clear glass, IOR 1.5, every object has a thicknessFactor. Nyx picks the refraction mode
    from the sidedness: single-sided = closed volume (the exit face is found on the mesh),
    doubleSided = slab (the exit is computed from the thickness value, no mesh lookup). This is a
    Nyx rule: glTF itself does not tie the two, so other viewers may differ on 2 and 4.
    Left to right:
    1. solid sphere, single-sided, thicknessFactor = diameter: closed volume, backdrop FLIPPED
       (lens).
    2. the same sphere, doubleSided: treated as a SLAB at every point, so NOT the lens image of 1
       but a magnified, distorted backdrop. A solid meant to act as a lens must be single-sided.
    3. flat quad, doubleSided, thicknessFactor 0.2, turned 30 deg: slab, backdrop SHIFTED sideways.
    4. the same quad, single-sided: asks for a closed volume but has no exit face, so it falls
       back to THIN-WALLED: backdrop straight through, no shift. An open sheet meant to act as a
       slab must be doubleSided."""
    tilt = mk.rot_y(30)
    d = 1.0
    put(g, g.mesh(mk.sphere(), glass(g, thickness=d)), -1.8)
    put(g, g.mesh(mk.sphere(), glass(g, thickness=d, doubleSided=True)), -0.6)
    put(g, g.mesh(mk.quad(0.9), glass(g, thickness=0.2, doubleSided=True)), 0.6, rotation=tilt)
    put(g, g.mesh(mk.quad(0.9), glass(g, thickness=0.2)), 1.8, rotation=tilt)


# ================================================================================================
# 09  | Absorption vs thickness (Beer's law, measurable)
# ================================================================================================
def absorption_depth_025_05_1_ior1_and_ior15(g):
    """Boxes seen through their depth, all with attenuationColor (1, 0.5, 0.25) and
    attenuationDistance 0.5, thicknessFactor = the box depth. Boxes 1-3 have IOR 1.0 (no
    reflection, no bending), so a pixel is exactly backdrop x attenuationColor ^ (depth / 0.5):
    1. depth 0.25 -> x (1, 0.71, 0.5)    light tint
    2. depth 0.5  -> x (1, 0.5, 0.25)    exactly the attenuation colour (on a white cell)
    3. depth 1.0  -> x (1, 0.25, 0.06)   deep orange
    4. depth 0.5 with IOR 1.5: same tint as 2 plus reflections and a shifted backdrop.
    Same tint on 1-3 = the thickness is ignored. No tint = the volume is ignored.
    (The camera looks slightly down, so the real path is ~1.5 % longer than the depth.)"""
    def mat(depth, ior):
        return glass(g, ior=ior, thickness=depth, att_color=ABSORB, att_dist=0.5)
    g.row([g.mesh(mk.box(0.7, 0.8, d), mat(d, ior)) for d, ior in ((0.25, 1.0), (0.5, 1.0), (1.0, 1.0), (0.5, 1.5))])


# ================================================================================================
# 10  | Absorption edge cases
# ================================================================================================
def absorption_reference_white_nodistance_thin_basecolor(g):
    """Solid glass spheres (IOR 1.5, thicknessFactor = the diameter). Left to right:
    1. reference: attenuationColor orange, attenuationDistance 0.4 -> tinted, DARKER in the centre
       (long path) and lighter toward the rim (short path).
    2. attenuationColor white (1, 1, 1), same distance -> clear, no tint.
    3. attenuationColor orange but NO attenuationDistance (default = infinity) -> clear, no tint.
    4. same attenuation as 1 but thicknessFactor 0 (thin-walled) -> clear, no tint, no flip.
    5. no attenuation, orange BASE COLOUR instead -> tinted evenly, the same at centre and rim.
    1 and 5 looking the same = absorption is applied as a flat tint."""
    d = 2 * R5
    g.row([g.mesh(mk.sphere(R5), glass(g, thickness=d, att_color=ABSORB, att_dist=0.4)),
           g.mesh(mk.sphere(R5), glass(g, thickness=d, att_color=(1, 1, 1), att_dist=0.4)),
           g.mesh(mk.sphere(R5), glass(g, thickness=d, att_color=ABSORB)),
           g.mesh(mk.sphere(R5), glass(g, thickness=0.0, att_color=ABSORB, att_dist=0.4)),
           g.mesh(mk.sphere(R5), glass(g, ABSORB, thickness=d))], spacing=SPACING5)


# ================================================================================================
# 11  | Alpha mode x transmission
# ================================================================================================
def transmission_with_alpha_opaque_blend_mask(g):
    """transmissionFactor 1 (thin-walled) combined with each alpha mode. Left to right:
    1. reference: plain clear glass (alpha 1, OPAQUE).
    2. OPAQUE with alpha 0.3: alpha is ignored, must look IDENTICAL to 1.
    3. BLEND with alpha 0.5: glass at half coverage -> reflections and highlight at HALF the
       strength of 1. Same as 1 = the blend was dropped. Opaque-looking = transmission dropped.
    4. MASK with the alpha checker (0.75 / 0.25), double-sided: glass with HOLES. Kept cells are
       clear glass with reflections, cut cells have none, and the far side shows through them."""
    checker = g.texture(mk.tex_alpha_checker(), "alpha_checker")
    g.row([g.mesh(mk.sphere(), glass(g)),
           g.mesh(mk.sphere(), glass(g, alpha=0.3, alphaMode="OPAQUE")),
           g.mesh(mk.sphere(), glass(g, alpha=0.5, alphaMode="BLEND")),
           g.mesh(mk.sphere(), glass(g, alphaMode="MASK", doubleSided=True, baseColorTexture=checker))])


# ================================================================================================
# 12  | Transmission and thickness TEXTURES
# ================================================================================================
def transmission_texture_and_thickness_texture(g):
    """Left to right:
    1. sphere, transmissionFactor 1 x transmissionTexture of stripes (R = 1 / 0): clear glass
       stripes alternating with opaque white stripes. All clear = the texture is not read.
    2. sphere, transmissionTexture with R = 0 but G = B = 1: fully OPAQUE white (only RED counts).
    3. double-sided slab quad, thicknessFactor 0.5 x thicknessTexture whose G goes 0 (left) -> 1
       (right), orange attenuation (distance 0.25): clear on the left, deep orange on the right.
       Even tint = the texture is not read.
    4. same quad, thicknessTexture with G = 0 but R = B = 1: thickness 0 -> clear, no tint (only
       GREEN counts)."""
    stripes = g.texture(tex_transmission_stripes(), "transmission_stripes")
    r_zero = g.texture(mk.tex_solid((0, 255, 255)), "transmission_r0")
    grad = g.texture(tex_thickness_gradient(), "thickness_gradient")
    g_zero = g.texture(mk.tex_solid((255, 0, 255)), "thickness_g0")

    def slab(tex):
        return glass(g, thickness=0.5, thickness_tex=tex, att_color=ABSORB, att_dist=0.25, doubleSided=True)
    g.row([g.mesh(mk.sphere(), glass(g, transmission_tex=stripes)),
           g.mesh(mk.sphere(), glass(g, transmission_tex=r_zero)),
           g.mesh(mk.quad(0.9), slab(grad)),
           g.mesh(mk.quad(0.9), slab(g_zero))])


# ---- stacking -----------------------------------------------------------------------------
# Several transparent surfaces behind one another. Groups are 1.2 apart; inside a group the front
# object is shifted right / down and the back one left / up, so each is also seen on its own.

GROUPS4 = (-1.8, -0.6, 0.6, 1.8)


def pair(g, x, front, back, dz=0.25, front_first=True):
    """Two meshes overlapping in the middle of the group: `front` toward the camera, `back` behind."""
    nodes = [(front, x + 0.15, -0.08, dz), (back, x - 0.15, 0.08, -dz)]
    for m, px, py, pz in (nodes if front_first else nodes[::-1]):
        put(g, m, px, py, pz)


# ================================================================================================
# 13  | Blend on blend
# ================================================================================================
def stack_blend_front_first_last_crossing_double_sided(g):
    """BLEND surfaces, alpha 0.5, red and blue. Left to right:
    1. red quad IN FRONT of a blue quad; the red one comes first in the file.
    2. the same two quads, but the red one comes LAST in the file. Must look IDENTICAL to 1:
       where they overlap the colour leans RED (front). Leaning blue in one of the two = the
       blending follows the file / draw order instead of the depth.
    3. two quads CROSSING each other (an X seen from above): red is in front on one side of the
       crossing line and blue on the other, so the overlap colour must swap at that line.
    4. double-sided orange sphere: its far half is seen through its near half, so the sphere looks
       denser than the single-sided alpha 0.5 sphere of test 01, with a visible inner surface."""
    def q(rgb):
        return g.mesh(mk.quad(0.65), blend(g, rgb, 0.5, doubleSided=True))
    pair(g, GROUPS4[0], q(RED), q(BLUE), front_first=True)
    pair(g, GROUPS4[1], q(RED), q(BLUE), front_first=False)
    put(g, q(RED), GROUPS4[2], rotation=mk.rot_y(35))
    put(g, q(BLUE), GROUPS4[2], rotation=mk.rot_y(-35))
    put(g, g.mesh(mk.sphere(0.45), blend(g, ORANGE, 0.5, doubleSided=True)), GROUPS4[3])


# ================================================================================================
# 14  | Glass on glass
# ================================================================================================
def stack_glass_panes_filters_both_orders_nested(g):
    """Transmissive surfaces behind one another. Left to right:
    1. three clear thin panes in a staircase: from left to right you look through 1, 2, 3, 2, 1
       panes. Each extra pane adds a little reflection and nothing else; no pane may disappear
       or turn opaque behind another.
    2. RED filter pane (red base colour, transmission 1) in front of a BLUE one.
    3. BLUE filter in front of the RED one. In 2 and 3 the overlap is nearly BLACK (red x blue),
       and the SAME in both: filtering does not depend on the order.
    4. small solid glass sphere INSIDE a large solid glass sphere. glTF does not define nested
       volumes, so there is no single right answer: check that the inner sphere is drawn, and
       that nothing turns black or vanishes."""
    x = GROUPS4[0]
    for k in range(3):
        put(g, g.mesh(rect(0.5, 0.8), glass(g, doubleSided=True)), x + (k - 1) * 0.2, 0, 0.3 - k * 0.3)

    def filt(rgb):
        return g.mesh(mk.quad(0.65), glass(g, rgb, doubleSided=True))
    red, blue = (1, 0.15, 0.15), (0.15, 0.15, 1)
    pair(g, GROUPS4[1], filt(red), filt(blue))
    pair(g, GROUPS4[2], filt(blue), filt(red))
    put(g, g.mesh(mk.sphere(0.45), glass(g, thickness=0.9)), GROUPS4[3])
    put(g, g.mesh(mk.sphere(0.2), glass(g, thickness=0.4)), GROUPS4[3])


GROUPS3 = (-1.4, 0.0, 1.4)


# ================================================================================================
# 15a | Blend and glass
# ================================================================================================
def stack_blend_behind_glass_glass_behind_blend_rough_glass(g):
    """BLEND and transmissive surfaces mixed. Left to right:
    1. orange BLEND quad (alpha 0.5) BEHIND a solid glass sphere: the part of the quad seen
       through the sphere is flipped and distorted like the backdrop. Missing through the sphere
       = blended surfaces are not visible to the refraction.
    2. clear thin glass pane BEHIND an orange BLEND quad: the pane's reflections are visible
       through the quad, tinted by it; the quad is unchanged where the pane is behind it.
    3. BLEND quad with the 4-colour quadrant texture (alpha 0.6) BEHIND a ROUGH thin glass pane
       (roughness 0.4): through the pane the quad is BLURRED like the backdrop; next to the pane
       it is sharp."""
    quad = g.mesh(mk.quad(0.7), blend(g, ORANGE, 0.5, doubleSided=True))
    pair(g, GROUPS3[0], g.mesh(mk.sphere(0.4), glass(g, thickness=0.8)), quad, dz=0.4)
    pair(g, GROUPS3[1], quad, g.mesh(mk.quad(0.7), glass(g, doubleSided=True)))
    quadrants = blend(g, (1, 1, 1), 0.6, doubleSided=True, baseColorTexture=g.texture(mk.tex_quadrants(), "quadrants"))
    pair(g, GROUPS3[2], g.mesh(mk.quad(0.7), glass(g, rough=0.4, doubleSided=True)), g.mesh(mk.quad(0.7), quadrants))


# ================================================================================================
# 15b | Mask and glass
# ================================================================================================
def stack_mask_front_of_glass_behind_glass_behind_rough_glass(g):
    """A MASK leaf (orange alpha checker, half the cells cut out) and transmissive surfaces:
    1. leaf IN FRONT of a solid glass sphere: the sphere is visible through the holes, the solid
       cells hide it.
    2. leaf BEHIND a solid glass sphere: seen through the sphere, the leaf is flipped and
       distorted, and its holes are still holes (backdrop visible through them).
    3. leaf BEHIND a rough thin glass pane (roughness 0.4): blurred checker of orange and
       backdrop. Solid orange = the cutout is lost behind glass."""
    def leaf_quad():
        return g.mesh(mk.quad(0.7), leaf(g))
    pair(g, GROUPS3[0], leaf_quad(), g.mesh(mk.sphere(0.4), glass(g, thickness=0.8)), dz=0.4)
    pair(g, GROUPS3[1], g.mesh(mk.sphere(0.4), glass(g, thickness=0.8)), leaf_quad(), dz=0.4)
    pair(g, GROUPS3[2], g.mesh(mk.quad(0.7), glass(g, rough=0.4, doubleSided=True)), leaf_quad())


# ================================================================================================
# 16  | Opaque / emissive content inside and behind glass
# ================================================================================================
def stack_opaque_and_emissive_inside_and_behind_glass(g):
    """Solid clear glass spheres (radius 0.45) with something inside or behind. Left to right:
    1. small opaque RED sphere at the centre: visible, magnified by the glass around it.
    2. small EMISSIVE orange sphere (strength 4) at the centre: visible, magnified, as bright as
       it would be outside, and it lights the glass from within in the path tracers.
    3. opaque red box BEHIND the sphere (half hidden by it): through the sphere the box is
       flipped to the other side.
    4. emissive orange box BEHIND the sphere: same, and still bright through the glass."""
    use_ext(g, "KHR_materials_emissive_strength")
    red = g.pbr(RED + (1,), 0.0, 0.5)
    glow = g.pbr((0, 0, 0, 1), emissiveFactor=[1, 0.4, 0],
                 extensions={"KHR_materials_emissive_strength": {"emissiveStrength": 4.0}})
    for x, mat, inside in zip(GROUPS4, (red, glow, red, glow), (True, True, False, False)):
        put(g, g.mesh(mk.sphere(0.45), glass(g, thickness=0.9)), x)
        if inside:
            put(g, g.mesh(mk.sphere(0.15), mat), x)
        else:
            put(g, g.mesh(mk.box(0.35, 0.35, 0.35), mat), x - 0.3, 0.1, -0.65)


# ================================================================================================
# 17  | Everything together
# ================================================================================================
def combined_all_transparency(g):
    """Every kind of transparency in one scene. Left to right:
    LEFT: a frosted thin pane (roughness 0.35) in front of a MASK leaf -> blurred leaf with holes.
    CENTRE: a clear solid glass sphere (flipped backdrop) next to a box of amber liquid (IOR 1.33,
      orange absorption, distance 0.3): the box is tinted, darker where the path is longer.
    RIGHT: a blue BLEND quad (alpha 0.5) in front of an opaque checker sphere: the sphere shows
      through it, tinted blue, not bent."""
    pair(g, -1.7, g.mesh(mk.quad(0.7), glass(g, rough=0.35, doubleSided=True)), g.mesh(mk.quad(0.7), leaf(g)))
    put(g, g.mesh(mk.sphere(0.4), glass(g, thickness=0.8)), -0.5)
    put(g, g.mesh(mk.box(0.5, 0.7, 0.5), glass(g, ior=1.33, thickness=0.5, att_color=(1, 0.55, 0.1), att_dist=0.3)), 0.4)
    checker = g.pbr((1, 1, 1, 1), 0.0, 0.5, baseColorTexture=g.texture(mk.tex_checker(), "checker"))
    pair(g, 1.7, g.mesh(mk.quad(0.6), blend(g, BLUE, 0.5, doubleSided=True)), g.mesh(mk.sphere(0.4), checker), dz=0.45)


TESTS = [alpha_blend_0_025_05_075_1, alpha_blend_factor_texture_product_vertex,
         alpha_mask_cutoff_025_default_075_below_equal,
         transmission_0_05_1_vs_alpha_0, transmission_tint_and_metallic_0_05_1,
         (transmission_roughness_0_025_05_1_thin, transmission_roughness_0_025_05_1_volume,
          transmission_roughness_step_01_thin, transmission_roughness_step_01_volume),
         (ior_none_1_133_15_24_thin, ior_none_1_133_15_24_volume),
         (refraction_modes_thin_volume_slab_box_scaled, refraction_sidedness_sphere_single_double_quad_double_single),
         absorption_depth_025_05_1_ior1_and_ior15, absorption_reference_white_nodistance_thin_basecolor,
         transmission_with_alpha_opaque_blend_mask, transmission_texture_and_thickness_texture,
         stack_blend_front_first_last_crossing_double_sided, stack_glass_panes_filters_both_orders_nested,
         (stack_blend_behind_glass_glass_behind_blend_rough_glass,
          stack_mask_front_of_glass_behind_glass_behind_rough_glass),
         stack_opaque_and_emissive_inside_and_behind_glass, combined_all_transparency]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, entry in enumerate(TESTS, 1):
        # A tuple in TESTS = variants of one test, numbered 06a, 06b, ...
        variants = [("", entry)] if callable(entry) else [(chr(ord("a") + k), t) for k, t in enumerate(entry)]
        for suffix, test in variants:
            name = "%02d%s_%s" % (i, suffix, test.__name__)
            g = mk.Gltf(name)
            test(g)
            add_backdrop(g)
            path = os.path.join(OUT_DIR, name + ".glb")
            g.save(path)
            print("wrote", path)


if __name__ == "__main__":
    main()
