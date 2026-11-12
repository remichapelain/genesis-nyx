"""Render every generated glTF test file with Nyx and save one image per test.

Usage:
    python examples/gltf_tests/make_gltf_tests.py      # create generated/*.glb first
    python examples/gltf_tests/render_gltf_tests.py               # all tests -> out/<test>_<n>_<mode>.png
    python examples/gltf_tests/render_gltf_tests.py 07 17        # only tests whose name contains "07" or "17"
    python examples/gltf_tests/render_gltf_tests.py anisotropy   # all anisotropy tests
    python examples/gltf_tests/render_gltf_tests.py sampler emissive

Each file is rendered in its own Python process, so a crash on one file
doesn't stop the others.
"""

import glob
import os
import subprocess
import sys

HERE    = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(HERE, "generated")
OUT_DIR = os.path.join(HERE, "out")
ENV_MAP = os.path.join(HERE, "..", "assets", "green_sanctuary_4k.hdr")

MODES      = ("Forward", "FastPathTracer", "RefPathTracer")
CAM_POS    = (0.0, -5.0, 1.5)   # glTF +Z (front) ends up facing Genesis -Y
CAM_LOOKAT = (0.0, 0.0, 0.6)
CAM_RES    = (1280, 640)
CAM_FOV    = 30.0

# Tests rendered WITHOUT the ground plane (matched on the start of the file name).
# The ground adds bounced light and reflections in the path tracers that Forward doesn't have,
# which makes subtle shading (normal maps) harder to compare between modes. Empty it to restore.
NO_GROUND_TESTS = ("06",)

# Extra Nyx lights added on top of the environment map, per test (matched on the start of the
# file name). Same dict format as NyxCameraOptions(lights=...), in Genesis Z-up coordinates
# (camera looks along +Y). Forward mode doesn't apply anisotropy to the environment map, only to
# lights, so the anisotropy tests need a light to show the effect in Forward.
KEY_LIGHT = {
    "type":      "directional",
    "dir":       (0.5, 0.6, -0.6),   # from the upper-left front -> highlight on the upper-left of each sphere
    "color":     (1.0, 1.0, 1.0),
    "intensity": 3.0,
}
EXTRA_LIGHTS = {
    "12": [KEY_LIGHT],
}


def render(model):
    import genesis as gs
    import gs_nyx.nyx_py_renderer as npr
    import gs_nyx.nyx_py_sdk as nps
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from PIL import Image

    gs.init()
    name = os.path.splitext(os.path.basename(model))[0]
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
    if not name.startswith(NO_GROUND_TESTS):
        scene.add_entity(morph=gs.morphs.Plane(plane_size=(20.0, 20.0)))
    # Test objects are ~1 unit tall and centred on the origin: lift them above the ground.
    scene.add_entity(morph=gs.morphs.Mesh(file=model, pos=(0.0, 0.0, 0.6), fixed=True))

    env_map            = nps.EnvironmentMapAsset()
    env_map.texture    = ENV_MAP
    env_map.layout     = nps.EEnvMapLayout.LongLat
    env_map.multiplier = 2

    lights = [l for prefix, ls in EXTRA_LIGHTS.items() if name.startswith(prefix) for l in ls]
    cam = scene.add_sensor(NyxCameraOptions(
        res=CAM_RES, pos=CAM_POS, lookat=CAM_LOOKAT, fov=CAM_FOV, spp=64,
        render_mode=getattr(npr.ERenderMode, MODES[0]), env_maps=(env_map,), lights=lights))
    scene.build(n_envs=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, mode in enumerate(MODES):
        if i > 0:
            cam._shared_metadata.renderer._renderer.set_render_mode(getattr(npr.ERenderMode, mode))
        scene.step()  # the camera renders during step
        # Number the modes so a file browser lists them in quality order.
        out = os.path.join(OUT_DIR, "%s_%d_%s.png" % (name, i + 1, mode))
        Image.fromarray(cam.read().rgb[0].cpu().numpy()).save(out)
        print("Saved", out)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":  # internal: render one file in this process
        render(sys.argv[2])
        return
    files = sorted(glob.glob(os.path.join(GEN_DIR, "*.glb")))
    if not files:
        sys.exit("No files in %s, run make_gltf_tests.py first." % GEN_DIR)
    filters = sys.argv[1:]
    if filters:
        # Keep a file if any filter is part of its name ("07", "aniso") or is its path.
        files = [f for f in files
                 if any(p in os.path.basename(f) or os.path.abspath(p) == f for p in filters)]
        if not files:
            sys.exit("No test matches %s. Available:\n  %s" % (filters, "\n  ".join(
                os.path.basename(f) for f in sorted(glob.glob(os.path.join(GEN_DIR, "*.glb"))))))
    for f in files:
        print("==>", os.path.basename(f))
        if subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", f]).returncode != 0:
            print("    FAILED:", os.path.basename(f))


if __name__ == "__main__":
    main()
