"""Render every generated glTF test file with Nyx and save one image per test.

Usage:
    python examples/gltf_tests/make_gltf_tests.py      # create generated/*.glb first
    python examples/gltf_tests/render_gltf_tests.py    # -> out/<test>_<mode>.png
    python examples/gltf_tests/render_gltf_tests.py generated/03_roughness_0_05_1.glb   # a single file

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

MODES      = ("Forward", "RefPathTracer")
CAM_POS    = (0.0, -5.0, 1.5)   # glTF +Z (front) ends up facing Genesis -Y
CAM_LOOKAT = (0.0, 0.0, 0.6)
CAM_RES    = (1280, 640)
CAM_FOV    = 30.0


def render(model):
    import genesis as gs
    import gs_nyx.nyx_py_renderer as npr
    import gs_nyx.nyx_py_sdk as nps
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from PIL import Image

    gs.init()
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane(plane_size=(20.0, 20.0)))
    # Test objects are ~1 unit tall and centred on the origin: lift them above the ground.
    scene.add_entity(morph=gs.morphs.Mesh(file=model, pos=(0.0, 0.0, 0.6), fixed=True))

    env_map            = nps.EnvironmentMapAsset()
    env_map.texture    = ENV_MAP
    env_map.layout     = nps.EEnvMapLayout.LongLat
    env_map.multiplier = 2

    cam = scene.add_sensor(NyxCameraOptions(
        res=CAM_RES, pos=CAM_POS, lookat=CAM_LOOKAT, fov=CAM_FOV, spp=64,
        render_mode=getattr(npr.ERenderMode, MODES[0]), env_maps=(env_map,)))
    scene.build(n_envs=1)

    name = os.path.splitext(os.path.basename(model))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, mode in enumerate(MODES):
        if i > 0:
            cam._shared_metadata.renderer._renderer.set_render_mode(getattr(npr.ERenderMode, mode))
        scene.step()  # the camera renders during step
        out = os.path.join(OUT_DIR, "%s_%s.png" % (name, mode))
        Image.fromarray(cam.read().rgb[0].cpu().numpy()).save(out)
        print("Saved", out)


def main():
    if len(sys.argv) > 1:
        render(os.path.abspath(sys.argv[1]))
        return
    files = sorted(glob.glob(os.path.join(GEN_DIR, "*.glb")))
    if not files:
        sys.exit("No files in %s, run make_gltf_tests.py first." % GEN_DIR)
    for f in files:
        print("==>", os.path.basename(f))
        if subprocess.run([sys.executable, os.path.abspath(__file__), f]).returncode != 0:
            print("    FAILED:", os.path.basename(f))


if __name__ == "__main__":
    main()
