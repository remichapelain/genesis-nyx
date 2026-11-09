"""Materials example for the Nyx renderer plugin.

Renders a row of spheres, each with a different ``gs.surfaces`` variant, to
show how common Genesis materials look when shaded by Nyx. The scene is lit
by an HDRI environment map plus a single directional key light so both
diffuse and reflective materials read clearly.

Usage:
    uv run python examples/08_torus_sphere.py
"""

from __future__ import annotations

import os
from PIL import Image

import genesis as gs
import gs_nyx.nyx_py_renderer as npr
import gs_nyx.nyx_py_sdk as nps
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions


HERE        = os.path.dirname(__file__)
MODEL       = os.path.join(HERE, "assets", "spheres.glb")
ENV_MAP     = os.path.join(HERE, "assets", "green_sanctuary_4k.hdr")
OUTPUT_PATH_REFPATHTRACER = os.path.join(HERE, "out", "08_torus_sphere_RefPathtracer.png")
OUTPUT_PATH_FORWARD = os.path.join(HERE, "out", "08_torus_sphere_Forward.png")

CAM_POS    = (0.0, -10, 0)
CAM_LOOKAT = (0.0,  0.0, 0.0)
CAM_RES    = (1920, 1080)
CAM_FOV    = 35.0

def main() -> None:
    gs.init()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )

    scene.add_entity(morph=gs.morphs.Plane(plane_size=(10.0, 10.0)))

    scene.add_entity(
        morph=gs.morphs.Mesh(file=MODEL, pos=(0.0, 0.0, 0.1), scale=(0.1, 0.1, 0.1)),
    )

    # Describe how the env map is encoded
    env_map            = nps.EnvironmentMapAsset()
    env_map.texture    = ENV_MAP
    env_map.layout     = nps.EEnvMapLayout.LongLat
    env_map.multiplier = 8

    # Describe the Nyx camera sensor
    cam = scene.add_sensor(NyxCameraOptions(
        res         = (1920, 1080),
        pos         = (-0.5, 0.0, 0.5),
        lookat      = (0.0, 0.0, 0.1),
        fov         = 20.0,
        spp         = 64,
        render_mode = npr.ERenderMode.RefPathTracer,
        env_maps    = (env_map,),
    ))

    # Build the scene with a single env
    scene.build(n_envs=1)

    # This simulation step makes the camera sensor render the frame
    scene.step()

    # Read back the sensor frame and save it
    rgb = cam.read().rgb[0].cpu().numpy()
    os.makedirs(os.path.dirname(OUTPUT_PATH_REFPATHTRACER), exist_ok=True)
    Image.fromarray(rgb).save(OUTPUT_PATH_REFPATHTRACER)
    print(f"Saved {OUTPUT_PATH_REFPATHTRACER}")
    
    cam._shared_metadata.renderer._renderer.set_render_mode(npr.ERenderMode.Forward)
    # scene.step()
    
    # Read back the sensor frame and save it
    rgb = cam.read().rgb[0].cpu().numpy()
    os.makedirs(os.path.dirname(OUTPUT_PATH_FORWARD), exist_ok=True)
    Image.fromarray(rgb).save(OUTPUT_PATH_FORWARD)
    print(f"Saved {OUTPUT_PATH_FORWARD}")

if __name__ == "__main__":
    main()
