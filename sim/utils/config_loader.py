from __future__ import annotations

import os

from omegaconf import OmegaConf
from sim.utils.ad_config import normalize_ad_name


def load_closed_loop_cfg(*, scenario_path: str, base_path: str, camera_path: str, kinematic_path: str, ad: str):
    scenario_config = OmegaConf.load(scenario_path)
    base_config = OmegaConf.load(base_path)
    camera_config = OmegaConf.load(camera_path)
    kinematic_config = OmegaConf.load(kinematic_path)

    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config},
    )
    cfg.base.output_dir = cfg.base.output_dir + normalize_ad_name(ad)

    scene_dir = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(scene_dir, "cfg.yaml"))
    cfg.update(model_config)

    # Exported scene cfg.yaml can contain stale absolute paths from another machine.
    # Keep the model settings, but always resolve the actual scene directory locally.
    cfg.model_path = scene_dir

    output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name + "_" + cfg.scenario.mode)
    return cfg, output
