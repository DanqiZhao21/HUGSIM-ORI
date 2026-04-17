from __future__ import annotations

from typing import Any

import numpy as np

from sim.utils.sim_utils import traj_transform_to_global


def build_frame_record(plan_traj, info: dict[str, Any]) -> dict[str, Any] | None:
    if plan_traj is None:
        return None

    imu_plan_traj = np.asarray(plan_traj, dtype=np.float32)[:, [1, 0]]
    imu_plan_traj[:, 1] *= -1
    global_traj = traj_transform_to_global(imu_plan_traj, info["ego_box"])

    return {
        "time_stamp": info["timestamp"],
        "is_key_frame": True,
        "ego_box": info["ego_box"],
        "obj_boxes": info["obj_boxes"],
        "obj_names": ["car" for _ in info["obj_boxes"]],
        "planned_traj": {
            "traj": global_traj,
            "timestep": 0.5,
        },
        "collision": info["collision"],
        "rc": info["rc"],
    }
