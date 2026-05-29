#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.utils.camera_grid_visualization import split_legacy_camera_grid, split_visual_camera_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-render SparseDrive visualizations from saved HUGSim outputs.")
    parser.add_argument("--scene-root", required=True, help="Scene output directory containing video.mp4, infos.pkl, data.pkl")
    parser.add_argument(
        "--model-base",
        default=os.environ.get("HUGSIM_MODEL_BASE", "/OpenDataset/HUGSIM_data/scenes/nuscenes"),
        help="Directory containing HUGSim scene folders with ground_param.pkl",
    )
    parser.add_argument("--scene-name", default=None, help="Scene name such as scene-0013; inferred from scene-root if omitted")
    parser.add_argument("--dt", type=float, default=float(os.environ.get("HUGSIM_DT", "0.25")))
    parser.add_argument("--wheelbase", type=float, default=float(os.environ.get("HUGSIM_WHEELBASE", "2.7")))
    parser.add_argument(
        "--can-bus-dir",
        default=os.environ.get("HUGSIM_CAN_BUS_DIR", "/root/clone/ReconDreamer-RL/assets/nuscenes/can_bus"),
        help="Directory containing nuScenes can_bus *_pose.json files. Falls back to ground_param differencing if missing.",
    )
    parser.add_argument(
        "--out-subdir",
        default="sparsedrive_v2_rerendered",
        help="Subdirectory under scene-root to write regenerated visualization frames",
    )
    parser.add_argument(
        "--input-grid-layout",
        choices=("visual", "legacy"),
        default="visual",
        help="Layout used by scene-root/video.mp4. Use legacy for videos generated before rear-row swap/flip.",
    )
    parser.add_argument(
        "--use-scene-meta-boxes",
        action="store_true",
        help="Experimental: fill missing obj_boxes from the original scene meta_data.json dynamics.",
    )
    return parser.parse_args()


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_scene_dynamic_box_provider(scene_name: str, model_base: Path):
    meta_path = model_base / scene_name / "meta_data.json"
    if not meta_path.is_file():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    verts_by_id = {key: np.asarray(value, dtype=np.float32) for key, value in meta.get("verts", {}).items()}
    front_frames = []
    for frame in meta.get("frames", []):
        rgb_path = str(frame.get("rgb_path", ""))
        if "/CAM_FRONT/" not in rgb_path:
            continue
        front_frames.append(frame)

    if not front_frames:
        return None

    times = np.asarray([float(frame.get("timestamp", 0.0)) for frame in front_frames], dtype=np.float32)

    def frame_near(timestamp: float | None) -> dict | None:
        query_time = 0.0 if timestamp is None else float(timestamp)
        index = int(np.searchsorted(times, query_time))
        if index >= len(times):
            index = len(times) - 1
        if index > 0 and abs(query_time - float(times[index - 1])) <= abs(float(times[index]) - query_time):
            index -= 1
        return front_frames[index]

    def boxes_near(timestamp: float | None, closed_loop_ego_box: list[float] | None = None) -> list[list[float]]:
        frame = frame_near(timestamp)
        if frame is None:
            return []

        boxes = []
        gt_ego_box = meta_front_frame_to_ego_box(frame)
        for instance_id, pose_values in frame.get("dynamics", {}).items():
            verts = verts_by_id.get(instance_id)
            if verts is None:
                continue
            pose = np.asarray(pose_values, dtype=np.float32)
            gt_box = meta_dynamic_pose_to_box(pose, verts)
            if closed_loop_ego_box is not None:
                gt_local_box = global_box_to_local(gt_box, gt_ego_box)
                gt_box = local_box_to_global(gt_local_box, np.asarray(closed_loop_ego_box, dtype=np.float32))
            boxes.append(gt_box.tolist())
        return boxes

    return SceneDynamicBoxProvider(boxes_near=boxes_near, frame_near=frame_near)


class SceneDynamicBoxProvider:
    def __init__(self, boxes_near, frame_near):
        self.boxes_near = boxes_near
        self.frame_near = frame_near


def meta_front_frame_to_ego_box(frame: dict) -> np.ndarray:
    pose = np.asarray(frame["camtoworld"], dtype=np.float32)
    yaw = Rotation.from_matrix(pose[:3, :3]).as_euler("YXZ")[0]
    return np.asarray([pose[2, 3], -pose[0, 3], -pose[1, 3], 1.6, 3.0, 1.5, -yaw], dtype=np.float32)


def meta_dynamic_pose_to_box(pose: np.ndarray, verts: np.ndarray) -> np.ndarray:
    extent = verts.max(axis=0) - verts.min(axis=0)
    yaw = Rotation.from_matrix(pose[:3, :3]).as_euler("YXZ")[0]
    return np.asarray(
        [
            float(pose[2, 3]),
            float(-pose[0, 3]),
            float(-pose[1, 3]),
            float(extent[0]),
            float(extent[1]),
            float(extent[2]),
            float(-yaw - 0.5 * np.pi),
        ],
        dtype=np.float32,
    )


def global_box_to_local(box: np.ndarray, ego_box: np.ndarray) -> np.ndarray:
    dx = float(box[0] - ego_box[0])
    dy = float(box[1] - ego_box[1])
    cos_yaw = float(np.cos(-ego_box[6]))
    sin_yaw = float(np.sin(-ego_box[6]))
    forward = dx * cos_yaw - dy * sin_yaw
    left = dx * sin_yaw + dy * cos_yaw
    return np.asarray(
        [
            -left,
            forward,
            float(box[2] - ego_box[2]),
            float(box[3]),
            float(box[4]),
            float(box[5]),
            float(box[6] - ego_box[6]),
        ],
        dtype=np.float32,
    )


def local_box_to_global(local_box: np.ndarray, ego_box: np.ndarray) -> np.ndarray:
    local_x = float(local_box[0])
    local_y = float(local_box[1])
    forward = local_y
    left = -local_x
    cos_yaw = float(np.cos(-ego_box[6]))
    sin_yaw = float(np.sin(-ego_box[6]))
    dx = forward * cos_yaw + left * sin_yaw
    dy = -forward * sin_yaw + left * cos_yaw
    return np.asarray(
        [
            float(ego_box[0] + dx),
            float(ego_box[1] + dy),
            float(ego_box[2] + local_box[2]),
            float(local_box[3]),
            float(local_box[4]),
            float(local_box[5]),
            float(ego_box[6] + local_box[6]),
        ],
        dtype=np.float32,
    )


def apply_dynamic_box_fallback(info: dict, box_provider) -> dict:
    if box_provider is None or info.get("obj_boxes"):
        return info
    info = dict(info)
    info["obj_boxes"] = box_provider.boxes_near(info.get("timestamp"), info["ego_box"])
    return info


def mirror_box_lateral_in_ego_frame(box: list[float], ego_box: list[float]) -> list[float]:
    box_array = np.asarray(box, dtype=np.float32).copy()
    ego_array = np.asarray(ego_box, dtype=np.float32)

    dx = float(box_array[0] - ego_array[0])
    dy = float(box_array[1] - ego_array[1])
    cos_yaw = float(np.cos(-ego_array[6]))
    sin_yaw = float(np.sin(-ego_array[6]))
    forward = dx * cos_yaw - dy * sin_yaw
    left = dx * sin_yaw + dy * cos_yaw

    mirrored_left = -left
    mirrored_dx = forward * cos_yaw + mirrored_left * sin_yaw
    mirrored_dy = -forward * sin_yaw + mirrored_left * cos_yaw
    box_array[0] = float(ego_array[0] + mirrored_dx)
    box_array[1] = float(ego_array[1] + mirrored_dy)

    local_yaw = float(box_array[6] - ego_array[6])
    box_array[6] = float(ego_array[6] - local_yaw)
    return box_array.tolist()


def split_camera_grid(frame: np.ndarray, layout: str = "visual") -> dict[str, np.ndarray]:
    if layout == "legacy":
        return split_legacy_camera_grid(frame)
    return split_visual_camera_grid(frame)


def load_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def planned_traj_to_local_xy(frame_record: dict, info: dict) -> np.ndarray:
    ego_x, ego_y, _, _, _, _, ego_yaw = map(float, info["ego_box"])
    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    local_points = []
    for x, y, _ in frame_record["planned_traj"]["traj"]:
        dx = float(x) - ego_x
        dy = float(y) - ego_y
        hugsim_forward = dx * cos_yaw - dy * sin_yaw
        hugsim_left = dx * sin_yaw + dy * cos_yaw
        local_x = -hugsim_left
        local_y = hugsim_forward
        local_points.append([local_x, local_y, 0.0])
    return np.asarray(local_points, dtype=np.float32)


def infer_scene_name(scene_root: Path) -> str:
    for part in scene_root.parts[::-1]:
        if part.startswith("scene-"):
            pieces = part.split("_")
            return pieces[0]
    raise RuntimeError(f"Could not infer scene name from {scene_root}")


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class ReferenceMotionProfile:
    def __init__(self, scene_name: str, model_base: Path, dt: float, wheelbase: float, can_bus_dir: Path | None = None):
        with open(model_base / scene_name / "ground_param.pkl", "rb") as f:
            cam_poses, _, _ = pickle.load(f)

        self.wheelbase = wheelbase
        self.can_bus_times: np.ndarray | None = None
        self.can_bus_speed: np.ndarray | None = None
        self.can_bus_steer: np.ndarray | None = None

        raw_poses = np.asarray(cam_poses)
        raw_yaw = Rotation.from_matrix(raw_poses[:, :3, :3]).as_euler("XYZ")[:, 1]
        raw_speed, raw_steer = self._compute_motion(
            raw_poses[:, :3, 3].astype(np.float32),
            np.unwrap(raw_yaw),
            dt,
            wheelbase,
        )
        dense_poses, source_indices = self._dense_cam_poses(raw_poses)
        self.positions = dense_poses[:, :3, 3].astype(np.float32)
        self.speed = raw_speed[source_indices].astype(np.float32)
        self.steer = raw_steer[source_indices].astype(np.float32)
        self._load_can_bus_pose(scene_name, can_bus_dir)

    def values_near(self, ego_pos: list[float] | tuple[float, ...], timestamp: float | None = None) -> tuple[float, float]:
        if self.can_bus_times is not None and self.can_bus_speed is not None:
            query_time = 0.0 if timestamp is None else float(timestamp)
            index = int(np.searchsorted(self.can_bus_times, query_time))
            if index >= len(self.can_bus_times):
                index = len(self.can_bus_times) - 1
            if index > 0:
                prev_index = index - 1
                if abs(query_time - float(self.can_bus_times[prev_index])) <= abs(float(self.can_bus_times[index]) - query_time):
                    index = prev_index
            steer = 0.0
            if self.can_bus_steer is not None:
                steer = float(self.can_bus_steer[index])
            return float(self.can_bus_speed[index]), steer

        ego = np.asarray(ego_pos, dtype=np.float32)
        distances = np.linalg.norm(self.positions - ego[:3], axis=1)
        index = int(np.argmin(distances))
        return float(self.speed[index]), float(self.steer[index])

    def _load_can_bus_pose(self, scene_name: str, can_bus_dir: Path | None) -> None:
        if can_bus_dir is None:
            return

        pose_path = Path(can_bus_dir) / f"{scene_name}_pose.json"
        if not pose_path.is_file():
            return

        try:
            with open(pose_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(rows, list):
            return

        times = []
        speeds = []
        steers = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                utime = int(row["utime"])
                vel = np.asarray(row.get("vel", []), dtype=np.float32).reshape(-1)
            except (KeyError, TypeError, ValueError):
                continue
            if vel.size == 0:
                continue

            speed = float(np.linalg.norm(vel[: min(3, vel.size)]))
            yaw_rate = 0.0
            try:
                rotation_rate = np.asarray(row.get("rotation_rate", []), dtype=np.float32).reshape(-1)
                if rotation_rate.size >= 3:
                    yaw_rate = float(rotation_rate[2])
            except (TypeError, ValueError):
                yaw_rate = 0.0

            times.append(utime)
            speeds.append(speed)
            steers.append(float(np.arctan2(self.wheelbase * yaw_rate, max(speed, 1e-3))))

        if not times:
            return

        order = np.argsort(np.asarray(times, dtype=np.int64))
        sorted_times = np.asarray(times, dtype=np.float64)[order]
        self.can_bus_times = ((sorted_times - sorted_times[0]) / 1e6).astype(np.float32)
        self.can_bus_speed = np.asarray(speeds, dtype=np.float32)[order]
        self.can_bus_steer = np.asarray(steers, dtype=np.float32)[order]

    @staticmethod
    def _compute_motion(positions: np.ndarray, yaw: np.ndarray, dt: float, wheelbase: float) -> tuple[np.ndarray, np.ndarray]:
        if len(positions) < 2:
            return np.zeros(len(positions), dtype=np.float32), np.zeros(len(positions), dtype=np.float32)
        deltas = np.diff(positions[:, [0, 2]], axis=0)
        segment_speed = np.linalg.norm(deltas, axis=1) / max(dt, 1e-6)
        yaw_rate = wrap_angle(np.diff(yaw)) / max(dt, 1e-6)
        segment_steer = np.arctan2(wheelbase * yaw_rate, np.maximum(segment_speed, 1e-3))
        return (
            np.concatenate([[segment_speed[0]], segment_speed]).astype(np.float32),
            np.concatenate([[segment_steer[0]], segment_steer]).astype(np.float32),
        )

    @staticmethod
    def _dense_cam_poses(cam_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dense_poses = cam_poses
        source_indices = np.arange(cam_poses.shape[0], dtype=np.int32)
        for _ in range(5):
            next_poses = []
            next_indices = []
            for idx in range(dense_poses.shape[0] - 1):
                pose1 = dense_poses[idx]
                pose2 = dense_poses[idx + 1]
                next_poses.append(pose1)
                next_indices.append(source_indices[idx])
                if np.linalg.norm(pose1[:3, 3] - pose2[:3, 3]) > 0.1:
                    euler1 = Rotation.from_matrix(pose1[:3, :3]).as_euler("XYZ")
                    euler2 = Rotation.from_matrix(pose2[:3, :3]).as_euler("XYZ")
                    interp_pose = np.eye(4, dtype=np.float32)
                    interp_pose[:3, :3] = Rotation.from_euler("XYZ", (euler1 + euler2) / 2.0).as_matrix()
                    interp_pose[:3, 3] = (pose1[:3, 3] + pose2[:3, 3]) / 2.0
                    next_poses.append(interp_pose)
                    next_indices.append(source_indices[idx])
            next_poses.append(dense_poses[-1])
            next_indices.append(source_indices[-1])
            dense_poses = np.stack(next_poses)
            source_indices = np.asarray(next_indices, dtype=np.int32)
        return dense_poses, source_indices


def append_motion_history(history: dict[str, list[float]], info: dict, reference_profile: ReferenceMotionProfile) -> dict:
    history["frame"].append(float(len(history["frame"])))
    history["time"].append(float(info.get("timestamp", len(history["time"]))))
    history["ego_speed"].append(float(info.get("ego_velo", np.nan)))
    history["ego_steer"].append(float(info.get("ego_steer", np.nan)))
    gt_speed, gt_steer = reference_profile.values_near(info.get("ego_pos", []), timestamp=info.get("timestamp"))
    history["gt_speed"].append(gt_speed)
    history["gt_steer"].append(gt_steer)
    return history


def save_ego_trajectory_comparison(out_dir: Path, infos: list[dict], box_provider) -> None:
    if box_provider is None:
        return

    closed_loop_xy = np.asarray([[float(info["ego_box"][0]), float(info["ego_box"][1])] for info in infos], dtype=np.float32)
    gt_points = []
    for info in infos:
        frame = box_provider.frame_near(info.get("timestamp"))
        if frame is None:
            continue
        gt_ego = meta_front_frame_to_ego_box(frame)
        gt_points.append([float(gt_ego[0]), float(gt_ego[1])])
    if len(gt_points) == 0:
        return

    gt_xy = np.asarray(gt_points, dtype=np.float32)
    all_xy = np.concatenate([closed_loop_xy, gt_xy], axis=0)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)

    size = 900
    margin = 80
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    scale = (size - 2 * margin) / float(np.max(span))
    center = (min_xy + max_xy) * 0.5

    def to_pixel(points: np.ndarray) -> np.ndarray:
        px = (points[:, 0] - center[0]) * scale + size * 0.5
        py = size * 0.5 - (points[:, 1] - center[1]) * scale
        return np.stack([px, py], axis=1).round().astype(np.int32)

    gt_pixels = to_pixel(gt_xy)
    closed_pixels = to_pixel(closed_loop_xy)
    _draw_polyline_with_points(canvas, gt_pixels, (80, 170, 60))
    _draw_polyline_with_points(canvas, closed_pixels, (220, 70, 70))

    if len(gt_pixels) > 0:
        cv2.circle(canvas, tuple(gt_pixels[0]), 7, (80, 170, 60), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(gt_pixels[-1]), 7, (80, 170, 60), 2, cv2.LINE_AA)
    if len(closed_pixels) > 0:
        cv2.circle(canvas, tuple(closed_pixels[0]), 7, (220, 70, 70), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(closed_pixels[-1]), 7, (220, 70, 70), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Ego trajectory", (34, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(canvas, (size - 280, 42), (size - 230, 42), (80, 170, 60), 4, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (size - 220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(canvas, (size - 155, 42), (size - 105, 42), (220, 70, 70), 4, cv2.LINE_AA)
    cv2.putText(canvas, "Closed", (size - 95, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "ego_trajectory_compare.jpg"), canvas)


def _draw_polyline_with_points(canvas: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    if len(points) > 1:
        cv2.polylines(canvas, [points], isClosed=False, color=color, thickness=4, lineType=cv2.LINE_AA)
    for point in points:
        cv2.circle(canvas, tuple(point), 3, color, -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    scene_root = Path(args.scene_root)
    out_dir = scene_root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    infos = load_pickle(scene_root / "infos.pkl")
    data = load_pickle(scene_root / "data.pkl")
    frame_records = data[0]["frames"]
    video_frames = load_video_frames(scene_root / "video.mp4")

    if not (len(infos) == len(frame_records) == len(video_frames)):
        raise RuntimeError(
            f"Frame count mismatch: infos={len(infos)} frame_records={len(frame_records)} video={len(video_frames)}"
        )

    sys.path.insert(0, "/root/clone/SparseDriveV2-HF")
    from hugsim.visualizer import save_visualization_frame

    scene_name = args.scene_name or infer_scene_name(scene_root)
    dynamic_box_provider = (
        load_scene_dynamic_box_provider(scene_name, Path(args.model_base)) if args.use_scene_meta_boxes else None
    )

    reference_profile = ReferenceMotionProfile(
        scene_name=scene_name,
        model_base=Path(args.model_base),
        dt=args.dt,
        wheelbase=args.wheelbase,
        can_bus_dir=Path(args.can_bus_dir) if args.can_bus_dir else None,
    )
    motion_history = {"frame": [], "time": [], "ego_speed": [], "gt_speed": [], "ego_steer": [], "gt_steer": []}

    for idx, (info, frame_record, video_frame) in enumerate(zip(infos, frame_records, video_frames)):
        obs = {"rgb": split_camera_grid(video_frame, layout=args.input_grid_layout)}
        info = apply_dynamic_box_fallback(info, dynamic_box_provider)
        plan_points = planned_traj_to_local_xy(frame_record, info)
        append_motion_history(motion_history, info, reference_profile)
        save_visualization_frame(out_dir / f"{idx:04d}", obs, info, plan_points, motion_history=motion_history)

    save_ego_trajectory_comparison(out_dir, infos, dynamic_box_provider)


if __name__ == "__main__":
    main()
