from __future__ import annotations

import cv2
import numpy as np


FRONT_ROW = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
REAR_VISUAL_ROW = ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")


def _flip_horizontal(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, ::-1])


def build_visual_camera_grid(camera_images: dict[str, np.ndarray]) -> np.ndarray:
    """Build the display-only six-camera grid without changing raw obs semantics."""
    row1 = np.concatenate([camera_images[name] for name in FRONT_ROW], axis=1)
    row2 = np.concatenate([_flip_horizontal(camera_images[name]) for name in REAR_VISUAL_ROW], axis=1)
    return np.concatenate([row1, row2], axis=0)


def split_visual_camera_grid(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Recover raw camera images from a display grid made by build_visual_camera_grid."""
    height, width = frame.shape[:2]
    half_h = height // 2
    third_w = width // 3

    rear_left = frame[half_h:, :third_w]
    rear_center = frame[half_h:, third_w : 2 * third_w]
    rear_right = frame[half_h:, 2 * third_w :]

    return {
        "CAM_FRONT_LEFT": frame[:half_h, :third_w].copy(),
        "CAM_FRONT": frame[:half_h, third_w : 2 * third_w].copy(),
        "CAM_FRONT_RIGHT": frame[:half_h, 2 * third_w :].copy(),
        "CAM_BACK_LEFT": _flip_horizontal(rear_left).copy(),
        "CAM_BACK": _flip_horizontal(rear_center).copy(),
        "CAM_BACK_RIGHT": _flip_horizontal(rear_right).copy(),
    }


def split_legacy_camera_grid(frame: np.ndarray) -> dict[str, np.ndarray]:
    height, width = frame.shape[:2]
    half_h = height // 2
    third_w = width // 3
    return {
        "CAM_FRONT_LEFT": frame[:half_h, :third_w].copy(),
        "CAM_FRONT": frame[:half_h, third_w : 2 * third_w].copy(),
        "CAM_FRONT_RIGHT": frame[:half_h, 2 * third_w :].copy(),
        "CAM_BACK_RIGHT": frame[half_h:, :third_w].copy(),
        "CAM_BACK": frame[half_h:, third_w : 2 * third_w].copy(),
        "CAM_BACK_LEFT": frame[half_h:, 2 * third_w :].copy(),
    }


def build_visual_camera_grid_resized(camera_images: dict[str, np.ndarray]) -> np.ndarray:
    images = [camera_images[name] for name in (*FRONT_ROW, *REAR_VISUAL_ROW)]
    target_h = max(image.shape[0] for image in images)
    target_w = max(image.shape[1] for image in images)
    normalized = {
        name: (
            cv2.resize(camera_images[name], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if camera_images[name].shape[:2] != (target_h, target_w)
            else camera_images[name]
        )
        for name in (*FRONT_ROW, *REAR_VISUAL_ROW)
    }
    return build_visual_camera_grid(normalized)
