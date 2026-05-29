from __future__ import annotations

import os


def _iter_candidate_paths(base_root: str, raw_model_path: str):
    base_root = os.path.abspath(base_root)
    raw_model_path = os.path.abspath(raw_model_path) if os.path.isabs(raw_model_path) else os.path.join(base_root, raw_model_path)
    current = os.path.abspath(raw_model_path)

    while True:
        yield current
        if current == base_root:
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        if os.path.commonpath([base_root, parent]) != base_root:
            break
        current = parent


def resolve_realcar_model_path(base_root: str, raw_model_path: str) -> str:
    for candidate in _iter_candidate_paths(base_root, raw_model_path):
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "wlh.json")):
            return candidate

    raise FileNotFoundError(
        f"Unable to resolve realcar model path from {raw_model_path!r} under base root {base_root!r}."
    )
