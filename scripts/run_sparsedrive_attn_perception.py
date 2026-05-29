#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = Path("/OpenDataset/HUGSIM_data/scenarios/nuscenes/scene-0254-extreme-00.yaml")
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs/sim/nuscenes_eval_sparsedrive_v2_navsim.yaml"
DEFAULT_CAMERA_CONFIG = REPO_ROOT / "configs/sim/nuscenes_camera.yaml"
DEFAULT_KINEMATIC_CONFIG = REPO_ROOT / "configs/sim/kinematic.yaml"
DEFAULT_CHECKPOINT = Path("/root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt")
DEFAULT_SPARSEDRIVE_REPO = Path("/root/clone/SparseDriveV2-HF")
DEFAULT_PYTHON_BIN = Path("/root/miniconda3/envs/recondreamerNew-rl/bin/python")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/attn_perception"


@dataclass(frozen=True)
class RuntimeFiles:
    runtime_dir: Path
    bridge_path: Path
    launcher_path: Path
    base_config_path: Path


def scenario_run_name(scenario_path: Path) -> str:
    stem = scenario_path.stem
    if "-" not in stem:
        return stem
    parts = stem.split("-")
    if len(parts) >= 4 and parts[0] == "scene":
        return f"{parts[0]}-{parts[1]}_{'_'.join(parts[2:])}"
    return stem.replace("-", "_")


def build_output_dir(output_root: Path, scenario_path: Path) -> Path:
    return Path(output_root) / scenario_run_name(Path(scenario_path))


def render_attention_overlay(
    image: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    overlay = image.copy()
    if points.size == 0 or weights.size == 0:
        return overlay

    height, width = image.shape[:2]
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    count = min(len(points), len(weights))
    if count == 0:
        return overlay

    points = points[:count]
    weights = weights[:count]
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(weights)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] <= 1.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= 1.0)
        & (weights > 0.0)
    )
    if not valid.any():
        return overlay

    heat = np.zeros((height, width), dtype=np.float32)
    valid_points = points[valid]
    valid_weights = weights[valid]
    valid_weights = valid_weights / max(float(valid_weights.max()), 1e-6)
    xs = np.clip(np.round(valid_points[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
    ys = np.clip(np.round(valid_points[:, 1] * (height - 1)).astype(np.int32), 0, height - 1)
    for x, y, weight in zip(xs, ys, valid_weights):
        cv2.circle(heat, (int(x), int(y)), 11, float(weight), -1, cv2.LINE_AA)

    heat = cv2.GaussianBlur(heat, (0, 0), 5.0)
    heat = heat / max(float(heat.max()), 1e-6)
    colored = cv2.applyColorMap(np.round(heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = heat[..., None]
    return np.clip(overlay * (1.0 - alpha * mask) + colored * (alpha * mask), 0, 255).astype(np.uint8)


def build_runtime_files(
    *,
    runtime_dir: Path,
    repo_root: Path,
    sparsedrive_repo: Path,
    python_bin: Path,
    checkpoint: Path,
) -> RuntimeFiles:
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    bridge_path = runtime_dir / "sparsedrive_attn_bridge.py"
    launcher_path = runtime_dir / "sparsedrive_attn_launcher.sh"
    base_config_path = runtime_dir / "base_attn.yaml"

    bridge_path.write_text(_bridge_source(repo_root=repo_root, sparsedrive_repo=sparsedrive_repo), encoding="utf-8")
    launcher_path.write_text(
        _launcher_source(bridge_path=bridge_path, sparsedrive_repo=sparsedrive_repo, python_bin=python_bin, checkpoint=checkpoint),
        encoding="utf-8",
    )
    launcher_path.chmod(0o755)
    return RuntimeFiles(
        runtime_dir=runtime_dir,
        bridge_path=bridge_path,
        launcher_path=launcher_path,
        base_config_path=base_config_path,
    )


def write_base_config_override(source_config: Path, destination: Path, launcher_path: Path, output_root: Path) -> None:
    cfg = OmegaConf.load(source_config)
    cfg.output_dir = str(output_root) + "/"
    cfg.sparsedrive_v2_path = str(launcher_path)
    OmegaConf.save(cfg, destination)


def _launcher_source(*, bridge_path: Path, sparsedrive_repo: Path, python_bin: Path, checkpoint: Path) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -eu

        SCRIPT_DIR={shlex.quote(str(sparsedrive_repo))}
        PYTHON_BIN="${{SPARSEDRIVE_PYTHON_BIN:-{shlex.quote(str(python_bin))}}}"
        export SPARSEDRIVE_ATTN_OUTPUT_DIR="$2/sparsedrive_v2_attention"

        cd "$SCRIPT_DIR"
        export PYTHONPATH="$SCRIPT_DIR:${{PYTHONPATH:-}}"
        if [ -n "${{NUPLAN_DEVKIT_ROOT:-}}" ]; then
          export PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$PYTHONPATH"
        fi

        CUDA_VISIBLE_DEVICES="$1" "$PYTHON_BIN" {shlex.quote(str(bridge_path))} --output "$2" --checkpoint "${{SPARSEDRIVE_CHECKPOINT:-{shlex.quote(str(checkpoint))}}}"
        """
    )


def _bridge_source(*, repo_root: Path, sparsedrive_repo: Path) -> str:
    return textwrap.dedent(
        f'''\
        #!/usr/bin/env python3
        from __future__ import annotations

        import argparse
        import os
        import pickle
        import sys
        from pathlib import Path

        import cv2
        import numpy as np
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from scipy.spatial.transform import Rotation as Rotation

        HUGSIM_REPO = Path({str(repo_root)!r})
        SPARSEDRIVE_REPO = Path({str(sparsedrive_repo)!r})
        for path in (str(HUGSIM_REPO), str(SPARSEDRIVE_REPO)):
            if path not in sys.path:
                sys.path.insert(0, path)

        from hugsim.dataparser import parse_raw, trajectory_to_plan
        from hugsim.visualizer import save_visualization_frame
        from scripts.run_sparsedrive_attn_perception import render_attention_overlay


        CAMERA_ORDER = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]


        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument("--output", required=True)
            parser.add_argument("--checkpoint", default=os.environ.get("SPARSEDRIVE_CHECKPOINT"))
            return parser.parse_args()


        def _build_agent(checkpoint_path, device):
            cfg = OmegaConf.load(SPARSEDRIVE_REPO / "navsim/planning/script/config/common/agent/sparsedrive_agent.yaml")
            if checkpoint_path:
                cfg.checkpoint_path = checkpoint_path
            agent = instantiate(cfg)
            agent.initialize()
            agent.to(device)
            agent.eval()
            return agent


        def _to_device(value, device):
            if isinstance(value, torch.Tensor):
                return value.unsqueeze(0).to(device)
            if isinstance(value, dict):
                return {{key: _to_device(item, device) for key, item in value.items()}}
            if hasattr(value, "dtype"):
                return torch.as_tensor(value).unsqueeze(0).to(device)
            return value


        class AttentionCapture:
            def __init__(self):
                self.records = []
                self._original_forward = None

            def install(self):
                from navsim.agents.sparsedrive.blocks import DeformableFeatureAggregation

                if self._original_forward is not None:
                    return
                self._original_forward = DeformableFeatureAggregation.forward
                capture = self

                def patched_forward(module, instance_feature, anchor, anchor_embed, feature_maps, metas, depth_prob, return_kps_features=False, **kwargs):
                    if not getattr(module, "use_deformable_func", False):
                        return capture._original_forward(module, instance_feature, anchor, anchor_embed, feature_maps, metas, depth_prob, return_kps_features, **kwargs)

                    bs, num_anchor = instance_feature.shape[:2]
                    key_points = module.kps_generator(anchor, instance_feature)
                    points_2d, depth, mask = module.project_points(
                        key_points,
                        metas["projection_mat"],
                        metas.get("image_wh"),
                    )
                    weights = module._get_weights(instance_feature, anchor_embed, metas, mask)

                    capture.records.append(
                        {{
                            "module": module.__class__.__name__,
                            "points_2d": points_2d.detach().float().cpu().numpy(),
                            "weights": weights.detach().float().cpu().numpy(),
                            "mask": mask.detach().cpu().numpy(),
                        }}
                    )

                    points_2d_formatted = points_2d.permute(0, 2, 3, 1, 4).reshape(bs, num_anchor * module.num_pts, -1, 2)
                    weights_formatted = (
                        weights.permute(0, 1, 4, 2, 3, 5)
                        .contiguous()
                        .reshape(bs, num_anchor * module.num_pts, module.num_cams, module.num_levels, module.num_groups)
                    )
                    from navsim.agents.sparsedrive.blocks import DAF

                    if depth_prob is not None:
                        depth_formatted = depth.permute(0, 2, 3, 1).reshape(bs, num_anchor * module.num_pts, -1, 1)
                        depth_formatted = (depth_formatted - module.min_depth) / (module.max_depth - module.min_depth)
                        depth_formatted = depth_formatted * (depth_prob.shape[-1] - 1)
                        features = DAF(*feature_maps, points_2d_formatted, weights_formatted, depth_prob, depth_formatted)
                    else:
                        features = DAF(*feature_maps, points_2d_formatted, weights_formatted)

                    features = features.reshape(bs, num_anchor, module.num_pts, module.embed_dims)
                    features = features.sum(dim=2)
                    output = module.proj_drop(module.output_proj(features))
                    if module.residual_mode == "add":
                        output = output + instance_feature
                    elif module.residual_mode == "cat":
                        output = torch.cat([output, instance_feature], dim=-1)
                    return output

                DeformableFeatureAggregation.forward = patched_forward

            def reset(self):
                self.records = []


        def infer_plan(agent, agent_input, device, capture):
            feature_builder = agent.get_feature_builders()[0]
            features = feature_builder.compute_features(agent_input)
            features, _, _ = feature_builder.pipeline(features, {{}}, token="hugsim", test_mode=True)
            batched_features = {{key: _to_device(value, device) for key, value in features.items()}}
            capture.reset()
            with torch.no_grad():
                predictions, _ = agent.forward(batched_features, None)
            trajectory = predictions["trajectory"].squeeze(0).detach().cpu().numpy()
            return trajectory, trajectory_to_plan(trajectory), predictions, list(capture.records)


        def _camera_attention(records):
            by_camera = {{camera: {{"points": [], "weights": []}} for camera in CAMERA_ORDER[:3]}}
            for record in records:
                points = record["points_2d"][0]  # cams, anchors, pts, xy
                weights = record["weights"][0]  # anchors, cams, levels, pts, groups
                mask = record["mask"][0]
                camera_count = min(points.shape[0], len(CAMERA_ORDER[:3]))
                for cam_idx in range(camera_count):
                    cam_points = points[cam_idx].reshape(-1, 2)
                    cam_weights = weights[:, cam_idx].mean(axis=(1, 2)).reshape(-1)
                    cam_mask = mask[cam_idx].reshape(-1)
                    count = min(len(cam_points), len(cam_weights), len(cam_mask))
                    if count == 0:
                        continue
                    valid = cam_mask[:count].astype(bool)
                    by_camera[CAMERA_ORDER[cam_idx]]["points"].append(cam_points[:count][valid])
                    by_camera[CAMERA_ORDER[cam_idx]]["weights"].append(cam_weights[:count][valid])
            return by_camera


        def save_attention_outputs(attn_root, frame_idx, obs, info, plan, records, predictions):
            attn_root = Path(attn_root)
            for subdir in ("attention_raw", "camera_attention", "combined"):
                (attn_root / subdir).mkdir(parents=True, exist_ok=True)

            raw_path = attn_root / "attention_raw" / f"{{frame_idx:04d}}_attn.npz"
            np.savez_compressed(
                raw_path,
                **{{f"record_{{idx}}_points_2d": record["points_2d"] for idx, record in enumerate(records)}},
                **{{f"record_{{idx}}_weights": record["weights"] for idx, record in enumerate(records)}},
                **{{f"record_{{idx}}_mask": record["mask"] for idx, record in enumerate(records)}},
                trajectory=predictions["trajectory"].detach().cpu().numpy(),
                candidate_scores=predictions.get("candidate_scores", torch.empty(0)).detach().cpu().numpy(),
            )

            by_camera = _camera_attention(records)
            rendered = {{}}
            for camera_name in CAMERA_ORDER:
                image = obs.get("rgb", {{}}).get(camera_name)
                if image is None:
                    continue
                image = image.copy()
                if camera_name in by_camera and by_camera[camera_name]["points"]:
                    points = np.concatenate(by_camera[camera_name]["points"], axis=0)
                    weights = np.concatenate(by_camera[camera_name]["weights"], axis=0)
                    image = render_attention_overlay(image, points, weights)
                rendered[camera_name] = image

            from hugsim.visualizer import build_camera_grid

            camera_grid = build_camera_grid(rendered)
            camera_path = attn_root / "camera_attention" / f"{{frame_idx:04d}}.jpg"
            cv2.imwrite(str(camera_path), camera_grid)

            vis_paths = save_visualization_frame(attn_root / "combined" / f"{{frame_idx:04d}}", obs, info, plan)
            combine = cv2.imread(str(vis_paths["combine"]))
            if combine is not None:
                target_h = combine.shape[0]
                camera_panel = cv2.resize(camera_grid, (camera_grid.shape[1], target_h), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(attn_root / "combined" / f"{{frame_idx:04d}}_attn_combine.jpg"), cv2.hconcat([camera_panel, combine]))


        def main():
            args = parse_args()
            output_dir = Path(args.output)
            attn_root = Path(os.environ.get("SPARSEDRIVE_ATTN_OUTPUT_DIR", output_dir / "sparsedrive_v2_attention"))
            attn_root.mkdir(parents=True, exist_ok=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            capture = AttentionCapture()
            capture.install()
            agent = _build_agent(args.checkpoint, device)

            obs_pipe = output_dir / "obs_pipe"
            plan_pipe = output_dir / "plan_pipe"
            if not obs_pipe.exists():
                os.mkfifo(obs_pipe)
            if not plan_pipe.exists():
                os.mkfifo(plan_pipe)

            frame_root = attn_root / "parsed_frames"
            frame_root.mkdir(parents=True, exist_ok=True)
            frame_idx = 0
            while True:
                with open(obs_pipe, "rb") as pipe:
                    raw_data = pickle.loads(pipe.read())
                if raw_data == "Done":
                    break

                try:
                    parsed = parse_raw(raw_data, frame_root / f"{{frame_idx:04d}}")
                    _trajectory, plan, predictions, records = infer_plan(agent, parsed["input"], device, capture)
                    obs, info = raw_data
                    save_attention_outputs(attn_root, frame_idx, obs, info, plan, records, predictions)
                except Exception as exc:
                    print(f"[SparseDriveV2 attention bridge] inference failed: {{exc}}", flush=True)
                    plan = None

                with open(plan_pipe, "wb") as pipe:
                    pipe.write(pickle.dumps(plan))
                if plan is None:
                    break
                frame_idx += 1


        if __name__ == "__main__":
            main()
        '''
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SparseDriveV2 on HUGSIM and save deformable-attention visualizations.")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--kinematic-config", type=Path, default=DEFAULT_KINEMATIC_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sparsedrive-repo", type=Path, default=DEFAULT_SPARSEDRIVE_REPO)
    parser.add_argument("--sparsedrive-python", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sim-cuda", default="0")
    parser.add_argument("--ad-cuda", default="0")
    parser.add_argument("--pipe-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_closed_loop_command(args: argparse.Namespace, runtime_files: RuntimeFiles) -> list[str]:
    return [
        sys_executable(),
        str(REPO_ROOT / "closed_loop.py"),
        "--scenario_path",
        str(args.scenario),
        "--base_path",
        str(runtime_files.base_config_path),
        "--camera_path",
        str(args.camera_config),
        "--kinematic_path",
        str(args.kinematic_config),
        "--ad",
        "sparsedrive-v2",
        "--ad_cuda",
        str(args.ad_cuda),
    ]


def sys_executable() -> str:
    import sys

    return sys.executable


def main() -> int:
    args = parse_args()
    output_dir = build_output_dir(args.output_root, args.scenario)
    runtime_dir = output_dir / "_runtime"
    runtime_files = build_runtime_files(
        runtime_dir=runtime_dir,
        repo_root=REPO_ROOT,
        sparsedrive_repo=args.sparsedrive_repo,
        python_bin=args.sparsedrive_python,
        checkpoint=args.checkpoint,
    )
    write_base_config_override(args.base_config, runtime_files.base_config_path, runtime_files.launcher_path, args.output_root)

    command = build_closed_loop_command(args, runtime_files)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.sim_cuda)
    env["HUGSIM_PIPE_TIMEOUT_SECONDS"] = str(args.pipe_timeout_seconds)
    env["SPARSEDRIVE_CHECKPOINT"] = str(args.checkpoint)
    env["SPARSEDRIVE_PYTHON_BIN"] = str(args.sparsedrive_python)

    print("Output directory:", output_dir)
    print("Attention directory:", output_dir / "sparsedrive_v2_attention")
    print("Command:", " ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
