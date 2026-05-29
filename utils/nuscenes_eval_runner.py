from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import yaml


METRICS = ("pdms", "rc", "hdscore", "nc", "dac", "ttc", "c")


def _normalize_ad_name(ad: str) -> str:
    aliases = {
        "uniad": "uniad",
        "vad": "vad",
        "ltf": "ltf",
        "sparsedrive-v2": "sparsedrive_v2",
        "sparsedrive_v2": "sparsedrive_v2",
    }
    normalized = aliases.get(ad)
    if normalized is None:
        raise NotImplementedError(f"Unsupported AD backend: {ad}")
    return normalized


@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    display_name: str
    config_path: str
    ad: str
    output_dir_prefix: str
    legacy_output_dirnames: tuple[str, ...] = ()

    @property
    def normalized_ad(self) -> str:
        return _normalize_ad_name(self.ad)

    @property
    def official_output_dirname(self) -> str:
        return f"{self.output_dir_prefix}{self.normalized_ad}"


@dataclass(frozen=True)
class TaskSpec:
    model_key: str
    display_name: str
    repeat: int
    scenario_path: Path
    output_dir: Path
    ad: str
    config_path: Path
    output_dir_prefix: str
    official_output_dirname: str
    legacy_output_dirnames: tuple[str, ...]

    @property
    def scene_id(self) -> str:
        return self.output_dir.name

    @property
    def group_id(self) -> str:
        return f"{self.model_key}/repeat_{self.repeat:02d}"

    @property
    def task_id(self) -> str:
        return f"repeat_{self.repeat:02d}__{self.model_key}__{self.scene_id}"


@dataclass(frozen=True)
class WorkerSlot:
    sim_gpu: str
    ad_gpu: str

    @property
    def label(self) -> str:
        return f"sim{self.sim_gpu}/ad{self.ad_gpu}"


MODEL_SPECS = (
    ModelSpec(
        model_key="ltf",
        display_name="LTF",
        config_path="configs/sim/nuscenes_eval_ltf.yaml",
        ad="ltf",
        output_dir_prefix="ltf_",
        legacy_output_dirnames=("nusc_ltf",),
    ),
    ModelSpec(
        model_key="uniad",
        display_name="UniAD",
        config_path="configs/sim/nuscenes_eval_uniad.yaml",
        ad="uniad",
        output_dir_prefix="uniad_",
        legacy_output_dirnames=("nusc_uniad", "uniad_uniad"),
    ),
    ModelSpec(
        model_key="sparsedrive_navsim",
        display_name="SparseDriveV2-NavSim",
        config_path="configs/sim/nuscenes_eval_sparsedrive_v2_navsim.yaml",
        ad="sparsedrive-v2",
        output_dir_prefix="sparsedrive_navsim_",
        legacy_output_dirnames=("nusc_navsim_sparsedrive_v2",),
    ),
    ModelSpec(
        model_key="sparsedrive_ppo_grpo_ver14",
        display_name="SparseDriveV2-PPO-GRPO-ver14",
        config_path="configs/sim/nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml",
        ad="sparsedrive-v2",
        output_dir_prefix="sparsedrive_ppo_grpo_ver14_",
        legacy_output_dirnames=("nusc_ppo_grpo_ver14_sparsedrive_v2",),
    ),
    ModelSpec(
        model_key="sparsedrive_craft_sparse",
        display_name="SparseDriveV2-CRAFT-sparse",
        config_path="configs/sim/nuscenes_eval_sparsedrive_v2_craft_sparse.yaml",
        ad="sparsedrive-v2",
        output_dir_prefix="sparsedrive_craft_sparse_",
        legacy_output_dirnames=(),
    ),
)


def build_task_specs(
    *,
    scenario_dir: Path,
    repo_root: Path,
    repeats: int,
    output_root: Path,
    model_keys: tuple[str, ...] | None = None,
    max_scenes: int | None = None,
) -> list[TaskSpec]:
    selected_models = [spec for spec in MODEL_SPECS if model_keys is None or spec.model_key in model_keys]
    scenario_paths = sorted(scenario_dir.glob("*.yaml"))
    if max_scenes is not None:
        scenario_paths = scenario_paths[:max_scenes]

    tasks: list[TaskSpec] = []
    for repeat in range(1, repeats + 1):
        repeat_root = output_root / f"repeat_{repeat:02d}"
        for scenario_path in scenario_paths:
            for model_spec in selected_models:
                group_root = repeat_root / model_spec.official_output_dirname
                scene_id = _scenario_output_name(scenario_path)
                tasks.append(
                    TaskSpec(
                        model_key=model_spec.model_key,
                        display_name=model_spec.display_name,
                        repeat=repeat,
                        scenario_path=scenario_path,
                        output_dir=group_root / scene_id,
                        ad=model_spec.ad,
                        config_path=repo_root / model_spec.config_path,
                        output_dir_prefix=model_spec.output_dir_prefix,
                        official_output_dirname=model_spec.official_output_dirname,
                        legacy_output_dirnames=model_spec.legacy_output_dirnames,
                    )
                )
    return tasks


def summarize_completed_results(tasks: list[TaskSpec]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for task in tasks:
        grouped.setdefault((task.model_key, task.repeat), []).append(_load_task_result(task))

    groups: list[dict[str, Any]] = []
    expected_groups = len(grouped)
    completed_groups = 0
    expected_scenes = 0
    completed_scenes = 0
    model_to_group_rows: dict[str, list[dict[str, Any]]] = {}

    for (model_key, repeat), results in sorted(grouped.items()):
        display_name = next(task.display_name for task in tasks if task.model_key == model_key and task.repeat == repeat)
        expected = len(results)
        valid = [row for row in results if row is not None]
        expected_scenes += expected
        completed_scenes += len(valid)
        if len(valid) == expected and expected > 0:
            completed_groups += 1

        group_row = {
            "model_key": model_key,
            "display_name": display_name,
            "repeat": repeat,
            "expected_scenes": expected,
            "completed_scenes": len(valid),
            "metrics": _aggregate_metric_rows(valid),
        }
        groups.append(group_row)
        model_to_group_rows.setdefault(model_key, []).append(group_row)

    models: dict[str, Any] = {}
    for model_key, group_rows in model_to_group_rows.items():
        display_name = group_rows[0]["display_name"]
        models[model_key] = {
            "display_name": display_name,
            "repeat_count": len(group_rows),
            "completed_repeats": sum(
                1 for row in group_rows if row["completed_scenes"] == row["expected_scenes"] and row["expected_scenes"] > 0
            ),
            "metrics": _aggregate_model_rows(group_rows),
        }

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "totals": {
            "expected_groups": expected_groups,
            "completed_groups": completed_groups,
            "expected_scenes": expected_scenes,
            "completed_scenes": completed_scenes,
        },
        "groups": groups,
        "models": models,
    }


def write_summary_artifacts(summary_root: Path, summary: dict[str, Any]) -> None:
    summary_root.mkdir(parents=True, exist_ok=True)
    (summary_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_group_csv(summary_root / "group_summary.csv", summary["groups"])
    _write_model_csv(summary_root / "model_summary.csv", summary["models"])
    (summary_root / "summary.md").write_text(_render_markdown_summary(summary), encoding="utf-8")


class BatchRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        tasks: list[TaskSpec],
        slots: list[WorkerSlot],
        summary_root: Path,
        retry_count: int,
    ) -> None:
        self.repo_root = repo_root
        self.tasks = tasks
        self.slots = slots
        self.summary_root = summary_root
        self.retry_count = retry_count
        self.generated_config_root = summary_root / "generated_configs"
        self.status_path = summary_root / "status.json"
        self.lock = threading.Lock()
        self.status: dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "retry_count": retry_count,
            "slots": [asdict(slot) for slot in slots],
            "tasks": {},
        }
        self._promote_existing_results()
        self._rebuild_status_from_disk()

    def run(self) -> int:
        if not self.slots:
            raise ValueError("At least one worker slot is required.")

        group_configs = self._prepare_group_configs()
        queue: Queue[TaskSpec] = Queue()
        for task in self.tasks:
            queue.put(task)

        threads = [
            threading.Thread(target=self._worker_loop, args=(slot, queue, group_configs), daemon=True)
            for slot in self.slots
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        summary = summarize_completed_results(self.tasks)
        write_summary_artifacts(self.summary_root, summary)
        self._flush_status()

        failures = [
            task_id
            for task_id, task_state in self.status["tasks"].items()
            if task_state["status"] not in {"completed", "skipped_existing"}
        ]
        return 0 if not failures else 1

    def _prepare_group_configs(self) -> dict[tuple[str, int], Path]:
        self.generated_config_root.mkdir(parents=True, exist_ok=True)
        configs: dict[tuple[str, int], Path] = {}
        unique_groups = {(task.model_key, task.repeat): task for task in self.tasks}
        for (model_key, repeat), task in unique_groups.items():
            target_dir = self.generated_config_root / f"repeat_{repeat:02d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            config_path = target_dir / f"{model_key}.yaml"
            config_data = yaml.safe_load(task.config_path.read_text(encoding="utf-8"))
            config_data["output_dir"] = str(task.output_dir.parent.parent / task.output_dir_prefix)
            config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
            configs[(model_key, repeat)] = config_path
        return configs

    def _worker_loop(self, slot: WorkerSlot, queue: Queue[TaskSpec], group_configs: dict[tuple[str, int], Path]) -> None:
        while True:
            try:
                task = queue.get_nowait()
            except Empty:
                return
            try:
                self._run_task(task, slot, group_configs[(task.model_key, task.repeat)])
            finally:
                queue.task_done()

    def _run_task(self, task: TaskSpec, slot: WorkerSlot, generated_config_path: Path) -> None:
        existing_metrics = _load_task_result(task)
        if existing_metrics is not None:
            self._update_task_status(
                task,
                {
                    "status": "skipped_existing",
                    "slot": slot.label,
                    "attempts": 0,
                    "completed_at": _utc_timestamp(),
                    "metrics": existing_metrics,
                    "resolved_output_dir": str(_resolved_task_output_dir(task)),
                },
            )
            return

        for attempt in range(1, self.retry_count + 1):
            self._reset_output_dir(task.output_dir)
            runner_log = task.output_dir / f"runner_attempt_{attempt}.log"
            self._update_task_status(
                task,
                {
                    "status": "running",
                    "slot": slot.label,
                    "attempts": attempt,
                    "started_at": _utc_timestamp(),
                    "runner_log": str(runner_log),
                },
            )

            command = [
                "pixi",
                "run",
                "python",
                str(self.repo_root / "closed_loop.py"),
                "--scenario_path",
                str(task.scenario_path),
                "--base_path",
                str(generated_config_path),
                "--camera_path",
                str(self.repo_root / "configs" / "sim" / "nuscenes_camera.yaml"),
                "--kinematic_path",
                str(self.repo_root / "configs" / "sim" / "kinematic.yaml"),
                "--ad",
                task.ad,
                "--ad_cuda",
                slot.ad_gpu,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = slot.sim_gpu

            with runner_log.open("w", encoding="utf-8") as handle:
                handle.write(f"[command] {' '.join(command)}\n")
                handle.write(f"[slot] {slot.label}\n")
                handle.flush()
                result = subprocess.run(
                    command,
                    cwd=self.repo_root,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            metrics = _load_eval_metrics(task.output_dir / "eval.json")
            if result.returncode == 0 and metrics is not None:
                self._update_task_status(
                    task,
                    {
                        "status": "completed",
                        "slot": slot.label,
                        "attempts": attempt,
                        "completed_at": _utc_timestamp(),
                        "returncode": result.returncode,
                        "metrics": metrics,
                        "resolved_output_dir": str(task.output_dir),
                    },
                )
                return

            self._update_task_status(
                task,
                {
                    "status": "failed",
                    "slot": slot.label,
                    "attempts": attempt,
                    "completed_at": _utc_timestamp(),
                    "returncode": result.returncode,
                    "error_tail": _tail_text(runner_log),
                },
            )

        print(f"[fail] {task.task_id} exhausted {self.retry_count} attempts", flush=True)

    def _promote_existing_results(self) -> None:
        for task in self.tasks:
            if task.output_dir.exists():
                continue
            for legacy_dir in _legacy_task_output_dirs(task):
                if not legacy_dir.exists():
                    continue
                task.output_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy_dir), str(task.output_dir))
                break

    def _rebuild_status_from_disk(self) -> None:
        for task in self.tasks:
            resolved_output_dir = _resolved_task_output_dir(task)
            metrics = _load_task_result(task)
            status = "pending"
            patch: dict[str, Any] = {}
            if metrics is not None:
                status = "skipped_existing"
                patch["metrics"] = metrics
                patch["completed_at"] = _utc_timestamp()
            elif task.output_dir.exists():
                status = "interrupted"

            self.status["tasks"][task.task_id] = {
                "task_id": task.task_id,
                "group_id": task.group_id,
                "model_key": task.model_key,
                "display_name": task.display_name,
                "repeat": task.repeat,
                "scenario_path": str(task.scenario_path),
                "output_dir": str(task.output_dir),
                "resolved_output_dir": str(resolved_output_dir),
                "status": status,
                **patch,
            }

    def _reset_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    def _update_task_status(self, task: TaskSpec, patch: dict[str, Any]) -> None:
        with self.lock:
            current = self.status["tasks"].get(task.task_id, {})
            current.update(
                {
                    "task_id": task.task_id,
                    "group_id": task.group_id,
                    "model_key": task.model_key,
                    "display_name": task.display_name,
                    "repeat": task.repeat,
                    "scenario_path": str(task.scenario_path),
                    "output_dir": str(task.output_dir),
                }
            )
            current.update(patch)
            self.status["tasks"][task.task_id] = current
            self._flush_status()
            summary = summarize_completed_results(self.tasks)
            write_summary_artifacts(self.summary_root, summary)
            status_value = current["status"]
            attempts = current.get("attempts", 0)
            print(f"[{status_value}] {task.task_id} ({task.display_name}, {task.scene_id}) attempts={attempts}", flush=True)

    def _flush_status(self) -> None:
        self.summary_root.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(self.status, indent=2, sort_keys=True), encoding="utf-8")


def _load_task_result(task: TaskSpec) -> dict[str, float] | None:
    for candidate_dir in _task_output_candidates(task):
        metrics = _load_eval_metrics(candidate_dir / "eval.json")
        if metrics is not None:
            return metrics
    return None


def _resolved_task_output_dir(task: TaskSpec) -> Path:
    for candidate_dir in _task_output_candidates(task):
        if candidate_dir.exists():
            return candidate_dir
    return task.output_dir


def _task_output_candidates(task: TaskSpec) -> list[Path]:
    candidates = [task.output_dir, *_legacy_task_output_dirs(task)]
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _legacy_task_output_dirs(task: TaskSpec) -> list[Path]:
    repeat_root = task.output_dir.parent.parent
    official_results_root = repeat_root.parent
    evaluate_root = official_results_root.parent
    legacy_dirs = [evaluate_root / f"repeat_{task.repeat:02d}" / dirname / task.scene_id for dirname in task.legacy_output_dirnames]
    # Older ad-hoc runs used a sibling result root and only stored some ckpts there.
    legacy_dirs.extend(
        evaluate_root / "nuscenes_4ckpt_5x" / f"repeat_{task.repeat:02d}" / dirname / task.scene_id
        for dirname in task.legacy_output_dirnames
    )
    return legacy_dirs


def _scenario_output_name(scenario_path: Path) -> str:
    parts = scenario_path.stem.split("-")
    if len(parts) >= 4 and parts[0] == "scene":
        return f"scene-{parts[1]}_{'_'.join(parts[2:])}"
    return scenario_path.stem


def _load_eval_metrics(eval_path: Path) -> dict[str, float] | None:
    if not eval_path.exists():
        return None
    try:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    metrics: dict[str, float] = {}
    for metric in METRICS:
        value = payload.get(metric)
        if value is None:
            return None
        metrics[metric] = float(value)
    return metrics


def _aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float | None]]:
    if not rows:
        return {
            metric: {"mean": None, "std": None, "min": None, "max": None}
            for metric in METRICS
        }

    aggregated: dict[str, dict[str, float | None]] = {}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        aggregated[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return aggregated


def _aggregate_model_rows(group_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    aggregated: dict[str, dict[str, float | None]] = {}
    for metric in METRICS:
        values = [row["metrics"][metric]["mean"] for row in group_rows if row["metrics"][metric]["mean"] is not None]
        aggregated[metric] = {
            "mean_of_repeats": statistics.fmean(values) if values else None,
            "std_of_repeats": statistics.stdev(values) if len(values) >= 2 else (0.0 if values else None),
            "min_repeat": min(values) if values else None,
            "max_repeat": max(values) if values else None,
        }
    return aggregated


def _write_group_csv(csv_path: Path, group_rows: list[dict[str, Any]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_key", "display_name", "repeat", "expected_scenes", "completed_scenes", *METRICS])
        for row in group_rows:
            writer.writerow(
                [
                    row["model_key"],
                    row["display_name"],
                    row["repeat"],
                    row["expected_scenes"],
                    row["completed_scenes"],
                    *[row["metrics"][metric]["mean"] for metric in METRICS],
                ]
            )


def _write_model_csv(csv_path: Path, models: dict[str, Any]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_key", "display_name", "repeat_count", "completed_repeats", *METRICS])
        for model_key, row in models.items():
            writer.writerow(
                [
                    model_key,
                    row["display_name"],
                    row["repeat_count"],
                    row["completed_repeats"],
                    *[row["metrics"][metric]["mean_of_repeats"] for metric in METRICS],
                ]
            )


def _render_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Nuscenes 4ckpt x 5 Repeats Summary",
        "",
        f"- Generated at: {summary['generated_at']} UTC",
        f"- Completed groups: {summary['totals']['completed_groups']} / {summary['totals']['expected_groups']}",
        f"- Completed scenes: {summary['totals']['completed_scenes']} / {summary['totals']['expected_scenes']}",
        "",
        "## Model Means",
        "",
        "| Model | Completed Repeats | " + " | ".join(metric.upper() for metric in METRICS) + " |",
        "| --- | ---: | " + " | ".join(["---:"] * len(METRICS)) + " |",
    ]
    for model_key, row in summary["models"].items():
        values = [
            _fmt_metric(row["metrics"][metric]["mean_of_repeats"])
            for metric in METRICS
        ]
        lines.append(f"| {row['display_name']} | {row['completed_repeats']}/{row['repeat_count']} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Group Means",
            "",
            "| Model | Repeat | Completed Scenes | " + " | ".join(metric.upper() for metric in METRICS) + " |",
            "| --- | ---: | ---: | " + " | ".join(["---:"] * len(METRICS)) + " |",
        ]
    )
    for row in summary["groups"]:
        values = [_fmt_metric(row["metrics"][metric]["mean"]) for metric in METRICS]
        lines.append(
            f"| {row['display_name']} | {row['repeat']} | {row['completed_scenes']}/{row['expected_scenes']} | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _parse_slots(slot_args: list[str]) -> list[WorkerSlot]:
    slots: list[WorkerSlot] = []
    for raw_slot in slot_args:
        sim_gpu, ad_gpu = raw_slot.split(":", maxsplit=1)
        slots.append(WorkerSlot(sim_gpu=sim_gpu, ad_gpu=ad_gpu))
    return slots


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and supervise Nuscenes 4ckpt x 5 repeat evaluation.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scenario-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--summary-root", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--slots", nargs="+", default=["0:0", "1:1", "2:2", "3:3"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    scenario_dir = (args.scenario_dir or (repo_root / "configs" / "scenarios" / "nuscenes")).resolve()
    evaluate_root = (repo_root / "outputs" / "evaluate").resolve()
    output_root = (args.output_root or (evaluate_root / "nuscenes_4ckpt_5x")).resolve()
    summary_root = (args.summary_root or (evaluate_root / "nuscenes_eval_4ckpt_5x")).resolve()
    model_keys = tuple(args.models) if args.models else None

    tasks = build_task_specs(
        scenario_dir=scenario_dir,
        repo_root=repo_root,
        repeats=args.repeats,
        output_root=output_root,
        model_keys=model_keys,
        max_scenes=args.max_scenes,
    )

    if args.dry_run:
        summary = summarize_completed_results(tasks)
        write_summary_artifacts(summary_root, summary)
        print(json.dumps(summary["totals"], indent=2, sort_keys=True))
        return 0

    slots = _parse_slots(args.slots)
    runner = BatchRunner(
        repo_root=repo_root,
        tasks=tasks,
        slots=slots,
        summary_root=summary_root,
        retry_count=args.retry_count,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
