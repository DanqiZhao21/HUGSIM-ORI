from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.nuscenes_eval_runner import (
    _load_eval_metrics,
    build_task_specs,
    summarize_completed_results,
    write_summary_artifacts,
)


DEFAULT_RESIDUAL_PATTERNS = (
    "closed_loop.py",
    "tools/closeloop/e2e.py",
    "uniad_e2e.sh",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume only missing UniAD evaluation tasks.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/OpenDataset/zhaodanqi/HUGSIM_data/outputs/evaluate/nuscenes_4ckpt_5x"),
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("/root/clone/HUGSIM-ORI/outputs/evaluate/nuscenes_eval_uniad_resume_manual"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--retry-count", type=int, default=1)
    parser.add_argument("--slots", nargs="+", default=["0:1"])
    parser.add_argument("--cooldown-seconds", type=float, default=20.0)
    parser.add_argument("--cleanup-grace-seconds", type=float, default=5.0)
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def parse_slots(slot_args: list[str]) -> list[tuple[str, str]]:
    slots: list[tuple[str, str]] = []
    for raw_slot in slot_args:
        sim_gpu, ad_gpu = raw_slot.split(":", maxsplit=1)
        slots.append((sim_gpu, ad_gpu))
    return slots


def validate_runtime_options(args: argparse.Namespace) -> None:
    if not args.skip_cleanup and len(args.slots) != 1:
        raise ValueError("Residual-process cleanup currently requires a single slot to avoid cross-worker interference.")


def find_residual_processes(
    ps_rows: list[str],
    *,
    patterns: tuple[str, ...] = DEFAULT_RESIDUAL_PATTERNS,
    current_pid: int | None = None,
) -> list[int]:
    residual_pids: list[int] = []
    for row in ps_rows:
        row = row.strip()
        if not row:
            continue
        pid_text, _, command = row.partition(" ")
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if current_pid is not None and pid == current_pid:
            continue
        if any(pattern in command for pattern in patterns):
            residual_pids.append(pid)
    return residual_pids


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_residual_processes(
    ps_rows: list[str],
    *,
    patterns: tuple[str, ...] = DEFAULT_RESIDUAL_PATTERNS,
    current_pid: int | None = None,
    signal_fn=os.kill,
    exists_fn=_pid_exists,
    sleep_fn=time.sleep,
    grace_seconds: float = 5.0,
    logger=print,
) -> list[int]:
    residual_pids = find_residual_processes(ps_rows, patterns=patterns, current_pid=current_pid)
    if not residual_pids:
        return []

    logger(f"[cleanup] terminating residual_pids={residual_pids}")
    for pid in residual_pids:
        try:
            signal_fn(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    sleep_fn(grace_seconds)

    survivors = [pid for pid in residual_pids if exists_fn(pid)]
    for pid in survivors:
        try:
            signal_fn(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    if survivors:
        logger(f"[cleanup] killed stubborn_pids={survivors}")
    return residual_pids


def _list_process_rows() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def refresh_summary(summary_root: Path, tasks) -> None:
    summary = summarize_completed_results(tasks)
    write_summary_artifacts(summary_root, summary)


def main() -> int:
    args = parse_args()
    validate_runtime_options(args)
    repo_root = args.repo_root.resolve()
    scenario_dir = (repo_root / "configs" / "scenarios" / "nuscenes").resolve()
    config_root = (repo_root / "outputs" / "evaluate" / "nuscenes_eval_4ckpt_5x" / "generated_configs").resolve()
    camera_path = (repo_root / "configs" / "sim" / "nuscenes_camera.yaml").resolve()
    kinematic_path = (repo_root / "configs" / "sim" / "kinematic.yaml").resolve()
    summary_root = args.summary_root.resolve()
    output_root = args.output_root.resolve()
    slots = parse_slots(args.slots)

    all_tasks = build_task_specs(
        scenario_dir=scenario_dir,
        repo_root=repo_root,
        repeats=args.repeats,
        output_root=output_root,
        model_keys=("uniad",),
    )
    missing_tasks = [task for task in all_tasks if _load_eval_metrics(task.output_dir / "eval.json") is None]

    summary_root.mkdir(parents=True, exist_ok=True)
    refresh_summary(summary_root, all_tasks)

    print(f"[start] total_tasks={len(all_tasks)} missing_tasks={len(missing_tasks)} slots={slots}", flush=True)
    if args.dry_run:
        return 0
    if not missing_tasks:
        print("[finish] nothing to resume", flush=True)
        return 0

    queue: Queue = Queue()
    for task in missing_tasks:
        queue.put(task)

    lock = threading.Lock()
    progress = {"done": 0, "failed": 0}

    def worker(sim_gpu: str, ad_gpu: str) -> None:
        while True:
            try:
                task = queue.get_nowait()
            except Empty:
                return

            try:
                config_path = config_root / f"repeat_{task.repeat:02d}" / "uniad.yaml"
                success = False
                for attempt in range(1, args.retry_count + 1):
                    if not args.skip_cleanup:
                        cleanup_residual_processes(
                            _list_process_rows(),
                            current_pid=os.getpid(),
                            grace_seconds=args.cleanup_grace_seconds,
                            logger=lambda message: print(
                                f"{message} before={task.task_id} slot=sim{sim_gpu}/ad{ad_gpu}",
                                flush=True,
                            ),
                        )
                    if task.output_dir.exists():
                        shutil.rmtree(task.output_dir)
                    task.output_dir.mkdir(parents=True, exist_ok=True)
                    runner_log = task.output_dir / f"manual_resume_attempt_{attempt}.log"

                    command = [
                        "pixi",
                        "run",
                        "python",
                        str(repo_root / "closed_loop.py"),
                        "--scenario_path",
                        str(task.scenario_path),
                        "--base_path",
                        str(config_path),
                        "--camera_path",
                        str(camera_path),
                        "--kinematic_path",
                        str(kinematic_path),
                        "--ad",
                        task.ad,
                        "--ad_cuda",
                        ad_gpu,
                    ]
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = sim_gpu
                    start = time.time()

                    with runner_log.open("w", encoding="utf-8") as handle:
                        handle.write("[command] " + " ".join(command) + "\n")
                        handle.write(f"[slot] sim{sim_gpu}/ad{ad_gpu}\n")
                        handle.flush()
                        result = subprocess.run(
                            command,
                            cwd=repo_root,
                            env=env,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )

                    metrics = _load_eval_metrics(task.output_dir / "eval.json")
                    elapsed = time.time() - start

                    if result.returncode == 0 and metrics is not None:
                        with lock:
                            progress["done"] += 1
                            refresh_summary(summary_root, all_tasks)
                            print(
                                f"[completed] {task.task_id} slot=sim{sim_gpu}/ad{ad_gpu} "
                                f"attempt={attempt} elapsed={elapsed:.1f}s done={progress['done']}/{len(missing_tasks)}",
                                flush=True,
                            )
                        success = True
                        break

                    print(
                        f"[retry] {task.task_id} slot=sim{sim_gpu}/ad{ad_gpu} "
                        f"attempt={attempt} returncode={result.returncode} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    if attempt < args.retry_count and args.cooldown_seconds > 0:
                        print(
                            f"[cooldown] {task.task_id} slot=sim{sim_gpu}/ad{ad_gpu} "
                            f"sleep={args.cooldown_seconds:.1f}s",
                            flush=True,
                        )
                        time.sleep(args.cooldown_seconds)

                if not success:
                    with lock:
                        progress["failed"] += 1
                        refresh_summary(summary_root, all_tasks)
                        print(
                            f"[failed] {task.task_id} slot=sim{sim_gpu}/ad{ad_gpu} failed={progress['failed']}",
                            flush=True,
                        )
            finally:
                queue.task_done()

    threads = [threading.Thread(target=worker, args=slot, daemon=True) for slot in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    refresh_summary(summary_root, all_tasks)
    remaining = [task.task_id for task in all_tasks if _load_eval_metrics(task.output_dir / "eval.json") is None]
    print(f"[finish] remaining={len(remaining)} failed={progress['failed']}", flush=True)
    if remaining:
        for task_id in remaining[:20]:
            print(f"[remaining_task] {task_id}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
