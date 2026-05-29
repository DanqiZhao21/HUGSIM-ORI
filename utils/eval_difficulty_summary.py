from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable


METRICS = ("pdms", "rc", "hdscore", "nc", "dac", "ttc", "c")
DIFFICULTIES = ("easy", "medium", "hard", "extreme")


@dataclass(frozen=True)
class EvalRecord:
    source_root: Path
    eval_path: Path
    model_key: str
    display_name: str
    repeat: int
    scene_id: str
    difficulty: str
    metrics: dict[str, float]


def collect_eval_records(roots: Iterable[Path]) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for eval_path in sorted(root.rglob("eval.json")):
            resolved = eval_path.resolve()
            if resolved in seen:
                continue
            parsed = _parse_eval_record(root.resolve(), resolved)
            if parsed is None:
                continue
            seen.add(resolved)
            records.append(parsed)
    return records


def aggregate_records(records: Iterable[EvalRecord]) -> dict[str, dict]:
    by_model: dict[str, list[EvalRecord]] = {}
    for record in records:
        by_model.setdefault(record.model_key, []).append(record)

    summary: dict[str, dict] = {}
    for model_key, model_records in sorted(by_model.items()):
        display_name = model_records[0].display_name
        all_repeats = {record.repeat for record in model_records}
        difficulties: dict[str, dict] = {}
        for difficulty in DIFFICULTIES:
            diff_records = [record for record in model_records if record.difficulty == difficulty]
            if not diff_records:
                continue
            difficulties[difficulty] = _aggregate_bucket(diff_records)
        summary[model_key] = {
            "display_name": display_name,
            "overall": _aggregate_bucket(model_records, total_repeats=all_repeats),
            "difficulties": difficulties,
        }
    return summary


def write_summary_csv(summary: dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "bucket", "completed_repeats", *[metric.upper() for metric in METRICS]])
        for model_key, payload in summary.items():
            writer.writerow(_csv_row(payload["display_name"], "overall", payload["overall"]))
            for difficulty in DIFFICULTIES:
                bucket = payload["difficulties"].get(difficulty)
                if bucket is None:
                    continue
                writer.writerow(_csv_row(payload["display_name"], difficulty, bucket))


def write_summary_markdown(summary: dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = render_summary_markdown(summary).splitlines()
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def render_summary_markdown(summary: dict[str, dict]) -> str:
    lines = ["# Evaluation Summary By Difficulty", ""]
    for _, payload in summary.items():
        lines.append(f"## {payload['display_name']}")
        lines.append("")
        lines.append("| Bucket | Completed Repeats | " + " | ".join(metric.upper() for metric in METRICS) + " |")
        lines.append("| --- | ---: | " + " | ".join(["---:"] * len(METRICS)) + " |")
        lines.append(_markdown_row("overall", payload["overall"]))
        for difficulty in DIFFICULTIES:
            bucket = payload["difficulties"].get(difficulty)
            if bucket is None:
                continue
            lines.append(_markdown_row(difficulty, bucket))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_compact_summary_markdown(summary: dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_compact_markdown(summary), encoding="utf-8")


def render_compact_markdown(summary: dict[str, dict]) -> str:
    lines = [
        "# Compact Evaluation Summary",
        "",
        "| Method | RC E | RC M | RC H | RC X | RC Avg. | HD-Score E | HD-Score M | HD-Score H | HD-Score X | HD-Score Avg. |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, payload in summary.items():
        overall = payload["overall"]["metrics"]
        difficulties = payload["difficulties"]
        rc_values = [_fmt_pct(difficulties[level]["metrics"]["rc"]) for level in DIFFICULTIES]
        hdscore_values = [_fmt_pct(difficulties[level]["metrics"]["hdscore"]) for level in DIFFICULTIES]
        lines.append(
            "| "
            + payload["display_name"]
            + " | "
            + " | ".join(
                [
                    *rc_values,
                    _fmt_pct(overall["rc"]),
                    *hdscore_values,
                    _fmt_pct(overall["hdscore"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_compact_summary_csv(summary: dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "rc_easy",
                "rc_medium",
                "rc_hard",
                "rc_extreme",
                "rc_avg",
                "hdscore_easy",
                "hdscore_medium",
                "hdscore_hard",
                "hdscore_extreme",
                "hdscore_avg",
            ]
        )
        for _, payload in summary.items():
            overall = payload["overall"]["metrics"]
            difficulties = payload["difficulties"]
            writer.writerow(
                [
                    payload["display_name"],
                    *[_fmt_pct(difficulties[level]["metrics"]["rc"]) for level in DIFFICULTIES],
                    _fmt_pct(overall["rc"]),
                    *[_fmt_pct(difficulties[level]["metrics"]["hdscore"]) for level in DIFFICULTIES],
                    _fmt_pct(overall["hdscore"]),
                ]
            )


def _parse_eval_record(source_root: Path, eval_path: Path) -> EvalRecord | None:
    try:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    metrics: dict[str, float] = {}
    for metric in METRICS:
        value = payload.get(metric)
        if value is None:
            return None
        metrics[metric] = float(value)

    parts = eval_path.parts
    try:
        repeat_part = next(part for part in parts if part.startswith("repeat_"))
        repeat = int(repeat_part.split("_", maxsplit=1)[1])
    except (StopIteration, ValueError):
        return None

    scene_dir = eval_path.parent.name
    scene_parts = scene_dir.split("_")
    if len(scene_parts) < 2:
        return None
    difficulty = scene_parts[1]
    if difficulty not in DIFFICULTIES:
        return None
    scene_id = scene_parts[0]

    model_key, display_name = _extract_model_identity(parts, scene_dir)
    if model_key is None or display_name is None:
        return None

    return EvalRecord(
        source_root=source_root,
        eval_path=eval_path,
        model_key=model_key,
        display_name=display_name,
        repeat=repeat,
        scene_id=scene_id,
        difficulty=difficulty,
        metrics=metrics,
    )


def _extract_model_identity(parts: tuple[str, ...], scene_dir: str) -> tuple[str | None, str | None]:
    try:
        scene_index = parts.index(scene_dir)
    except ValueError:
        return None, None
    if scene_index < 2:
        return None, None
    model_part = parts[scene_index - 1]
    repeat_part = parts[scene_index - 2]
    if not repeat_part.startswith("repeat_"):
        return None, None
    if scene_index >= 4 and parts[scene_index - 4] == "results":
        ckpt_name = parts[scene_index - 3]
        return ckpt_name, ckpt_name
    return model_part, model_part


def _aggregate_bucket(records: list[EvalRecord], total_repeats: set[int] | None = None) -> dict:
    repeats_present = {record.repeat for record in records}
    repeats_total = total_repeats if total_repeats is not None else repeats_present
    metrics = {
        metric: fmean(record.metrics[metric] for record in records)
        for metric in METRICS
    }
    return {
        "completed_repeats": f"{len(repeats_present)}/{len(repeats_total)}",
        "metrics": metrics,
        "scene_count": len(records),
    }


def _csv_row(model_name: str, bucket_name: str, bucket: dict) -> list[str | float]:
    return [model_name, bucket_name, bucket["completed_repeats"], *[bucket["metrics"][metric] for metric in METRICS]]


def _markdown_row(bucket_name: str, bucket: dict) -> str:
    values = [f"{bucket['metrics'][metric]:.6f}" for metric in METRICS]
    return f"| {bucket_name} | {bucket['completed_repeats']} | " + " | ".join(values) + " |"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate eval.json results by difficulty.")
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            Path("/root/clone/HUGSIM-ORI/outputs/evaluate-auto"),
            Path("/root/clone/HUGSIM-ORI/outputs/evaluate"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/clone/HUGSIM-ORI/outputs/evaluate-difficulty-summary"),
    )
    args = parser.parse_args()

    records = collect_eval_records(args.roots)
    summary = aggregate_records(records)
    write_summary_markdown(summary, args.output_dir / "summary_by_difficulty.md")
    write_summary_csv(summary, args.output_dir / "summary_by_difficulty.csv")
    write_compact_summary_markdown(summary, args.output_dir / "summary_compact.md")
    write_compact_summary_csv(summary, args.output_dir / "summary_compact.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
