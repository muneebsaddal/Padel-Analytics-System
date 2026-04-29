from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_run_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path_str = value.split("=", 1)
    else:
        path_str = value
        label = Path(path_str).name
    return label.strip(), Path(path_str).resolve()


def load_args(run_dir: Path) -> dict[str, Any]:
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        return {}
    return yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}


def load_results_rows(run_dir: Path) -> list[dict[str, str]]:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results.csv in {run_dir}")
    with results_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def metric_value(row: dict[str, str], *candidates: str) -> float | None:
    for key in candidates:
        if key in row:
            return to_float(row[key])
    return None


def choose_best_row(rows: list[dict[str, str]]) -> tuple[int, dict[str, str]]:
    best_index = len(rows) - 1
    best_row = rows[best_index]
    best_fitness = metric_value(best_row, "fitness")

    if best_fitness is None:
        return best_index, best_row

    for index, row in enumerate(rows):
        fitness = metric_value(row, "fitness")
        if fitness is not None and fitness > best_fitness:
            best_fitness = fitness
            best_index = index
            best_row = row
    return best_index, best_row


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def collect_summary(label: str, run_dir: Path) -> dict[str, str]:
    args_data = load_args(run_dir)
    rows = load_results_rows(run_dir)
    best_index, best_row = choose_best_row(rows)
    final_row = rows[-1]

    summary = {
        "label": label,
        "run_dir": str(run_dir),
        "task": str(args_data.get("task", "-")),
        "model": Path(str(args_data.get("model", "-"))).name,
        "imgsz": str(args_data.get("imgsz", "-")),
        "batch": str(args_data.get("batch", "-")),
        "epochs": str(args_data.get("epochs", len(rows))),
        "best_epoch": str(best_index + 1),
        "final_epoch": str(len(rows)),
        "fitness_best": fmt(metric_value(best_row, "fitness")),
        "fitness_final": fmt(metric_value(final_row, "fitness")),
        "box_p": fmt(metric_value(best_row, "metrics/precision(B)")),
        "box_r": fmt(metric_value(best_row, "metrics/recall(B)")),
        "box_map50": fmt(metric_value(best_row, "metrics/mAP50(B)")),
        "box_map5095": fmt(metric_value(best_row, "metrics/mAP50-95(B)")),
        "pose_p": fmt(metric_value(best_row, "metrics/precision(P)")),
        "pose_r": fmt(metric_value(best_row, "metrics/recall(P)")),
        "pose_map50": fmt(metric_value(best_row, "metrics/mAP50(P)")),
        "pose_map5095": fmt(metric_value(best_row, "metrics/mAP50-95(P)")),
        "box_loss_final": fmt(metric_value(final_row, "train/box_loss")),
        "cls_loss_final": fmt(metric_value(final_row, "train/cls_loss")),
        "dfl_loss_final": fmt(metric_value(final_row, "train/dfl_loss")),
        "pose_loss_final": fmt(metric_value(final_row, "train/pose_loss")),
        "kobj_loss_final": fmt(metric_value(final_row, "train/kobj_loss")),
    }
    return summary


def markdown_table(summaries: list[dict[str, str]]) -> str:
    headers = [
        "label",
        "task",
        "model",
        "imgsz",
        "batch",
        "best_epoch",
        "box_p",
        "box_r",
        "box_map50",
        "box_map5095",
        "pose_p",
        "pose_r",
        "pose_map50",
        "pose_map5095",
        "fitness_best",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(summary.get(header, "-") for header in headers) + " |")
    return "\n".join(lines)


def detailed_markdown(summaries: list[dict[str, str]]) -> str:
    sections = ["# Baseline Metrics", "", markdown_table(summaries), "", "## Run Details", ""]
    for summary in summaries:
        sections.extend(
            [
                f"### {summary['label']}",
                f"- run_dir: `{summary['run_dir']}`",
                f"- task: `{summary['task']}`",
                f"- model: `{summary['model']}`",
                f"- imgsz: `{summary['imgsz']}`",
                f"- batch: `{summary['batch']}`",
                f"- best_epoch: `{summary['best_epoch']}` / final_epoch: `{summary['final_epoch']}`",
                f"- fitness_best: `{summary['fitness_best']}` / fitness_final: `{summary['fitness_final']}`",
                (
                    f"- detection: `P={summary['box_p']}` `R={summary['box_r']}` "
                    f"`mAP50={summary['box_map50']}` `mAP50-95={summary['box_map5095']}`"
                ),
                (
                    f"- pose: `P={summary['pose_p']}` `R={summary['pose_r']}` "
                    f"`mAP50={summary['pose_map50']}` `mAP50-95={summary['pose_map5095']}`"
                ),
                (
                    f"- final_losses: `box={summary['box_loss_final']}` `cls={summary['cls_loss_final']}` "
                    f"`dfl={summary['dfl_loss_final']}` `pose={summary['pose_loss_final']}` "
                    f"`kobj={summary['kobj_loss_final']}`"
                ),
                "",
            ]
        )
    return "\n".join(sections).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Ultralytics training runs into a baseline metrics table.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run in the form label=/abs/or/relative/path/to/run_dir. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the markdown report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summaries: list[dict[str, str]] = []

    for run_spec in args.run:
        label, run_dir = parse_run_spec(run_spec)
        summaries.append(collect_summary(label, run_dir))

    report = detailed_markdown(summaries)
    print(report)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
