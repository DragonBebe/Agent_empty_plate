"""
Utility functions for analysing SmolVLA evaluation notebooks and datasets.

This module centralises helpers that are shared by multiple analysis notebooks:
* parsing evaluation results stored in Jupyter notebooks
* aggregating success-rate statistics across different experiment groups
* loading recorded block positions from collected datasets
* comparing block positions from datasets with success / failure test Runs
* producing quick Matplotlib visualisations for success rates and distances
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nbformat
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except Exception:  # pragma: no cover - matplotlib might be unavailable in some envs
    plt = None  # type: ignore
    Circle = None  # type: ignore


RESULT_LABELS_CN_TO_EN = {
    "成功": "success",
    "失败": "failure",
    "超时": "timeout",
}


@dataclass
class NotebookTestResult:
    """Structured result parsed from a single evaluation notebook."""

    notebook: Path
    success_rate: Optional[float]
    total_tests: Optional[int]
    successes: Optional[int]
    failures: Optional[int]
    timeouts: Optional[int]
    block_records: List[Dict[str, object]]
    success_times: List[float]

    @property
    def label(self) -> str:
        return self.notebook.stem

    def success_positions(self) -> np.ndarray:
        return _records_to_array(self.block_records, target="success")

    def failure_positions(self) -> np.ndarray:
        return _records_to_array(self.block_records, target="failure")

    def timeout_positions(self) -> np.ndarray:
        return _records_to_array(self.block_records, target="timeout")


def _records_to_array(records: Sequence[Dict[str, object]], target: str) -> np.ndarray:
    coords: List[List[float]] = []
    for rec in records:
        if rec.get("result") == target and "position" in rec:
            pos = rec["position"]
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                coords.append([float(pos[0]), float(pos[1]), float(pos[2])])
    if coords:
        return np.asarray(coords, dtype=float)
    return np.empty((0, 3), dtype=float)


def load_notebook_test_result(notebook_path: Path | str) -> NotebookTestResult:
    """
    Parse a Jupyter notebook that contains SmolVLA evaluation logs.

    The function scans all text outputs and extracts summary statistics such as:
      * success rate
      * total / successful / failed / timeout counts
      * per-round block positions if they were printed
    """
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    nb = nbformat.read(notebook_path, as_version=4)
    lines: List[str] = []
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream" and "text" in output:
                lines.extend(str(output["text"]).replace("\r", "").splitlines())
            elif output.get("output_type") == "execute_result":
                data = output.get("data", {})
                text = data.get("text/plain")
                if isinstance(text, str):
                    lines.extend(text.replace("\r", "").splitlines())

    success_rate = _extract_first_float(lines, r"成功率[:：]\s*([0-9.]+)")
    total_tests = _extract_first_int(lines, r"总测试轮数[:：]\s*(\d+)")
    successes = _extract_first_int(lines, r"成功(?:完成)?[:：]\s*(\d+)")
    failures = _extract_first_int(lines, r"失败[:：]\s*(\d+)")
    timeouts = _extract_first_int(lines, r"超时[:：]\s*(\d+)")
    block_records = _extract_block_records(lines)
    success_times = _extract_success_times(lines)

    return NotebookTestResult(
        notebook=notebook_path,
        success_rate=success_rate,
        total_tests=total_tests,
        successes=successes,
        failures=failures,
        timeouts=timeouts,
        block_records=block_records,
        success_times=success_times,
    )


def _extract_first_float(lines: Sequence[str], pattern: str) -> Optional[float]:
    regex = re.compile(pattern)
    for line in lines:
        match = regex.search(line)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _extract_first_int(lines: Sequence[str], pattern: str) -> Optional[int]:
    regex = re.compile(pattern)
    for line in lines:
        match = regex.search(line)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _extract_block_records(lines: Sequence[str]) -> List[Dict[str, object]]:
    """
    Extract per-round block position records from textual output lines.

    Expected format (produced by the updated analysis cell):
      第01轮：成功，坐标为 (0.321, -0.198, 0.842)
    Older formats are ignored gracefully.
    """
    records: List[Dict[str, object]] = []
    pattern = re.compile(
        r"第\s*(\d+)\s*轮[：:]\s*([^\s，:,：]+)[，,]\s*坐标为\s*\(([^)]+)\)"
    )
    for line in lines:
        match = pattern.search(line.strip())
        if not match:
            continue
        round_idx = int(match.group(1))
        result_cn = match.group(2)
        result_key = RESULT_LABELS_CN_TO_EN.get(result_cn, result_cn)
        coords_text = match.group(3)
        try:
            coords = [float(val.strip()) for val in coords_text.split(",")[:3]]
        except ValueError:
            continue
        if len(coords) != 3:
            continue
        records.append(
            {
                "round": round_idx,
                "result": result_key,
                "position": coords,
            }
        )
    return records


def _extract_success_times(lines: Sequence[str]) -> List[float]:
    """
    Extract per-run success completion times from textual outputs.
    """
    # Prefer the explicit list printed in the summary if available.
    list_pattern = re.compile(r"所有成功时间[:：]\s*\[([^\]]+)\]")
    for line in lines:
        match = list_pattern.search(line)
        if match:
            candidates = re.findall(r"([0-9]+(?:\.[0-9]+)?)", match.group(1))
            return [float(item) for item in candidates]

    # Fallback: parse individual run logs.
    per_run_pattern = re.compile(r"第\s*\d+\s*轮.*?用时[:：]\s*([0-9]+(?:\.[0-9]+)?)")
    times: List[float] = []
    for line in lines:
        if "平均" in line or "最短" in line or "最长" in line:
            continue
        match = per_run_pattern.search(line)
        if match:
            times.append(float(match.group(1)))
    return times


def aggregate_success_rates(
    group_to_notebooks: Dict[str, Iterable[Path | str]]
) -> Dict[str, Dict[str, object]]:
    """
    Aggregate success rates for multiple experiment groups.

    Args:
        group_to_notebooks: mapping from group name (e.g., "5 episodes")
            to an iterable of notebook paths containing evaluation results.

    Returns:
        Dictionary containing per-group statistics such as mean / standard
        deviation of success rates and detailed run information.
    """
    summary: Dict[str, Dict[str, object]] = {}
    for group, notebooks in group_to_notebooks.items():
        runs: List[NotebookTestResult] = []
        for nb_path in notebooks:
            try:
                runs.append(load_notebook_test_result(nb_path))
            except FileNotFoundError:
                continue
        rates = [run.success_rate for run in runs if run.success_rate is not None]
        mean_rate = float(np.mean(rates)) if rates else None
        std_rate = float(np.std(rates, ddof=1)) if len(rates) > 1 else (0.0 if rates else None)
        summary[group] = {
            "runs": runs,
            "success_rates": rates,
            "mean_success_rate": mean_rate,
            "std_success_rate": std_rate,
            "num_runs": len(runs),
        }
    return summary


def plot_success_rate_summary(
    summary: Dict[str, Dict[str, object]],
    ax: Optional["plt.Axes"] = None,
    title: str | None = None,
) -> "plt.Axes":
    """
    Plot success-rate means with standard-deviation error bars for each group.
    """
    if plt is None:
        raise RuntimeError("Matplotlib is required for plotting.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    groups: List[str] = []
    means: List[float] = []
    errors: List[float] = []
    for group, data in summary.items():
        mean = data.get("mean_success_rate")
        if mean is None:
            # Skip groups without valid results to avoid plotting NaNs/None.
            continue
        groups.append(group)
        means.append(float(mean))
        std = data.get("std_success_rate")
        errors.append(float(std) if std is not None else 0.0)

    if not groups:
        raise ValueError("No success-rate data available for plotting.")

    positions = np.arange(len(groups))
    ax.bar(positions, means, yerr=errors, capsize=6, color="#4C72B0", alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=0)
    ax.set_ylabel("success-rate (%)")
    if title:
        ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return ax


def load_dataset_block_positions(dataset_root: Path | str) -> np.ndarray:
    """
    Load block positions recorded during dataset collection.

    The function scans all ``block_pose_log.json`` files under the given root
    and aggregates the XYZ coordinates into a single NumPy array.
    """
    dataset_root = Path(dataset_root)
    positions: List[List[float]] = []
    for json_path in dataset_root.rglob("block_pose_log.json"):
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue
        for entry in data:
            pose = entry.get("block_pose")
            if isinstance(pose, list) and len(pose) >= 3:
                positions.append([float(pose[0]), float(pose[1]), float(pose[2])])
    if positions:
        return np.asarray(positions, dtype=float)
    return np.empty((0, 3), dtype=float)


def load_dataset_durations(dataset_root: Path | str) -> np.ndarray:
    """
    Load execution durations (seconds) from dataset block_pose logs.
    """
    dataset_root = Path(dataset_root)
    durations: List[float] = []
    for json_path in dataset_root.rglob("block_pose_log.json"):
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue
        for entry in data:
            duration = entry.get("duration_sec")
            if isinstance(duration, (int, float)):
                durations.append(float(duration))
    if durations:
        return np.asarray(durations, dtype=float)
    return np.empty((0,), dtype=float)


def compute_min_distances(
    reference_positions: np.ndarray, query_positions: np.ndarray
) -> np.ndarray:
    """
    Compute the minimum Euclidean distance from each query position to the set
    of reference positions.
    """
    if reference_positions.size == 0 or query_positions.size == 0:
        return np.empty((0,), dtype=float)
    diffs = query_positions[:, None, :] - reference_positions[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    return np.min(distances, axis=1)


def plot_block_position_comparison(
    dataset_positions: np.ndarray,
    success_positions: np.ndarray,
    failure_positions: np.ndarray,
    ax_xy: Optional["plt.Axes"] = None,
    ax_hist: Optional["plt.Axes"] = None,
) -> Tuple["plt.Axes", "plt.Axes"]:
    """
    Produce a 2-panel visualisation comparing dataset vs success/failure positions.

    Left subplot: top-down scatter (X-Y plane).
    Right subplot: histogram of nearest-distance distributions.
    """
    if plt is None:
        raise RuntimeError("Matplotlib is required for plotting.")

    if ax_xy is None or ax_hist is None:
        fig, (ax_xy, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_xy.scatter(
        dataset_positions[:, 0] if dataset_positions.size else [],
        dataset_positions[:, 1] if dataset_positions.size else [],
        label="Dataset samples",
        c="#C0C0C0",
        alpha=0.6,
        s=35,
        edgecolor="none",
    )

    if success_positions.size:
        ax_xy.scatter(
            success_positions[:, 0],
            success_positions[:, 1],
            label="Successful rounds",
            c="#2E8B57",
            s=60,
            marker="o",
        )
    if failure_positions.size:
        ax_xy.scatter(
            failure_positions[:, 0],
            failure_positions[:, 1],
            label="Failed/timeout rounds",
            c="#D9534F",
            s=60,
            marker="x",
        )

    ax_xy.set_xlabel("Block X (m)")
    ax_xy.set_ylabel("Block Y (m)")
    ax_xy.set_title("Block placements vs dataset (top view)")
    ax_xy.legend()
    ax_xy.grid(alpha=0.3, linestyle="--")
    ax_xy.set_aspect("equal", "box")

    # Draw the usable plate area as a circle (radius derived from env config ~0.102 m).
    if dataset_positions.size and Circle is not None:
        plate_center = dataset_positions[:, :2].mean(axis=0)
        plate_radius = 0.102
        plate_outline = Circle(
            (plate_center[0], plate_center[1]),
            plate_radius,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor="#A0A0A0",
        )
        ax_xy.add_patch(plate_outline)

    success_dist = compute_min_distances(dataset_positions, success_positions)
    failure_dist = compute_min_distances(dataset_positions, failure_positions)

    bins = np.linspace(
        0,
        max(
            [success_dist.max() if success_dist.size else 0.0,
             failure_dist.max() if failure_dist.size else 0.0,
             0.05],
        ),
        15,
    )
    if success_dist.size:
        ax_hist.hist(
            success_dist,
            bins=bins,
            alpha=0.7,
            label="Success min distance to dataset",
            color="#2E8B57",
        )
    if failure_dist.size:
        ax_hist.hist(
            failure_dist,
            bins=bins,
            alpha=0.7,
            label="Failure/timeout min distance to dataset",
            color="#D9534F",
        )

    ax_hist.set_xlabel("Minimum distance to dataset (m)")
    ax_hist.set_ylabel("Runs")
    ax_hist.set_title("distance distribution")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3, linestyle="--")

    return ax_xy, ax_hist


def summarise_distance_stats(distances: np.ndarray) -> Dict[str, float]:
    """Return simple statistics for a distance array."""
    if distances.size == 0:
        return {}
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "std": float(np.std(distances, ddof=1)) if distances.size > 1 else 0.0,
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
    }


__all__ = [
    "NotebookTestResult",
    "aggregate_success_rates",
    "compute_min_distances",
    "load_dataset_block_positions",
    "load_dataset_durations",
    "load_notebook_test_result",
    "plot_block_position_comparison",
    "plot_success_rate_summary",
    "summarise_distance_stats",
]
