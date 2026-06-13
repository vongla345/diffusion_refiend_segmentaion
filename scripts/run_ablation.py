#!/usr/bin/env python3
"""
Generic ablation runner driven by list-valued fields in a single YAML config.

Rules
-----
* Any leaf in the config whose value is a *list* becomes an ablation axis,
  EXCEPT the explicitly excluded data lists (segmentation.aspp.rates and
  text.class_names), which are passed through unchanged.
* All axes are combined as a Cartesian product, one training job per
  combination. For example::

      use_pyramid_feature: [true, false]                  -> 2 runs
      use_text_encoder: [true, false]                      -> 4 runs
      use_ema_teacher:  [true, false]                         (2 x 2)

* When the config has no ablation axis (no non-excluded list), the base config
  is run in place; a single-row report is still produced under
  ``outputs/reports/<dataset>/`` (no group folder).

* ``--train_nums N`` repeats each config N times. Per repeat ``i`` the run's
  artefacts are indexed so they don't overwrite each other: ``train_<i>.log``,
  ``training_curves_<i>.png`` and ``..._<i>.pt`` checkpoints. Report metrics
  (test_dice / test_iou / val_iou) are then aggregated as ``mean ± std`` over
  the repeats, best_epoch as the rounded mean, plus a runs (ok/total) column.

Artefact layout (ablation)
--------------------------
Group folder name: ``<leaf1>__<leaf2>__<YYYYMMDD_HHMMSS>`` placed under each
of the dataset folders::

    configs/<dataset>/<group>/<combo>.yaml
    outputs/logs/<dataset>/<group>/<combo>/train_<i>.log
    outputs/checkpoints/<dataset>/<group>/<combo>/..._<i>.pt
    outputs/reports/<dataset>/<group>/ablation_report_<dataset>_<stamp>.{csv,png}

where ``<combo>`` is e.g. ``use_text_encoder=true__use_ema_teacher=false``.

Usage
-----
    python scripts/run_ablation.py --config configs/glas.yaml
    python scripts/run_ablation.py --config configs/crag.yaml --skip-test
    python scripts/run_ablation.py --config configs/glas.yaml --train_nums 5
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import os
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Lists that describe data, never an ablation axis (dotted paths).
EXCLUDED_LIST_PATHS = {
    "segmentation.aspp.rates",
    "text.class_names",
}


# ---------------------------------------------------------------------------
# Config tree helpers
# ---------------------------------------------------------------------------
def find_ablation_axes(cfg: dict) -> list[tuple[str, str, list]]:
    """Walk the config tree and collect ablation axes.

    Returns an ordered list of ``(dotted_path, leaf_key, values)`` for every
    list-valued leaf that is not in EXCLUDED_LIST_PATHS.
    """
    axes: list[tuple[str, str, list]] = []

    def _walk(node, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _walk(value, dotted)
            elif isinstance(value, list):
                if dotted in EXCLUDED_LIST_PATHS:
                    continue
                axes.append((dotted, key, list(value)))

    _walk(cfg, "")
    return axes


def set_by_path(cfg: dict, dotted_path: str, value) -> None:
    """Set a nested value addressed by a dotted path."""
    keys = dotted_path.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def value_tag(value) -> str:
    """Filesystem-safe label for one value."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def combo_filename(axes: list[tuple[str, str, list]], combo: tuple) -> str:
    """e.g. use_text_encoder=true__use_ema_teacher=false"""
    return "__".join(
        f"{leaf}={value_tag(val)}"
        for (_, leaf, _), val in zip(axes, combo)
    )


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
def parse_run_metrics(log_path: Path) -> dict:
    """Parse one run's train.log. Values are floats (epoch int) or None."""
    metrics: dict = {
        "test_dice": None,
        "test_iou": None,
        "best_val_iou": None,
        "best_epoch": None,
    }
    if not log_path.exists():
        return metrics

    test_line_re = re.compile(
        r"Test .* Dice:\s*([0-9]*\.?[0-9]+).*\| IoU:\s*([0-9]*\.?[0-9]+)"
    )
    best_val_re = re.compile(r"Training done\. Best val IoU:\s*([0-9]*\.?[0-9]+)")
    epoch_re = re.compile(r"Epoch\s+(\d+)/\d+")
    new_best_re = re.compile(r"New best val IoU:\s*([0-9]*\.?[0-9]+)")

    current_epoch = None
    best_epoch = None

    with open(log_path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            m_epoch = epoch_re.search(line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))

            m_best = new_best_re.search(line)
            if m_best and current_epoch is not None:
                best_epoch = current_epoch

            m_test = test_line_re.search(line)
            if m_test:
                metrics["test_dice"] = float(m_test.group(1))
                metrics["test_iou"] = float(m_test.group(2))

            m_val = best_val_re.search(line)
            if m_val:
                metrics["best_val_iou"] = float(m_val.group(1))

    metrics["best_epoch"] = best_epoch
    return metrics


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
METRIC_COLS = [
    ("test_dice", "best test_dice"),
    ("test_iou", "best test_iou"),
    ("best_val_iou", "best val_iou"),
    ("best_epoch", "best epoch"),
    ("runs", "runs (ok/total)"),
    ("config", "configs_path"),
]


def _fmt_mean_std(values, decimals: int = 4) -> str:
    """mean ± std over non-None values (sample std; 0 when a single value)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "N/A"
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    else:
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _fmt_epoch_mean(values) -> str:
    """Mean best-epoch rounded to the unit place over non-None values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "N/A"
    return str(round(sum(vals) / len(vals)))


def aggregate_metrics(per_run: list[dict], statuses: list[bool], rel_cfg: str) -> dict:
    """Collapse the N repeats of one config into a single report row."""
    return {
        "test_dice": _fmt_mean_std([m["test_dice"] for m in per_run]),
        "test_iou": _fmt_mean_std([m["test_iou"] for m in per_run]),
        "best_val_iou": _fmt_mean_std([m["best_val_iou"] for m in per_run]),
        "best_epoch": _fmt_epoch_mean([m["best_epoch"] for m in per_run]),
        "runs": f"{sum(1 for s in statuses if s)}/{len(statuses)}",
        "config": rel_cfg,
    }


def write_ablation_report(
    dataset: str,
    axis_leaves: list[str],
    rows: list[dict[str, str]],
    out_dir: Path,
    name_prefix: str = "ablation_report",
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{name_prefix}_{dataset}_{stamp}.csv"
    png_path = out_dir / f"{name_prefix}_{dataset}_{stamp}.png"

    headers = list(axis_leaves) + [key for key, _ in METRIC_COLS]
    col_labels = list(axis_leaves) + [label for _, label in METRIC_COLS]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception:
        return csv_path, None

    cell_text = [[str(row.get(k, "")) for k in headers] for row in rows]

    # Width: give configs_path room. Roughly scale by the longest path string.
    max_cfg_len = max(
        [len("configs_path")] + [len(str(row.get("config", ""))) for row in rows]
    )
    fig_width = max(14.0, 6.0 + 0.10 * max_cfg_len + 1.2 * len(axis_leaves))
    fig_height = max(2.8, 1.0 + 0.5 * (len(rows) + 1))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(headers))))
    table.scale(1.0, 1.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


# ---------------------------------------------------------------------------
# Training launch
# ---------------------------------------------------------------------------
def _terminate_process_group(proc: "subprocess.Popen") -> None:
    """Tear down the child's whole process group so no GPU process is orphaned.

    Escalates SIGINT -> SIGTERM -> SIGKILL. Killing the process *group* ensures
    DataLoader workers (and any other children of train.py) are released too,
    freeing the GPU memory they hold.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def run_training(config_rel: str, device: str, skip_test: bool) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--config",
        config_rel,
        "--device",
        device,
    ]
    if skip_test:
        cmd.append("--skip-test")

    t0 = datetime.now()
    # start_new_session=True puts the child in its own process group, so a
    # terminal Ctrl+C only interrupts this runner; we then forward the signal
    # to the whole child group and wait for the GPU memory to be released.
    proc = subprocess.Popen(cmd, cwd=str(ROOT), start_new_session=True)
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C — terminating training and releasing GPU memory...")
        _terminate_process_group(proc)
        raise
    elapsed = str(datetime.now() - t0).split(".")[0]
    return returncode, elapsed


def _finalize_run_artifacts(
    log_dir: Path, ckpt_dir: Path, dataset: str, index: int
) -> Path:
    """Rename train.log, training_curves.png and best checkpoints to carry the
    run index, so repeats of the same config don't overwrite each other.

    Returns the indexed log path (for metric parsing).
    """
    log_src = log_dir / "train.log"
    log_dst = log_dir / f"train_{index}.log"
    if log_src.exists():
        log_src.replace(log_dst)
    curve_src = log_dir / "training_curves.png"
    if curve_src.exists():
        curve_src.replace(log_dir / f"training_curves_{index}.png")
    for name in (f"seg_{dataset}_uni_conch_best.pt", "diffusion_best.pt"):
        src = ckpt_dir / name
        if src.exists():
            src.replace(ckpt_dir / f"{src.stem}_{index}{src.suffix}")
    return log_dst


def run_config_n_times(
    rel_cfg: str,
    cfg: dict,
    dataset: str,
    train_nums: int,
    device: str,
    skip_test: bool,
) -> tuple[list[dict], list[bool]]:
    """Run one config `train_nums` times, indexing artefacts per repeat.

    Returns (per_run_metrics, statuses).
    """
    log_dir = ROOT / cfg["paths"]["log_dir"]
    ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]
    per_run: list[dict] = []
    statuses: list[bool] = []
    for i in range(train_nums):
        print(f"   run {i + 1}/{train_nums} (index {i}) ...")
        returncode, elapsed = run_training(rel_cfg, device, skip_test)
        log_dst = _finalize_run_artifacts(log_dir, ckpt_dir, dataset, i)
        per_run.append(parse_run_metrics(log_dst))
        statuses.append(returncode == 0)
        status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
        print(f"     -> {status}  (wall {elapsed})")
    return per_run, statuses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic ablation runner driven by list-valued config fields."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Base YAML config (e.g. configs/glas.yaml or configs/crag.yaml)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip full-image test evaluation after each run",
    )
    parser.add_argument(
        "--train_nums",
        type=int,
        default=1,
        help="How many times to train each config (repeats; default 1). "
        "Report metrics are aggregated as mean ± std over the repeats.",
    )
    args = parser.parse_args()

    config_arg = Path(args.config)
    base_cfg_path = config_arg if config_arg.is_absolute() else ROOT / config_arg
    if not base_cfg_path.is_file():
        parser.error(f"Config not found: {base_cfg_path}")

    with open(base_cfg_path) as fh:
        base_cfg = yaml.safe_load(fh)

    dataset = str(base_cfg.get("dataset", base_cfg_path.stem))
    axes = find_ablation_axes(base_cfg)
    train_nums = max(1, int(args.train_nums))

    # --- No ablation axis: run base config in place N times, single-row report
    if not axes:
        rel_cfg = (
            str(base_cfg_path.relative_to(ROOT))
            if base_cfg_path.is_relative_to(ROOT)
            else str(base_cfg_path)
        )
        print(f"Dataset    : {dataset}")
        print("Ablation   : none (no list-valued fields) — single config")
        print(f"Repeats    : {train_nums}")
        print(f"Config     : {rel_cfg}")
        print()

        per_run, statuses = run_config_n_times(
            rel_cfg, base_cfg, dataset, train_nums, args.device, args.skip_test
        )
        row = aggregate_metrics(per_run, statuses, rel_cfg)

        report_dir = ROOT / "outputs" / "reports" / dataset
        csv_path, png_path = write_ablation_report(
            dataset, [], [row], report_dir, name_prefix="report"
        )
        print(f"\nReport CSV : {csv_path}")
        if png_path is not None:
            print(f"Report PNG : {png_path}")
        else:
            print("Report PNG : skipped (matplotlib not available)")
        n_ok = sum(1 for s in statuses if s)
        sys.exit(0 if n_ok == len(statuses) else 1)

    # --- Ablation: Cartesian product over all list axes ----------------------
    axis_leaves = [leaf for _, leaf, _ in axes]
    combos = list(itertools.product(*[values for _, _, values in axes]))
    total = len(combos)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = "__".join(axis_leaves) + f"__{stamp}"

    configs_dir = ROOT / "configs" / dataset / group
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_log = base_cfg["paths"]["log_dir"].rstrip("/")
    base_ckpt = base_cfg["paths"]["checkpoint_dir"].rstrip("/")

    print(f"Dataset    : {dataset}")
    print(f"Ablation   : {total} config(s) over axes {axis_leaves}")
    print(f"Repeats    : {train_nums} per config ({total * train_nums} runs total)")
    print(f"Group      : {group}")
    print(f"Configs    : {configs_dir.relative_to(ROOT)}")
    print()

    results: list[tuple[str, str]] = []   # (combo_name, runs_ok/total)
    report_rows: list[dict[str, str]] = []

    for idx, combo in enumerate(combos, start=1):
        combo_name = combo_filename(axes, combo)

        cfg = copy.deepcopy(base_cfg)
        for (dotted, _, _), val in zip(axes, combo):
            set_by_path(cfg, dotted, val)
        cfg["paths"]["log_dir"] = f"{base_log}/{group}/{combo_name}"
        cfg["paths"]["checkpoint_dir"] = f"{base_ckpt}/{group}/{combo_name}"

        cfg_path = configs_dir / f"{combo_name}.yaml"
        with open(cfg_path, "w") as fh:
            yaml.dump(
                cfg, fh, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
        rel_cfg = str(cfg_path.relative_to(ROOT))

        print(f"{'=' * 72}")
        print(f"[{idx:>2}/{total}]  {combo_name}   (x{train_nums})")
        print(f"         config={rel_cfg}")
        print(f"{'=' * 72}")

        per_run, statuses = run_config_n_times(
            rel_cfg, cfg, dataset, train_nums, args.device, args.skip_test
        )

        row: dict[str, str] = {}
        for (_, leaf, _), val in zip(axes, combo):
            row[leaf] = value_tag(val)
        row.update(aggregate_metrics(per_run, statuses, rel_cfg))
        report_rows.append(row)
        results.append((combo_name, row["runs"]))

        print(f"\n  → repeats OK: {row['runs']}\n")

    print(f"\n{'=' * 72}")
    print(f"ABLATION COMPLETE [{dataset.upper()}] — SUMMARY")
    print(f"{'=' * 72}")
    pad = max(len(r[0]) for r in results)
    for combo_name, runs_str in results:
        ok, tot = runs_str.split("/")
        marker = "✓" if ok == tot else "✗"
        print(f"  {marker}  {combo_name:<{pad}}  repeats OK: {runs_str}")
    print()

    report_dir = ROOT / "outputs" / "reports" / dataset / group
    csv_path, png_path = write_ablation_report(
        dataset, axis_leaves, report_rows, report_dir
    )
    print(f"Report CSV : {csv_path}")
    if png_path is not None:
        print(f"Report PNG : {png_path}")
    else:
        print("Report PNG : skipped (matplotlib not available)")
    print()

    failed = [(c, r) for c, r in results if r.split("/")[0] != r.split("/")[1]]
    if failed:
        print(f"{len(failed)} config(s) had FAILED repeats:")
        for combo_name, runs_str in failed:
            print(f"    {combo_name}  →  repeats OK: {runs_str}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAblation cancelled by user (Ctrl+C). Remaining runs skipped.")
        sys.exit(130)
