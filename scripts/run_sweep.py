#!/usr/bin/env python3
"""
Hyperparameter sweep over:
    freeze_enc       : "partly" (fixed)
    use_diffusion    : [true, false]
    use_text_encoder : [true, false]

Produces 4 independent training runs (1 × 2 × 2) per dataset.

The dataset name (read from the config's ``dataset:`` field) is always
embedded in every artefact path, so crag and glas results never collide:

    configs/_sweep_crag__freeze_enc=partly__use_diffusion=true__...yaml
    configs/_sweep_glas__freeze_enc=partly__use_diffusion=true__...yaml

    outputs/logs/crag/crag__freeze_enc=partly__use_diffusion=true__.../train.log
    outputs/logs/glas/glas__freeze_enc=partly__use_diffusion=true__.../train.log

    outputs/checkpoints/crag/crag__freeze_enc=partly__use_diffusion=true.../
    outputs/checkpoints/glas/glas__freeze_enc=partly__use_diffusion=true.../

Usage:
    python scripts/run_sweep.py                          # crag (default)
    python scripts/run_sweep.py --config configs/glas.yaml
    python scripts/run_sweep.py --skip-test              # skip per-run test eval
"""
import argparse
import copy
import csv
import itertools
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
GRID = {
    "use_diffusion":    [True,     False],  # True  = diffusion-refined pseudo-labels
    "use_text_encoder": [True,     False],  # True  = CONCH text-guided fusion
}


def param_tag(key: str, value) -> str:
    """Filesystem-safe label for one param=value pair."""
    if value is True:
        return f"{key}=true"
    if value is False:
        return f"{key}=false"
    return f"{key}={value}"


def parse_run_metrics(log_path: Path) -> dict[str, str]:
    """
    Parse final metrics from one run's train.log.
    Returns 'N/A' placeholders when fields are missing.
    """
    metrics = {
        "mDice": "N/A",
        "mJaccard": "N/A",
        "best_val_iou": "N/A",
        "best_epoch": "N/A",
    }
    if not log_path.exists():
        return metrics

    test_line_re = re.compile(r"Test .* Dice:\s*([0-9]*\.?[0-9]+).*\| IoU:\s*([0-9]*\.?[0-9]+)")
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
                metrics["mDice"] = f"{float(m_test.group(1)):.4f}"
                metrics["mJaccard"] = f"{float(m_test.group(2)):.4f}"

            m_val = best_val_re.search(line)
            if m_val:
                metrics["best_val_iou"] = f"{float(m_val.group(1)):.4f}"

    if best_epoch is not None:
        metrics["best_epoch"] = str(best_epoch)
    return metrics


def write_sweep_report(dataset: str, rows: list[dict[str, str]], out_dir: Path) -> tuple[Path, Path | None]:
    """Write CSV report and (if matplotlib is available) PNG table report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"sweep_report_{dataset}_{stamp}.csv"
    png_path = out_dir / f"sweep_report_{dataset}_{stamp}.png"

    headers = ["text", "diff", "freeze", "mDice", "mJaccard", "best_val_iou", "best_epoch", "status", "elapsed"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception:
        return csv_path, None

    figure_height = max(2.8, 1.0 + 0.5 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(13.5, figure_height))
    ax.axis("off")
    col_labels = ["text", "diff", "freeze", "mDice", "mJaccard", "best valiou", "best epoch", "status", "elapsed"]
    cell_text = [[row[k] for k in headers] for row in rows]
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--config", default="configs/crag.yaml",
                        help="Base YAML config (default: configs/crag.yaml)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip full-image test evaluation after each run")
    args = parser.parse_args()

    base_cfg_path = ROOT / args.config
    with open(base_cfg_path) as fh:
        base_cfg = yaml.safe_load(fh)

    # Dataset name is used as a prefix so every artefact is self-descriptive
    dataset   = str(base_cfg.get("dataset", base_cfg_path.stem))
    base_log  = base_cfg["paths"]["log_dir"].rstrip("/")
    base_ckpt = base_cfg["paths"]["checkpoint_dir"].rstrip("/")

    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    total  = len(combos)

    print(f"Dataset    : {dataset}")
    print(f"Sweep      : {total} experiments")
    print(f"Base config: {base_cfg_path}")
    print()

    results: list[tuple[str, str, str]] = []
    report_rows: list[dict[str, str]] = []

    for idx, combo in enumerate(combos, start=1):
        params   = dict(zip(keys, combo))
        # e.g. "crag__freeze_enc=partly__use_diffusion=true__use_text_encoder=false"
        exp_name = f"{dataset}__" + "__".join(param_tag(k, v) for k, v in params.items())

        print(f"{'='*72}")
        print(f"[{idx:>2}/{total}]  {exp_name}")
        print(f"{'='*72}")

        # Build per-experiment config
        cfg = copy.deepcopy(base_cfg)
        cfg["segmentation"]["freeze_enc"] = "partly"
        cfg["train"]["use_diffusion"]     = params["use_diffusion"]
        cfg["text"]["use_text_encoder"]   = params["use_text_encoder"]
        # Sub-folders carry the full experiment name → fully self-descriptive
        cfg["paths"]["log_dir"]           = f"{base_log}/{exp_name}"
        cfg["paths"]["checkpoint_dir"]    = f"{base_ckpt}/{exp_name}"

        sweep_cfg_path = ROOT / "configs" / f"_sweep_{exp_name}.yaml"
        with open(sweep_cfg_path, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True,
                      sort_keys=False)

        # Launch training
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config", str(sweep_cfg_path),
            "--device", args.device,
        ]
        if args.skip_test:
            cmd.append("--skip-test")

        t0  = datetime.now()
        ret = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = str(datetime.now() - t0).split(".")[0]

        status = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
        results.append((exp_name, status, elapsed))

        run_log_path = ROOT / cfg["paths"]["log_dir"] / "train.log"
        metrics = parse_run_metrics(run_log_path)
        report_rows.append({
            "text": "✓" if params["use_text_encoder"] else "x",
            "diff": "✓" if params["use_diffusion"] else "x",
            "freeze": "✓",  # fixed to partly for this sweep script
            "mDice": metrics["mDice"],
            "mJaccard": metrics["mJaccard"],
            "best_val_iou": metrics["best_val_iou"],
            "best_epoch": metrics["best_epoch"],
            "status": status,
            "elapsed": elapsed,
        })

        print(f"\n  → {status}  (wall time {elapsed})\n")

    # Summary
    print(f"\n{'='*72}")
    print(f"SWEEP COMPLETE [{dataset.upper()}] — SUMMARY")
    print(f"{'='*72}")
    pad = max(len(r[0]) for r in results)
    for name, status, elapsed in results:
        marker = "✓" if status == "OK" else "✗"
        print(f"  {marker}  {name:<{pad}}  {elapsed:>8}  {status}")
    print()

    report_dir = ROOT / "outputs" / "reports" / dataset
    csv_path, png_path = write_sweep_report(dataset, report_rows, report_dir)
    print(f"Report CSV : {csv_path}")
    if png_path is not None:
        print(f"Report PNG : {png_path}")
    else:
        print("Report PNG : skipped (matplotlib not available)")
    print()

    failed = [r for r in results if r[1] != "OK"]
    if failed:
        print(f"{len(failed)} run(s) FAILED:")
        for name, status, _ in failed:
            print(f"    {name}  →  {status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
