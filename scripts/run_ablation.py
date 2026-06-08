#!/usr/bin/env python3
"""
Run pseudo-label ablation experiments (one training job per mode).

Modes (train.pseudo_ablation):
    strong_weak   – FixMatch: strong logits vs refined weak mask
    weak_weak     – DiffRect L_Rect: weak logits vs refined weak mask
    strong_strong – self-distillation: strong logits vs refined strong mask
    diffrect      – weak_weak + strong_weak combined

Uses pre-made configs in configs/ablations/ when present, otherwise generates
them from a base YAML (same idea as run_sweep.py but isolated from that grid).

Usage:
    # Run all 4 GLAS ablations (configs/ablations/glas_*.yaml)
    python scripts/run_ablation.py --dataset glas

    # Generate + run from base config (e.g. CRAG — no pre-made ablation yamls)
    python scripts/run_ablation.py --base-config configs/crag.yaml

    # Subset of modes
    python scripts/run_ablation.py --dataset glas --only weak_weak,diffrect

    # Training only, no full-image test eval
    python scripts/run_ablation.py --dataset glas --skip-test
"""
from __future__ import annotations

import argparse
import copy
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

ABLATION_MODES = ("strong_weak", "weak_weak", "strong_strong", "diffrect")


def parse_run_metrics(log_path: Path) -> dict[str, str]:
    metrics = {
        "mDice": "N/A",
        "mJaccard": "N/A",
        "best_val_iou": "N/A",
        "best_epoch": "N/A",
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
                metrics["mDice"] = f"{float(m_test.group(1)):.4f}"
                metrics["mJaccard"] = f"{float(m_test.group(2)):.4f}"

            m_val = best_val_re.search(line)
            if m_val:
                metrics["best_val_iou"] = f"{float(m_val.group(1)):.4f}"

    if best_epoch is not None:
        metrics["best_epoch"] = str(best_epoch)
    return metrics


def write_ablation_report(
    dataset: str, rows: list[dict[str, str]], out_dir: Path
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"ablation_report_{dataset}_{stamp}.csv"
    png_path = out_dir / f"ablation_report_{dataset}_{stamp}.png"

    headers = [
        "mode",
        "teacher",
        "mDice",
        "mJaccard",
        "best_val_iou",
        "best_epoch",
        "status",
        "elapsed",
        "config",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception:
        return csv_path, None

    figure_height = max(2.8, 1.0 + 0.5 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(14.0, figure_height))
    ax.axis("off")
    col_labels = [
        "mode",
        "teacher",
        "mDice",
        "mJaccard",
        "best val iou",
        "best epoch",
        "status",
        "elapsed",
        "config",
    ]
    cell_text = [[row[k] for k in headers] for row in rows]
    table = ax.table(
        cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path


def discover_ablation_configs(dataset: str, ablations_dir: Path) -> dict[str, Path]:
    """Return {mode: yaml_path} for pre-made configs like glas_strong_weak.yaml."""
    found: dict[str, Path] = {}
    for mode in ABLATION_MODES:
        path = ablations_dir / f"{dataset}_{mode}.yaml"
        if path.is_file():
            found[mode] = path
    return found


def build_ablation_config(
    base_cfg: dict,
    mode: str,
    use_ema_teacher: bool = True,
    ema_alpha: float = 0.999,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    dataset = str(cfg.get("dataset", "run"))
    base_log = cfg["paths"]["log_dir"].rstrip("/")
    base_ckpt = cfg["paths"]["checkpoint_dir"].rstrip("/")
    teacher_tag = "ema" if use_ema_teacher else "no_ema"
    exp_name = f"ablation_{mode}_{teacher_tag}"

    cfg["train"]["use_diffusion"] = True
    cfg["train"]["pseudo_ablation"] = mode
    cfg["train"]["use_ema_teacher"] = use_ema_teacher
    cfg["train"]["ema_alpha"] = ema_alpha
    cfg["paths"]["log_dir"] = f"{base_log}/{exp_name}"
    cfg["paths"]["checkpoint_dir"] = f"{base_ckpt}/{exp_name}"
    return cfg


def resolve_run_configs(
    dataset: str | None,
    base_config: Path | None,
    modes: list[str],
    ablations_dir: Path,
    write_dir: Path,
    use_ema_teacher: bool = True,
    ema_alpha: float = 0.999,
) -> list[tuple[str, Path]]:
    """Return ordered list of (mode, config_path) to pass to train.py.

    Resolution order:
      1. Explicit --base-config path  → use directly
      2. configs/{dataset}.yaml found → copy + patch (preferred)
      3. Pre-made configs/ablations/{dataset}_{mode}.yaml → copy + patch (fallback)

    In all cases only the ablation-specific keys are overridden:
      use_diffusion, pseudo_ablation, use_ema_teacher, ema_alpha, output paths.
    All other settings (lr, epochs, freeze_enc, …) come unchanged from the source.
    """
    if dataset is None and base_config is None:
        raise ValueError("Provide --dataset or --base-config")

    write_dir.mkdir(parents=True, exist_ok=True)
    teacher_tag = "ema" if use_ema_teacher else "no_ema"
    runs: list[tuple[str, Path]] = []

    # --- Resolve source config -----------------------------------------------
    if base_config is not None:
        source_path = base_config
    else:
        # Try the main config first: configs/{dataset}.yaml
        candidate = ROOT / "configs" / f"{dataset}.yaml"
        if candidate.is_file():
            source_path = candidate
        else:
            source_path = None   # fall back to pre-made ablation configs

    # --- Generate one config per mode from a single source -------------------
    if source_path is not None:
        with open(source_path) as fh:
            base_cfg = yaml.safe_load(fh)
        ds = dataset or str(base_cfg.get("dataset", source_path.stem))
        for mode in modes:
            cfg = build_ablation_config(base_cfg, mode, use_ema_teacher, ema_alpha)
            out_path = write_dir / f"_{ds}_ablation_{mode}_{teacher_tag}.yaml"
            with open(out_path, "w") as fh:
                yaml.dump(cfg, fh, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)
            runs.append((mode, out_path))
        return runs

    # --- Fallback: pre-made ablation configs ---------------------------------
    pre_made = discover_ablation_configs(dataset, ablations_dir)
    missing = [m for m in modes if m not in pre_made]
    if missing:
        raise FileNotFoundError(
            f"No configs/{dataset}.yaml found and no pre-made ablation configs for "
            f"{dataset}: {missing}. Create configs/{dataset}.yaml or add "
            f"configs/ablations/{{dataset}}_{{mode}}.yaml files."
        )

    for mode in modes:
        # Read the pre-made config, patch only ablation-specific keys, write a copy.
        with open(pre_made[mode]) as fh:
            cfg = yaml.safe_load(fh)
        cfg["train"]["use_ema_teacher"] = use_ema_teacher
        cfg["train"]["ema_alpha"] = ema_alpha
        for path_key in ("log_dir", "checkpoint_dir"):
            base_p = cfg["paths"][path_key].rstrip("/")
            cfg["paths"][path_key] = f"{base_p}_{teacher_tag}"
        out_path = write_dir / f"_{dataset}_{mode}_{teacher_tag}.yaml"
        with open(out_path, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        runs.append((mode, out_path))

    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pseudo-label ablation experiments (4 modes, separate from run_sweep.py)"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name prefix for configs/ablations/{dataset}_{mode}.yaml (e.g. glas)",
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Base YAML; generates per-mode configs under configs/ablations/_generated/",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of modes (default: all four)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip full-image test evaluation after each run",
    )
    parser.add_argument(
        "--ablations-dir",
        default="configs/ablations",
        help="Directory with pre-made ablation YAML files",
    )
    # --- EMA teacher ---
    teacher_grp = parser.add_mutually_exclusive_group()
    teacher_grp.add_argument(
        "--use-teacher",
        dest="use_ema_teacher",
        action="store_true",
        default=True,
        help="Enable EMA teacher for pseudo-label generation (default: on)",
    )
    teacher_grp.add_argument(
        "--no-teacher",
        dest="use_ema_teacher",
        action="store_false",
        help="Disable EMA teacher; use current student model for pseudo-labels",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.999,
        metavar="ALPHA",
        help="EMA decay (default: 0.999). Higher = slower teacher update.",
    )
    args = parser.parse_args()

    if args.dataset is None and args.base_config is None:
        args.dataset = "glas"

    modes = list(ABLATION_MODES)
    if args.only:
        modes = [m.strip() for m in args.only.split(",") if m.strip()]
        bad = [m for m in modes if m not in ABLATION_MODES]
        if bad:
            parser.error(f"Unknown mode(s): {bad}. Choose from {ABLATION_MODES}")

    ablations_dir = ROOT / args.ablations_dir
    base_config = ROOT / args.base_config if args.base_config else None
    write_dir = ablations_dir / "_generated"

    runs = resolve_run_configs(
        dataset=args.dataset,
        base_config=base_config,
        modes=modes,
        ablations_dir=ablations_dir,
        write_dir=write_dir,
        use_ema_teacher=args.use_ema_teacher,
        ema_alpha=args.ema_alpha,
    )

    dataset_label = args.dataset
    if dataset_label is None and base_config is not None:
        with open(base_config) as fh:
            dataset_label = str(yaml.safe_load(fh).get("dataset", base_config.stem))

    teacher_label = f"EMA teacher ON  (alpha={args.ema_alpha})" if args.use_ema_teacher else "EMA teacher OFF"
    total = len(runs)
    print(f"Dataset    : {dataset_label}")
    print(f"Teacher    : {teacher_label}")
    print(f"Ablation   : {total} run(s) — {', '.join(m for m, _ in runs)}")
    print(f"Configs    : generated in {write_dir}")
    print()

    results: list[tuple[str, str, str]] = []
    report_rows: list[dict[str, str]] = []

    for idx, (mode, cfg_path) in enumerate(runs, start=1):
        rel_cfg = cfg_path.relative_to(ROOT)
        teacher_str = f"EMA(α={args.ema_alpha})" if args.use_ema_teacher else "none"
        print(f"{'=' * 72}")
        print(f"[{idx:>2}/{total}]  pseudo_ablation={mode}  teacher={teacher_str}")
        print(f"         config={rel_cfg}")
        print(f"{'=' * 72}")

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(rel_cfg),
            "--device",
            args.device,
        ]
        if args.skip_test:
            cmd.append("--skip-test")

        t0 = datetime.now()
        ret = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = str(datetime.now() - t0).split(".")[0]

        status = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
        results.append((mode, status, elapsed))

        with open(cfg_path) as fh:
            run_cfg = yaml.safe_load(fh)
        run_log_path = ROOT / run_cfg["paths"]["log_dir"] / "train.log"
        metrics = parse_run_metrics(run_log_path)
        report_rows.append(
            {
                "mode": mode,
                "teacher": f"EMA(α={args.ema_alpha})" if args.use_ema_teacher else "none",
                "mDice": metrics["mDice"],
                "mJaccard": metrics["mJaccard"],
                "best_val_iou": metrics["best_val_iou"],
                "best_epoch": metrics["best_epoch"],
                "status": status,
                "elapsed": elapsed,
                "config": str(rel_cfg),
            }
        )

        print(f"\n  → {status}  (wall time {elapsed})\n")

    print(f"\n{'=' * 72}")
    print(f"ABLATION COMPLETE [{str(dataset_label).upper()}] — SUMMARY")
    print(f"{'=' * 72}")
    pad = max(len(r[0]) for r in results)
    for mode, status, elapsed in results:
        marker = "✓" if status == "OK" else "✗"
        print(f"  {marker}  {mode:<{pad}}  {elapsed:>8}  {status}")
    print()

    report_dir = ROOT / "outputs" / "reports" / str(dataset_label)
    csv_path, png_path = write_ablation_report(
        str(dataset_label), report_rows, report_dir
    )
    print(f"Report CSV : {csv_path}")
    if png_path is not None:
        print(f"Report PNG : {png_path}")
    else:
        print("Report PNG : skipped (matplotlib not available)")
    print()

    failed = [r for r in results if r[1] != "OK"]
    if failed:
        print(f"{len(failed)} run(s) FAILED:")
        for mode, status, _ in failed:
            print(f"    {mode}  →  {status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
