import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.test_runner import run_test_evaluation
from utils.checkpoint import load_yaml_config, resolve_project_paths
from utils.hf_auth import configure_hf_token
from utils.logger import setup_logging
from utils.seed import set_seed

import torch


def main():
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on test set")
    parser.add_argument("--config", type=str, default="configs/crag.yaml")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Checkpoint path; default: paths.checkpoint_dir/seg_crag_uni_conch_best.pt",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)
    project_root = ROOT
    cfg = load_yaml_config(str(project_root / args.config))
    cfg = resolve_project_paths(cfg, project_root)
    configure_hf_token(cfg)
    set_seed(cfg.get("seed", 42))
    dataset = cfg.get("dataset", "crag")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if (args.weights or "").strip():
        weights = Path(args.weights)
    else:
        weights = Path(cfg["paths"]["checkpoint_dir"]) / f"seg_{dataset}_uni_conch_best.pt"
    run_test_evaluation(cfg, device, weights, project_root, log=log)


if __name__ == "__main__":
    main()
