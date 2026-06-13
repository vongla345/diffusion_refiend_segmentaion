import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from evaluation.test_runner import run_test_evaluation
from models.backbone import build_conch_text_encoder
from trainers.trainer import train
from utils.checkpoint import load_yaml_config, resolve_project_paths
from utils.hf_auth import configure_hf_token
from utils.logger import setup_logging
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="CRAG 10%% UNI+CONCH unified training")
    parser.add_argument("--config", type=str, default="configs/crag.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not run full-image test evaluation after training",
    )
    args = parser.parse_args()

    project_root = ROOT

    cfg = load_yaml_config(str(project_root / args.config))
    cfg = resolve_project_paths(cfg, project_root)

    paths = cfg["paths"]
    log_dir = Path(paths.get("log_dir", paths["checkpoint_dir"]))
    log_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = log_dir / "train.log"

    setup_logging(log_file=train_log_path, overwrite_file=True)
    log = logging.getLogger(__name__)
    log.info("=" * 72)
    log.info(
        "Run start %s | config=%s | log_file=%s",
        datetime.now().isoformat(timespec="seconds"),
        args.config,
        train_log_path,
    )
    log.info("=" * 72)

    configure_hf_token(cfg)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    for d in (
        paths["data_root"],
        paths["checkpoint_dir"],
        log_dir,
    ):
        os.makedirs(d, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    text_cfg = cfg.get("text") or {}
    use_text_encoder = bool(text_cfg.get("use_text_encoder", True))

    if use_text_encoder:
        text_enc = build_conch_text_encoder(cfg, device)
        def get_text_emb(batch_size: int):
            return text_enc(batch_size)
        log.info("Text encoder: CONCH (use_text_encoder=true)")
    else:
        def get_text_emb(batch_size: int):
            return None
        log.info("Text encoder: disabled (use_text_encoder=false) — tokens pass straight to decoder")

    train(cfg, device, get_text_emb)
    log.info("Training loop completed at %s", datetime.now().isoformat(timespec="seconds"))
    
    dataset = cfg.get("dataset", "glas")

    if not args.skip_test:
        ckpt_name = f"seg_{dataset}_uni_conch_best.pt"
        ckpt = Path(paths["checkpoint_dir"]) / ckpt_name
        if ckpt.is_file():
            log.info("Running test evaluation on %s", ckpt)
            run_test_evaluation(cfg, device, ckpt, project_root, log=log)
        else:
            log.warning("No %s at %s; skip test evaluation.", ckpt_name, ckpt)


def _release_gpu() -> None:
    """Free GPU memory and tear down DataLoader workers on shutdown."""
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning(
            "Stopped by KeyboardInterrupt (Ctrl+C) — releasing GPU memory."
        )
        _release_gpu()
        # Exit immediately so lingering DataLoader workers don't keep the GPU busy.
        os._exit(130)
    except Exception:
        logging.getLogger(__name__).exception("Training crashed with an exception.")
        _release_gpu()
        raise
