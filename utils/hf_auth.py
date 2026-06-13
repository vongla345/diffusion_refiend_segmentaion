import logging
import os
from typing import Dict


logger = logging.getLogger(__name__)


def configure_hf_token(cfg: Dict) -> None:
    """
    Configure Hugging Face token from config.
    Priority:
    1) hf.token in yaml
    2) existing env vars
    """
    hf_cfg = cfg.get("hf", {}) or {}
    token = (hf_cfg.get("token") or "").strip()
    if token.startswith("${") and token.endswith("}"):
        logger.warning(
            "HF token placeholder %s was not resolved (missing .env entry?).", token
        )
        token = ""
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        masked = f"{token[:5]}…{token[-2:]}" if len(token) > 8 else "set"
        logger.info("HF token loaded from config (%s).", masked)
    else:
        if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"):
            logger.info("HF token loaded from environment.")
        else:
            logger.warning("No HF token configured (yaml/env). Model download may fail.")
