"""Optional LoRA hook — unused by default; extend if you add PEFT later."""

from typing import Any, Dict


def apply_lora_if_configured(model: Any, cfg: Dict) -> Any:
    return model
