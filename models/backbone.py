import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import conch.open_clip_custom as open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tokenizers import Tokenizer as HFTokenizer

from models.deeplab_decoder import DeepLabV3Decoder


class CONCHTextEncoder(nn.Module):
    """Frozen CONCH text encoder. Caches class embeddings after first forward."""

    def __init__(
        self,
        device: torch.device,
        class_names: Sequence[str],
        prompt_template: str = "a histopathology image of {class_name}",
    ):
        super().__init__()
        self.device = device
        self.class_names = [str(name) for name in class_names]
        if not self.class_names:
            raise ValueError("class_names is empty.")
        if "{class_name}" not in prompt_template:
            raise ValueError("text.prompt_template must contain '{class_name}'.")
        self.prompt_template = str(prompt_template)

        model, _ = open_clip.create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint_path="hf_hub:MahmoodLab/conch",
        )
        self.model = model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self._tokenizer_path = os.path.join(
            os.path.dirname(open_clip.__file__),
            "tokenizers",
            "conch_byte_level_bpe_uncased.json",
        )
        self._cache: Optional[torch.Tensor] = None
        self._tok: Optional[HFTokenizer] = None

    def _tokenizer(self) -> HFTokenizer:
        if self._tok is None:
            self._tok = HFTokenizer.from_file(self._tokenizer_path)
        return self._tok

    @torch.no_grad()
    def _encode(self) -> torch.Tensor:
        """Returns cached class embeddings with shape (n_cls, D)."""
        tok = self._tokenizer()
        pad_id = tok.token_to_id("<pad>")
        ids_list = []
        prompts = [
            self.prompt_template.format(class_name=class_name)
            for class_name in self.class_names
        ]
        for p in prompts:
            enc = tok.encode(p)
            ids = enc.ids[:127]
            ids += [pad_id] * (127 - len(ids))
            ids += [pad_id]
            ids_list.append(ids)
        tokens = torch.tensor(ids_list, dtype=torch.long).to(self.device)
        emb = self.model.encode_text(tokens)
        return F.normalize(emb, dim=-1)

    @property
    def num_text_classes(self) -> int:
        return len(self.class_names)

    def forward(self, batch_size: int) -> torch.Tensor:
        if self._cache is None:
            self._cache = self._encode()
        return self._cache.unsqueeze(0).expand(batch_size, -1, -1)

def build_conch_text_encoder(cfg: Dict[str, Any], device: torch.device) -> CONCHTextEncoder:
    text_cfg = cfg.get("text")
    if not isinstance(text_cfg, dict):
        raise ValueError("Config must define `text:` with `class_names`.")
    names = text_cfg.get("class_names")
    if not isinstance(names, (list, tuple)) or len(names) == 0:
        raise ValueError("Define non-empty `text.class_names`.")
    prompt_template = str(
        text_cfg.get("prompt_template", "a histopathology image of {class_name}")
    )

    return CONCHTextEncoder(
        device=device,
        class_names=[str(n) for n in names],
        prompt_template=prompt_template,
    )

class TextGuidedFusion(nn.Module):
    """Cross-attention: image patch tokens attend to CONCH text embeddings."""

    def __init__(self, img_dim: int, txt_dim: int = 512, heads: int = 4):
        super().__init__()
        self.proj_img = nn.Linear(img_dim, img_dim)
        self.proj_txt = nn.Linear(txt_dim, img_dim)
        self.proj_out = nn.Linear(img_dim, img_dim)
        self.attn = nn.MultiheadAttention(img_dim, heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(img_dim)

    def forward(self, img_feat: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        q = self.proj_img(img_feat)
        k = v = self.proj_txt(txt_emb)
        h, _ = self.attn(q, k, v)
        h = self.norm(q + h)
        return img_feat + self.proj_out(h)


class SegmentationModel(nn.Module):
    """UNI encoder + optional TextGuidedFusion (CONCH) + DeepLabV3 decoder.

    When ``use_text_encoder=False`` the CONCH fusion module is not created and
    the spatial tokens from the image encoder are forwarded straight to the
    decoder, saving ~2 M parameters and removing the CONCH dependency.
    """

    def __init__(
        self,
        txt_emb_dim: int = 512,
        freeze_enc: bool = True,
        aspp_cfg: Optional[Dict[str, Any]] = None,
        use_text_encoder: bool = True,
        use_pyramid_feature: bool = False,
    ):
        super().__init__()
        self.use_text_encoder = use_text_encoder
        self.use_pyramid_feature = use_pyramid_feature
        self.encoder = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
            num_classes=0,
        )
        self.enc_dim = self.encoder.embed_dim
        self.ms_layer = [4, 8, 12]
        self.ms_scale = 2
        self.ms_reduce = nn.Conv2d(self.enc_dim * len(self.ms_layer), self.enc_dim, kernel_size=1)
        if freeze_enc:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if use_text_encoder:
            self.prompt_offset = nn.Parameter(torch.zeros(1, 1, txt_emb_dim))
            self.fusion = TextGuidedFusion(self.enc_dim, txt_emb_dim, heads=4)

        acfg = dict(aspp_cfg or {})
        aspp_out = int(acfg.get("aspp_out", 256))
        rates = acfg.get("rates", (6, 12, 18))
        dil_t = (int(rates[0]), int(rates[1]), int(rates[2]))
        self.decoder = DeepLabV3Decoder(
            in_dim=self.enc_dim,
            aspp_out=aspp_out,
            rates=dil_t,
        )

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, txt_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        grid = h // 16
        
        if self.use_pyramid_feature:
            # using pyramid feature
            target = grid * self.ms_scale
            
            inter = self.encoder.get_intermediate_layers(x, self.ms_layer, reshape=False)
            ms_feats = []
            for feat in inter:
                sp = feat.permute(0, 2, 1).reshape(b, self.enc_dim, grid, grid)
                sp = F.interpolate(sp, size=(target, target), mode="bilinear", align_corners=False)
                ms_feats.append(sp)
            
            spatial = self.ms_reduce(torch.cat(ms_feats, dim=1))
            tokens = spatial.flatten(2).transpose(2, 1)

        
        else:
            # using feature from last layer of uni
            feats = self.encoder.forward_features(x)
            n_patches = (h // 16) * (w // 16)
            tokens = feats[:, -n_patches:, :]
        
        if self.use_text_encoder and txt_emb is not None:
            txt_emb = txt_emb + self.prompt_offset
            fused = self.fusion(tokens, txt_emb)
        else:
            fused = tokens  # skip fusion — tokens go straight to decoder
        grid = int(math.sqrt(tokens.shape[1]))
        
        spatial = fused.permute(0, 2, 1).reshape(b, self.enc_dim, grid, grid)
        logits = self.decoder(spatial)

        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits, tokens


def build_segmentation_model(
    cfg: Dict[str, Any],
    freeze_enc: bool,
    use_pyramid_feature: bool = False,
    grad_checkpointing: bool = False,
) -> SegmentationModel:
    seg = cfg.get("segmentation") or {}
    if grad_checkpointing:
        pass
    aspp_cfg = seg.get("aspp")
    if not isinstance(aspp_cfg, dict):
        aspp_cfg = {}
    text_cfg = cfg.get("text") or {}
    use_text_encoder = bool(text_cfg.get("use_text_encoder", True))
    return SegmentationModel(
        freeze_enc=freeze_enc,
        aspp_cfg=aspp_cfg,
        use_text_encoder=use_text_encoder,
        use_pyramid_feature=use_pyramid_feature,
    )
