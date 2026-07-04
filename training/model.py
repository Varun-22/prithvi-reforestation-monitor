"""
Prithvi-100M change-detection model.

Architecture:
  - Backbone: Prithvi ViT-B/16 encoder (6-band input, patch=16, embed=768)
    loaded from ibm-nasa-geospatial/Prithvi-EO-1.0-100M on HuggingFace.
    Backbone is frozen during fine-tuning.
  - Head: lightweight CNN decoder on |after − before| feature difference.
    14×14 → 28 → 56 → 112 → 224 via bilinear upsample + Conv.
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("pip install huggingface_hub")

PRITHVI_REPO     = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
PRITHVI_FILENAME = "Prithvi_100M.pt"


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class PrithviEncoder(nn.Module):
    """Prithvi-100M ViT-B backbone for a single time-point."""

    PATCH_GRID = 14  # 224 // 16

    def __init__(self, img_size: int = 224, in_chans: int = 6, embed_dim: int = 768):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            in_chans=in_chans,
            num_classes=0,       # remove classification head
            img_size=img_size,
            pretrained=False,
        )
        self._frozen = False

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_prithvi_weights(self, ckpt_path: str) -> None:
        """Load Prithvi encoder weights; handle temporal pos_embed mismatch."""
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw.get("model", raw)  # MAE checkpoint may wrap under 'model'

        # Strip known prefixes, drop decoder weights
        cleaned: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if "decoder" in k:
                continue
            for prefix in ("encoder.", "model.", "module."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v

        # pos_embed: Prithvi trained with T=3 frames → (1, 3*196+1, 768).
        # timm ViT expects (1, 197, 768).  Take CLS + first 196 spatial tokens.
        expected_pe_shape = self.backbone.pos_embed.shape  # (1, 197, 768)
        if "pos_embed" in cleaned and cleaned["pos_embed"].shape != expected_pe_shape:
            pe = cleaned["pos_embed"]
            cls_pe     = pe[:, :1, :]
            spatial_pe = pe[:, 1:, :][:, : expected_pe_shape[1] - 1, :]
            cleaned["pos_embed"] = torch.cat([cls_pe, spatial_pe], dim=1)
            print(f"  pos_embed: trimmed to {cleaned['pos_embed'].shape}")

        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        loaded = len(cleaned) - len(missing)
        print(f"  Weights loaded: {loaded}/{len(cleaned)}  "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    @classmethod
    def from_pretrained(cls, cache_dir: str | None = None) -> "PrithviEncoder":
        enc = cls()
        print(f"Downloading {PRITHVI_FILENAME} from {PRITHVI_REPO}...")
        path = hf_hub_download(
            repo_id=PRITHVI_REPO,
            filename=PRITHVI_FILENAME,
            cache_dir=cache_dir,
        )
        print(f"Loading encoder weights from {path}")
        enc.load_prithvi_weights(path)
        return enc

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self._frozen = True

    def unfreeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        self._frozen = False

    # Keep backbone in eval mode when the outer model is set to train
    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._frozen:
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 6, 224, 224) normalised tile
        out: (B, 768, 14, 14) spatial feature map
        """
        feats = self.backbone.forward_features(x)   # (B, 197, 768)
        patch = feats[:, 1:, :]                      # drop CLS → (B, 196, 768)
        B, _, D = patch.shape
        return patch.transpose(1, 2).reshape(B, D, self.PATCH_GRID, self.PATCH_GRID)


# ---------------------------------------------------------------------------
# Decoder head
# ---------------------------------------------------------------------------

class ChangeDetectionHead(nn.Module):
    """Bilinear-upsample CNN decoder: (B, 768, 14, 14) → (B, 1, 224, 224)."""

    def __init__(self, in_channels: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 256, 1)
        self.up1  = self._up_block(256, 128)   # 14  → 28
        self.up2  = self._up_block(128,  64)   # 28  → 56
        self.up3  = self._up_block( 64,  32)   # 56  → 112
        self.up4  = self._up_block( 32,  16)   # 112 → 224
        self.out  = nn.Conv2d(16, 1, 1)        # logit

    @staticmethod
    def _up_block(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.up1(x); x = self.up2(x)
        x = self.up3(x); x = self.up4(x)
        return self.out(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PrithviChangeDetector(nn.Module):
    """
    End-to-end change-detection model.

    Both time points are encoded independently by the frozen Prithvi backbone;
    the absolute feature difference is decoded by the trainable head.
    """

    def __init__(
        self,
        pretrained:      bool = True,
        freeze_backbone: bool = True,
        cache_dir:       str | None = None,
    ):
        super().__init__()
        self.encoder = (
            PrithviEncoder.from_pretrained(cache_dir=cache_dir)
            if pretrained else PrithviEncoder()
        )
        if freeze_backbone:
            self.encoder.freeze()
        self.head = ChangeDetectionHead()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen backbone in eval even when model is set to train
        if mode and self.encoder._frozen:
            self.encoder.backbone.eval()
        return self

    def forward(
        self,
        before: torch.Tensor,   # (B, 6, 224, 224)
        after:  torch.Tensor,   # (B, 6, 224, 224)
    ) -> torch.Tensor:          # (B, 1, 224, 224) logits
        f_b = self.encoder(before)
        f_a = self.encoder(after)
        return self.head(torch.abs(f_a - f_b))

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
