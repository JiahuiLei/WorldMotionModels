"""Frozen DINOv3 image encoder with view position embeddings.

Encodes multi-view RGB images into patch tokens projected to the model's
latent dimension, with learnable per-view-type position embeddings.

Uses DINOv3 (ViT-S/16 by default) loaded via the facebookresearch/dinov3
repo (torch.hub local source). Compile-safe: no ``@torch.compiler.disable``
needed — ``forward_features()`` traces cleanly through ``torch.compile``.

Example::

    enc = DINOImageEncoder(model_name="dinov3_vits16", latent_dim=512)
    tokens, mask = enc(images, view_ids)
    # tokens: (B, V*num_patches, 512)  — 196 patches for 224x224
"""

import logging
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# DINOv3 repo location (cloned via git)
_DINOV3_REPO = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")

# Local weights directory (downloaded via Meta access form)
_DINOV3_CKPT_DIR = os.path.expanduser("~/.cache/torch/hub/checkpoints")

# Model name → (hub function name, weights filename, embed_dim)
_DINOV3_REGISTRY = {
    "dinov3_vits16": (
        "dinov3_vits16",
        "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        384,
    ),
    "dinov3_vits16plus": (
        "dinov3_vits16plus",
        "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        384,
    ),
    "dinov3_vitb16": (
        "dinov3_vitb16",
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        768,
    ),
    "dinov3_vitl16": (
        "dinov3_vitl16",
        "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        1024,
    ),
}


def _load_dinov3_backbone(
    model_name: str,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """Load a DINOv3 backbone from local repo + local weights.

    Args:
        model_name: Key in ``_DINOV3_REGISTRY`` (e.g. ``"dinov3_vits16"``).
        weights_path: Explicit path to ``.pth`` file. If None, looks up
            the default filename in ``_DINOV3_CKPT_DIR``.

    Returns:
        ``DinoVisionTransformer`` instance with loaded weights.
    """
    if model_name not in _DINOV3_REGISTRY:
        raise ValueError(
            f"Unknown DINOv3 model: {model_name}. "
            f"Available: {list(_DINOV3_REGISTRY.keys())}"
        )
    hub_fn_name, default_ckpt, _ = _DINOV3_REGISTRY[model_name]

    if weights_path is None:
        weights_path = os.path.join(_DINOV3_CKPT_DIR, default_ckpt)

    if not os.path.isdir(_DINOV3_REPO):
        raise FileNotFoundError(
            f"DINOv3 repo not found at {_DINOV3_REPO}. "
            "Clone it: git clone --depth 1 https://github.com/facebookresearch/dinov3.git "
            f"{_DINOV3_REPO}"
        )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"DINOv3 weights not found at {weights_path}. "
            "Download from Meta: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"
        )

    # Add repo to sys.path so dinov3.hub.backbones can be imported
    if _DINOV3_REPO not in sys.path:
        sys.path.insert(0, _DINOV3_REPO)

    from dinov3.hub.backbones import (  # noqa: E402
        dinov3_vitb16,
        dinov3_vitl16,
        dinov3_vits16,
        dinov3_vits16plus,
    )

    _fn_map = {
        "dinov3_vits16": dinov3_vits16,
        "dinov3_vits16plus": dinov3_vits16plus,
        "dinov3_vitb16": dinov3_vitb16,
        "dinov3_vitl16": dinov3_vitl16,
    }

    model = _fn_map[hub_fn_name](pretrained=True, weights=weights_path)
    return model


class DINOImageEncoder(nn.Module):
    """Frozen DINOv3 encoder for RGB images with view position embeddings.

    Loads a DINOv3 model (ViT-S/16 by default) from the local
    ``facebookresearch/dinov3`` repo, freezes it, and projects patch tokens
    to the model's latent dimension. Adds learnable per-view-type
    embeddings so the model can distinguish head, wrist, etc.

    **Compile-safe**: uses ``forward_features()`` (not
    ``get_intermediate_layers()``) so the full encoder traces cleanly
    through ``torch.compile`` without graph breaks.

    Args:
        model_name: DINOv3 model key (e.g. ``"dinov3_vits16"``).
        latent_dim: Output dimension for projected tokens.
        num_views: Maximum number of distinct view types.
        trainable: If False, freeze the DINOv3 backbone weights.
        weights_path: Explicit path to ``.pth`` weights. If None, uses
            the default path in ``~/.cache/torch/hub/checkpoints/``.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        latent_dim: int = 512,
        num_views: int = 8,
        trainable: bool = False,
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.trainable = trainable

        # Load DINOv3 backbone
        logging.info(f"Loading DINOv3 model: {model_name}")
        self.encoder = _load_dinov3_backbone(model_name, weights_path)
        self.patch_size = self.encoder.patch_size  # 16
        self.embed_dim = self.encoder.embed_dim  # 384 for vits16, 768 for vitb16
        logging.info(
            f"DINOv3 loaded: patch_size={self.patch_size}, embed_dim={self.embed_dim}"
        )

        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # Project from DINO embed_dim to model latent_dim
        self.proj = nn.Linear(self.embed_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

        # Learnable view position embedding (one per view type)
        self.view_emb = nn.Embedding(num_views, latent_dim)

    def _extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Run frozen DINOv3 backbone and return patch tokens.

        Uses ``forward_features()`` which is compile-safe (no dynamic list
        building like ``get_intermediate_layers``).

        Args:
            images: (N, 3, H, W) float32, ImageNet-normalized.

        Returns:
            patch_tokens: (N, num_patches, embed_dim)
                196 patches for 224x224 with patch_size=16.
        """
        with torch.no_grad():
            feat = self.encoder.forward_features(images)
            return feat["x_norm_patchtokens"]  # (N, P, embed_dim)

    def forward(
        self, images: torch.Tensor, view_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode multi-view images into projected patch tokens.

        Args:
            images: (B, V, 3, H, W) float32, ImageNet-normalized RGB.
            view_ids: (B, V) int, view type indices
                (0=head, 1=left_wrist, 2=right_wrist, etc.).

        Returns:
            tokens: (B, V*num_patches, latent_dim) projected DINO patch tokens.
            tokens_mask: (B, V*num_patches) bool, all True (no masking).
        """
        B, V, C, H, W = images.shape  # (B, V, 3, H, W)

        # Encode all views in one batch
        flat_images = images.reshape(B * V, C, H, W)  # (B*V, 3, H, W)
        flat_tokens = self._extract_patch_tokens(flat_images)
        # flat_tokens: (B*V, num_patches, embed_dim)

        # Project to latent_dim
        flat_tokens = self.proj(flat_tokens.float())  # (B*V, num_patches, latent_dim)
        flat_tokens = self.ln(flat_tokens)

        num_patches = flat_tokens.shape[1]  # 196 for 224x224 with patch_size=16
        tokens = flat_tokens.reshape(B, V, num_patches, -1)  # (B, V, P, D)

        # Add view position embeddings
        view_embs = self.view_emb(view_ids)  # (B, V, D)
        tokens = (
            tokens + view_embs[:, :, None, :]
        )  # broadcast over patches: (B, V, P, D)

        # Flatten views into token sequence
        tokens = tokens.reshape(B, V * num_patches, -1)  # (B, V*P, D)
        tokens_mask = torch.ones(
            B, V * num_patches, device=tokens.device, dtype=torch.bool
        )  # (B, V*P)

        return tokens, tokens_mask

    def encode_spatial(
        self, images: torch.Tensor, view_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images and return spatial feature maps.

        Args:
            images: (B, V, 3, H, W) float32, ImageNet-normalized RGB.
            view_ids: (B, V) int, view type indices.

        Returns:
            feature_maps: (B, V, grid_h, grid_w, latent_dim)
            flat_tokens: (B, V*grid_h*grid_w, latent_dim) -- for add_tokens
            flat_mask: (B, V*grid_h*grid_w) bool
        """
        B, V, C, H, W = images.shape  # (B, V, 3, H, W)

        flat_images = images.reshape(B * V, C, H, W)  # (B*V, 3, H, W)
        flat_tokens = self._extract_patch_tokens(flat_images)
        # flat_tokens: (B*V, num_patches, embed_dim)

        flat_tokens = self.proj(flat_tokens.float())  # (B*V, num_patches, latent_dim)
        flat_tokens = self.ln(flat_tokens)

        num_patches = flat_tokens.shape[1]  # 196 for 224x224 with patch_size=16
        grid_h = grid_w = int(num_patches**0.5)  # 14 for 224px with patch_size=16

        tokens = flat_tokens.reshape(B, V, grid_h, grid_w, -1)  # (B, V, GH, GW, D)

        # Add view position embeddings
        view_embs = self.view_emb(view_ids)  # (B, V, D)
        tokens = tokens + view_embs[:, :, None, None, :]  # broadcast over spatial dims

        # Flat tokens for add_tokens (cross-attention)
        flat_tk = tokens.reshape(B, V * grid_h * grid_w, -1)  # (B, V*GH*GW, D)
        flat_mask = torch.ones(
            B, V * grid_h * grid_w, device=tokens.device, dtype=torch.bool
        )  # (B, V*GH*GW)

        return tokens, flat_tk, flat_mask

    def query_at_pixels(
        self,
        feature_maps: torch.Tensor,
        pixel_uv_normalized: torch.Tensor,
        pixel_view_idx: torch.Tensor,
        pixel_query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Query DINO features at specific pixel UV locations.

        Uses F.grid_sample for efficient bilinear interpolation.

        Args:
            feature_maps: (B, V, grid_h, grid_w, latent_dim) from encode_spatial
            pixel_uv_normalized: (B, P, 2) UV in [0, 1], (u_norm, v_norm)
            pixel_view_idx: (B, P) int, which view for each query
            pixel_query_mask: (B, P) bool, True for pixel queries

        Returns:
            pixel_features: (B, P, latent_dim) -- features at queried locations.
                Zero for parts where pixel_query_mask is False.
        """
        B, V, GH, GW, D = feature_maps.shape
        P = pixel_uv_normalized.shape[1]
        device = feature_maps.device

        # Reshape feature maps to (B*V, D, GH, GW) for grid_sample
        feat_2d = feature_maps.permute(0, 1, 4, 2, 3).reshape(
            B * V, D, GH, GW
        )  # (B*V, D, GH, GW)

        # Compute flat indices into B*V dimension
        batch_offsets = torch.arange(B, device=device)[:, None] * V  # (B, 1)
        flat_view_idx = (batch_offsets + pixel_view_idx).reshape(-1)  # (B*P,)

        # Select the right view's feature map for each query
        selected_maps = feat_2d[flat_view_idx]  # (B*P, D, GH, GW)

        # Convert normalized UV [0,1] to grid_sample coords [-1, 1]
        # pixel_uv_normalized[:, :, 0] = u (horizontal), [:, :, 1] = v (vertical)
        grid_uv = pixel_uv_normalized * 2.0 - 1.0  # (B, P, 2) in [-1, 1]
        grid = grid_uv.reshape(B * P, 1, 1, 2)  # (B*P, 1, 1, 2)

        # Bilinear sample
        sampled = F.grid_sample(
            selected_maps,
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )  # (B*P, D, 1, 1)

        pixel_features = sampled.squeeze(-1).squeeze(-1).reshape(B, P, D)  # (B, P, D)

        # Zero out non-query parts
        pixel_features = pixel_features * pixel_query_mask[:, :, None].float()

        return pixel_features

    def train(self, mode: bool = True):
        """Override train to keep frozen DINO encoder in eval mode."""
        super().train(mode)
        if not self.trainable:
            self.encoder.eval()
        return self

    def get_optim_groups(
        self, weight_decay: float = 0.0, base_lr: Optional[float] = None
    ):
        """Build optimizer parameter groups (excluding frozen DINO backbone).

        Args:
            weight_decay: Weight decay for trainable parameters.
            base_lr: If provided, sets ``lr`` on each group.

        Returns:
            list[dict]: Parameter groups for ``torch.optim`` optimizers.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        if base_lr is not None:
            for group in optim_groups:
                group["lr"] = base_lr
        return optim_groups
