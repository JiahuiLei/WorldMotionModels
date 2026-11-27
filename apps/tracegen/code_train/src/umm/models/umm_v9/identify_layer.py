"""Dual-path input layer: PointNet for robot links, DINO pixel query for objects.

Extends IdentifyLayerV2 with an ``external_geo_emb`` path so that parts
without real point clouds (objects) can receive geometry embeddings from
the DINO spatial feature map, while parts with real point clouds (robot
links) still use the PointNet encoder.

Example::

    layer = IdentifyLayerV2Dual(latent_dim=512, feat_dim=512,
                                 point_encoder_latent=256)
    feat = layer(sem_id, group_id, pcl, pcl_valid_mask,
                 external_geo_emb=dino_features)
"""

import logging

import torch
import torch.nn as nn

from umm.models.umm_v9.world_models.moma_v2 import IdentifyLayerV2
from umm.models.umm_v9.world_models.wm_utils import HardLayerNorm


class IdentifyLayerV2Dual(IdentifyLayerV2):
    """Input layer with dual geo_emb: PointNet for robot links, DINO for objects.

    For parts where pcl_valid_mask is True (robot links):
        geo_emb = PointNet(local_pcl)
    For parts where pcl_valid_mask is False (object pixel queries):
        geo_emb = external_geo_emb (from DINO spatial feature lookup)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geo_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.dino_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.ln_local_feat = HardLayerNorm(self.latent_dim)

    @torch.compiler.disable
    def forward(
        self, sem_id, group_id, pcl=None, pcl_valid_mask=None, external_geo_emb=None
    ):
        """Forward pass with dual geometry embedding paths.

        Args:
            sem_id: (B, P) semantic type IDs.
            group_id: (B, P) group IDs.
            pcl: (B, P, M, D) local point clouds (only valid for pcl_valid_mask=True).
            pcl_valid_mask: (B, P) bool, True for parts with real PCL (robot links).
            external_geo_emb: (B, P, latent_dim) DINO features for pixel query parts.

        Returns:
            feat: (B, P, feat_dim)
        """
        B, N = sem_id.shape
        type_emb = self.type_emb(sem_id)  # (B, P, D)
        if self.skip_group_emb:
            group_emb = torch.zeros(
                B, N, self.latent_dim, device=sem_id.device, dtype=type_emb.dtype
            )
        else:
            group_emb = self.group_emb(group_id)  # (B, P, D)

        # Initialize geo_emb with empty embedding
        geo_emb = (
            self.empty_point_emb(torch.tensor([0], device=sem_id.device))[None]
            .repeat(B, N, 1)
            .contiguous()
        )  # (B, P, D)

        # PointNet path for parts with valid PCL (robot links)
        if self.point_flag and pcl is not None and pcl_valid_mask is not None:
            if pcl_valid_mask.any():
                valid_pcl = pcl[pcl_valid_mask].clone()  # (K, M, D_pcl)
                valid_pcl[..., :3] = valid_pcl[..., :3] * self.point_xyz_scale_factor
                valid_pcl_feat = self.geo_proj(
                    self.pntnet(valid_pcl)
                )  # (K, latent_dim)
                geo_emb[pcl_valid_mask] = (
                    geo_emb[pcl_valid_mask].clone() * 0.0 + valid_pcl_feat
                )

        # DINO path for parts with pixel queries (objects without real PCL)
        if external_geo_emb is not None and pcl_valid_mask is not None:
            dino_mask = ~pcl_valid_mask  # parts WITHOUT valid PCL use DINO
            if dino_mask.any():
                geo_emb[dino_mask] = geo_emb[dino_mask].clone() * 0.0 + self.dino_proj(
                    external_geo_emb[dino_mask]
                )

        type_emb = self.ln_type_emb(type_emb)  # (B, P, D)
        group_emb = self.ln_group_emb(group_emb)  # (B, P, D)
        geo_emb = self.ln_local_feat(geo_emb)  # (B, P, D)

        feat = torch.cat([type_emb, group_emb, geo_emb], dim=-1)  # (B, P, 3*D)
        feat = self.bind_fc(feat)  # (B, P, feat_dim)

        return feat
