"""AgentE10Hybrid: SIM(3)-Equivariant Hybrid Trajectory Diffusion Forcing Agent.

Variant B: Uses hybrid equivariant->invariant->equivariant architecture.
Shares canonicalization, diffusion, loss, and inference logic with AgentE10.
Only the backbone differs: VNHybridTransformer instead of VNTransformer.
"""

import logging

from umm.models.umm_e10.agent import AgentE10
from umm.models.umm_e10.vn_hybrid_transformer import VNHybridTransformer


class AgentE10Hybrid(AgentE10):
    """Variant B: hybrid equivariant-invariant transformer.

    Inherits everything from AgentE10, only replaces self.vn_transformer
    with VNHybridTransformer.
    """

    def __init__(self, cfg):
        # Initialize parent fully (creates VNTransformer + all shared components)
        super().__init__(cfg)

        # Replace VNTransformer with VNHybridTransformer
        vn_cfg = self.model_cfg.get("vn_transformer", {})
        self.vn_transformer = VNHybridTransformer(
            d_vn=vn_cfg.get("d_vn", 64),
            n_heads=vn_cfg.get("n_heads", 8),
            n_layers=vn_cfg.get("n_layers", 6),
            ffn_factor=vn_cfg.get("ffn_factor", 4),
            input_vn_channels=3,
            output_vn_channels=3,
            num_types=vn_cfg.get("num_types", 1024),
            num_groups=vn_cfg.get("num_groups", 1024),
            max_noise_steps=self.noise_scheduler.config.num_train_timesteps,
            d_cond=vn_cfg.get("d_cond", 256),
            qk_norm=vn_cfg.get("qk_norm", True),
            time_multiplier=vn_cfg.get("time_multiplier", 10.0),
            n_basis_dots=vn_cfg.get("n_basis_dots", 8),
        )

        logging.info("AgentE10Hybrid initialized (hybrid equivariant-invariant transformer)")
