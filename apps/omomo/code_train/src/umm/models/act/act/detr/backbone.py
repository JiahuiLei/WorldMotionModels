# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Backbone modules for visual feature extraction.

Provides ResNet-based backbone networks with frozen batch normalization,
intermediate layer extraction, and positional encoding integration for the
DETR-style ACT model.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from act.utils import NestedTensor, is_main_process
from act.detr.position_encoding import build_position_encoding

import IPython

e = IPython.embed


class FrozenBatchNorm2d(torch.nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.

    Args:
        n: Number of features (channels).

    Example:
        >>> fbn = FrozenBatchNorm2d(64)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = fbn(x)
    """

    def __init__(self, n):
        """Initialize FrozenBatchNorm2d with fixed weight, bias, mean, and variance buffers.

        Args:
            n: Number of features (channels) for the batch normalization.
        """
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Override state dict loading to remove num_batches_tracked key.

        Args:
            state_dict: The state dictionary to load from.
            prefix: The prefix for parameter names in the state dict.
            local_metadata: Local metadata for the module.
            strict: Whether to strictly enforce key matching.
            missing_keys: List to append missing keys to.
            unexpected_keys: List to append unexpected keys to.
            error_msgs: List to append error messages to.
        """
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        """Apply frozen batch normalization to the input tensor.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Normalized tensor of the same shape as input.
        """
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """Base class for backbone feature extractors using IntermediateLayerGetter.

    Wraps a CNN backbone (e.g., ResNet) and extracts features from
    intermediate or final layers.

    Args:
        backbone: The underlying CNN backbone module.
        train_backbone: Whether to allow gradient updates for the backbone.
        num_channels: Number of output channels from the backbone.
        return_interm_layers: If True, return features from all intermediate
            layers (layer1-layer4). Otherwise, return only layer4 features.

    Example:
        >>> import torchvision
        >>> resnet = torchvision.models.resnet18()
        >>> base = BackboneBase(resnet, train_backbone=True, num_channels=512,
        ...                     return_interm_layers=False)
    """

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        """Initialize BackboneBase with an IntermediateLayerGetter wrapper.

        Args:
            backbone: The underlying CNN backbone module (e.g., ResNet).
            train_backbone: Whether to allow gradient updates for the backbone.
            num_channels: Number of output channels from the backbone.
            return_interm_layers: If True, return features from layers 1-4.
                Otherwise, return only the final layer4 features.
        """
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        """Extract features from the backbone.

        Args:
            tensor: Input image tensor of shape (N, C, H, W).

        Returns:
            Dictionary mapping layer names to feature tensors.
        """
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm.

    Constructs a ResNet backbone from torchvision with FrozenBatchNorm2d
    layers and optional dilation and depth channel support.

    Args:
        name: Name of the ResNet variant (e.g., 'resnet18', 'resnet50').
        train_backbone: Whether to train the backbone parameters.
        return_interm_layers: Whether to return intermediate layer features.
        dilation: Whether to replace the last stride with dilation.
        include_depth: Whether to expand the first conv layer to accept an
            additional depth channel (4-channel input instead of 3).

    Example:
        >>> backbone = Backbone('resnet18', train_backbone=True,
        ...                     return_interm_layers=False, dilation=False,
        ...                     include_depth=False)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> features = backbone(x)
    """

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
        include_depth: bool,
    ):
        """Initialize Backbone with a pretrained ResNet and optional depth support.

        Args:
            name: Name of the ResNet variant (e.g., 'resnet18', 'resnet50').
            train_backbone: Whether to train the backbone parameters.
            return_interm_layers: Whether to return intermediate layer features.
            dilation: Whether to replace the last stride with dilation.
            include_depth: Whether to expand conv1 to accept RGBD input.
        """
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d,
        )  # pretrained # TODO do we want frozen batch_norm??

        # for rgbd data
        if include_depth:
            w = backbone.conv1.weight
            w = torch.cat([w, torch.full((64, 1, 7, 7), 0)], dim=1)
            backbone.conv1.weight = nn.Parameter(w)

        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """Joins a backbone with a position embedding module.

    Passes input through the backbone to extract features, then computes
    positional encodings for each feature map.

    Args:
        backbone: The backbone feature extractor module.
        position_embedding: The positional encoding module.

    Example:
        >>> backbone = Backbone('resnet18', True, False, False, False)
        >>> pos_embed = PositionEmbeddingSine(128, normalize=True)
        >>> joiner = Joiner(backbone, pos_embed)
    """

    def __init__(self, backbone, position_embedding):
        """Initialize Joiner with a backbone and position embedding.

        Args:
            backbone: The backbone feature extractor module.
            position_embedding: The positional encoding module.
        """
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        """Extract features and compute positional encodings.

        Args:
            tensor_list: Input image tensor (NestedTensor or raw tensor).

        Returns:
            Tuple of (features_list, positions_list) where features_list is a
            list of feature tensors and positions_list is a list of
            corresponding positional encoding tensors.
        """
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(
    hidden_dim=256,
    position_embedding="sine",
    lr_backbone=1.0e-05,
    masks_flag=False,
    backbone_type="resnet18",
    include_depth=False,
    dilation=False,
):
    """Build a backbone model with positional encoding.

    Constructs a ResNet backbone joined with a positional encoding module
    for use in the DETR-style ACT model.

    Args:
        hidden_dim: Hidden dimension size for the position embedding.
        position_embedding: Type of position embedding ('sine' or 'learned').
        lr_backbone: Learning rate for the backbone. If <= 0, backbone is frozen.
        masks_flag: If True, return intermediate layer features.
        backbone_type: Name of the ResNet variant (e.g., 'resnet18', 'resnet50').
        include_depth: Whether to include depth channel in input.
        dilation: Whether to use dilation in the last ResNet stage.

    Returns:
        A Joiner module combining the backbone and position embedding,
        with a ``num_channels`` attribute indicating the output channel count.

    Example:
        >>> model = build_backbone(hidden_dim=256, backbone_type='resnet18')
        >>> print(model.num_channels)
        512
    """
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    train_backbone = lr_backbone > 0
    return_interm_layers = masks_flag
    backbone = Backbone(
        backbone_type,
        train_backbone,
        return_interm_layers,
        dilation,
        include_depth,
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
