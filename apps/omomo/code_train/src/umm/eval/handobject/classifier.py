"""Hand-object-adapted LSTM action classifier.

Adapts the IMoS SMPL-X body classifier (22 joints) to the hand-object setting
(43 points: 21 left-hand + 21 right-hand joints + 1 object-translation center).
Two-stream LSTM on XY and XZ 2D projections, sum-fused via Conv1d, followed by
a linear head to action logits. The penultimate tanh(y3)*0.1 activation
exposes the 128-dim motion feature consumed by FID/Diversity/Multimodality
metrics (IMoS action_rec.py:63).

Key differences vs IMoS LSTM_Action_Classifier (action_rec.py:14-82):
    - 43 points per frame (21L + 21R + 1 obj) instead of 22 SMPL-X body joints
      -> `joints_dim2d = 86` instead of 44.
    - `predict`/`forward` take a `(B, T, 43, 3)` tensor plus per-sample `lengths`
      instead of IMoS's flat `(T*B, 22, 3)` layout with baked-in batch_size.
    - Uses last-valid-frame gather via `lengths - 1` for variable-length handling
      (IMoS runs fixed-length with `batch_size=1`).
    - Multi-class softmax+CE (returns logits, loss in train loop) instead of
      IMoS's multilabel sigmoid+BCE — standard convention for single-verb
      per-episode classification (same adaptation is made by the Text2HOI
      evaluation pipeline).

Every forward call is O(T) LSTM; the classifier is intentionally small
(~256K params) to train fast on a single GPU.

References:
    - IMoS: /tmp/IMoS/src/models/action_rec.py lines 14-82.
    - Canonical schema: umm/eval/handobject/canonical.py.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class HandObjectActionClassifier(nn.Module):
    """Hand-object-adapted LSTM action classifier (IMoS action_rec.py:14-82).

    Architecture:
        points (B, T, 43, 3)
          split --> XY (B, T, 43, 2) --flatten--> (T, B, 86) --LSTM--> (T, B, 128)
                                                               last-valid-frame --> (B, 128)
                                                               Conv1d(1->1, k=3) --> (B, 128)
          split --> XZ (B, T, 43, 2) --flatten--> (T, B, 86) --LSTM--> (T, B, 128)
                                                               last-valid-frame --> (B, 128)
                                                               Conv1d(1->1, k=3) --> (B, 128)
          sum --> (B, 128) --tanh * 0.1--> features (B, 128)
                       |
                       Linear(128 -> n_labels) --> logits (B, n_labels)

    IMoS's `predict` returned (sigmoid(y4), tanh(y3)*0.1). We return logits
    (pre-softmax) + features; the training loop applies `CrossEntropyLoss`
    which fuses log-softmax+NLL.
    """

    def __init__(
        self,
        n_points: int = 43,
        hidden_dim: int = 128,
        n_labels: int = 29,
        num_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        """Initialize the classifier.

        Args:
            n_points:   Points per frame. Default 43 = 21L + 21R + 1 obj.
            hidden_dim: LSTM hidden + Conv channel. Default 128 (per IMoS).
            n_labels:   Number of output classes (29 GRAB / 11 ARCTIC).
            num_layers: LSTM stack depth. Default 2 (per IMoS).
            kernel_size: Conv1d kernel size over the hidden dimension. Default 3 (per IMoS).
        """
        super().__init__()
        if n_points <= 0 or hidden_dim <= 0 or n_labels <= 0 or num_layers <= 0:
            raise ValueError(
                f"HandObjectActionClassifier: all sizes must be positive; "
                f"got n_points={n_points}, hidden_dim={hidden_dim}, "
                f"n_labels={n_labels}, num_layers={num_layers}"
            )
        if kernel_size % 2 == 0:
            raise ValueError(
                f"HandObjectActionClassifier: kernel_size must be odd (got {kernel_size}) "
                f"— required for 'same' padding via pad=(k-1)/2."
            )

        self.n_points = n_points
        self.hidden_dim = hidden_dim
        self.n_labels = n_labels
        self.num_layers = num_layers

        # 2D projection width: each of the 43 points contributes 2 coordinates -> 86.
        # IMoS action_rec.py:22,24 uses `joints_dim=44` (22 body joints * 2).
        self.proj_dim: int = n_points * 2                                       # 86 by default

        # Two parallel LSTMs — XY stream and XZ stream. batch_first=False
        # matches IMoS's (T, B, feat) convention (action_rec.py:53-54).
        self.lstm_xy = nn.LSTM(self.proj_dim, hidden_dim, num_layers=num_layers)  # out (T, B, H)
        self.lstm_xz = nn.LSTM(self.proj_dim, hidden_dim, num_layers=num_layers)  # out (T, B, H)

        # Post-LSTM Conv1d applied to the last-valid-frame vector reshaped to
        # (B, 1, H). IMoS action_rec.py:23,25: Conv1d(1, 1, k, stride=1, padding=1).
        pad = (kernel_size - 1) // 2
        self.conv_xy = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=pad)  # (B, 1, H) -> (B, 1, H)
        self.conv_xz = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=pad)  # (B, 1, H) -> (B, 1, H)

        # Linear head from fused features to logits (IMoS action_rec.py:30).
        self.head = nn.Linear(hidden_dim, n_labels)                             # (B, H) -> (B, n_labels)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        points: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample logits and 128-dim motion features.

        Args:
            points:  (B, T, n_points, 3) float32. Padded with zeros after the last valid frame.
            lengths: (B,) int64. 1 <= lengths[b] <= T, actual valid frames per sample.

        Returns:
            Dict with:
                logits:   (B, n_labels) float32 — pre-softmax action scores.
                features: (B, hidden_dim) float32 — tanh(y3)*0.1 penultimate features,
                          for FID / Diversity / Multimodality metrics.

        Raises:
            ValueError: On any shape/dtype mismatch.
        """
        if points.ndim != 4 or points.shape[-1] != 3 or points.shape[-2] != self.n_points:
            raise ValueError(
                f"forward: points expected (B, T, {self.n_points}, 3), got {tuple(points.shape)}"
            )
        if lengths.ndim != 1 or lengths.shape[0] != points.shape[0]:
            raise ValueError(
                f"forward: lengths expected (B={points.shape[0]},), got {tuple(lengths.shape)}"
            )
        if lengths.dtype not in (torch.int64, torch.int32):
            raise ValueError(
                f"forward: lengths dtype must be int32/int64, got {lengths.dtype}"
            )

        B, T, P, _ = points.shape                                               # B, T, 43, 3
        H = self.hidden_dim

        # Bounds check — all samples must have at least 1 valid frame, at most T.
        if int(lengths.min().item()) < 1:
            raise ValueError(
                f"forward: minimum length must be >=1, got min={int(lengths.min().item())}"
            )
        if int(lengths.max().item()) > T:
            raise ValueError(
                f"forward: max length ({int(lengths.max().item())}) exceeds T ({T})"
            )

        # 1) Two 2D projections.
        #    XY: take x, y -> coords [..., 0:2].
        #    XZ: take x, z -> coords [..., [0, 2]] (IMoS action_rec.py:52).
        xy = points[..., 0:2]                                                   # (B, T, 43, 2)
        xz = points[..., [0, 2]]                                                # (B, T, 43, 2)

        # Flatten point+coord axes -> feature vector of length 86.
        xy_flat = xy.reshape(B, T, self.proj_dim)                               # (B, T, 86)
        xz_flat = xz.reshape(B, T, self.proj_dim)                               # (B, T, 86)

        # 2) LSTM expects (T, B, feat) under batch_first=False (IMoS default).
        xy_tbc = xy_flat.transpose(0, 1).contiguous()                           # (T, B, 86)
        xz_tbc = xz_flat.transpose(0, 1).contiguous()                           # (T, B, 86)

        lstm_xy_out, _ = self.lstm_xy(xy_tbc)                                   # (T, B, H)
        lstm_xz_out, _ = self.lstm_xz(xz_tbc)                                   # (T, B, H)

        # 3) Last-valid-frame gather per sample. IMoS uses fixed-length
        #    `lstm_out[-1]` (action_rec.py:55-56); we gather the per-sample
        #    last valid time index from `lengths - 1` for variable lengths.
        #    Move back to batch-first for gather convenience.
        lstm_xy_btc = lstm_xy_out.transpose(0, 1).contiguous()                  # (B, T, H)
        lstm_xz_btc = lstm_xz_out.transpose(0, 1).contiguous()                  # (B, T, H)

        last_idx = (lengths - 1).to(torch.int64)                                # (B,)
        # Indexing: gather across dim=1 (time). Expand to (B, 1, H) for gather.
        gather_idx = last_idx.view(B, 1, 1).expand(B, 1, H)                     # (B, 1, H)
        t_xy = lstm_xy_btc.gather(dim=1, index=gather_idx).squeeze(1)           # (B, H)
        t_xz = lstm_xz_btc.gather(dim=1, index=gather_idx).squeeze(1)           # (B, H)

        # 4) Conv1d on (B, 1, H) — IMoS reshapes `lstm_out[-1].view(batch, 1, -1)`.
        y_xy = self.conv_xy(t_xy.unsqueeze(1)).squeeze(1)                       # (B, H)
        y_xz = self.conv_xz(t_xz.unsqueeze(1)).squeeze(1)                       # (B, H)

        # 5) Sum-fuse the two streams. Penultimate feature = tanh(y3) * 0.1
        #    (IMoS action_rec.py:63).
        y3 = y_xy + y_xz                                                        # (B, H)
        features = torch.tanh(y3) * 0.1                                         # (B, H)

        # 6) Linear head -> logits.
        logits = self.head(y3)                                                  # (B, n_labels)

        return {"logits": logits, "features": features}

    # ------------------------------------------------------------------
    # Predict alias
    # ------------------------------------------------------------------

    def predict(
        self,
        points: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Alias for `forward` — retained for IMoS API compatibility (action_rec.py:48).

        Identical semantics to `forward` (no training-only branches exist here).
        """
        return self.forward(points, lengths)


class Text2HOIActionClassifier(nn.Module):
    """IMoS-adapted 2-stream action classifier for Text2HOI hand-object input.

    **Faithful port of IMoS `LSTM_Action_Classifier` (action_rec.py:14-82) with
    joints_dim set for 42 hand joints (21 left + 21 right), fed from Text2HOI's
    `get_action_classifier_input` (proc_output.py:77-104) at seq_len=15.**

    Architecture (per /tmp/IMoS/src/models/action_rec.py:14-82, with joints_dim
    adjusted from 44=22*2 to 84=42*2):

        points (B, 15, 42, 3)
          -> x = view(-1, 42, 3)                           # (B*15, 42, 3)
          -> x_xy = x[..., 0:2].contiguous().view(-1, 1, 84)        # XY stream
          -> x_xz = x[..., [0, 2]].contiguous().view(-1, 1, 84)     # XZ stream
          -> lstm_xy = LSTM(84, 128, num_layers=2, batch_first=False)
          -> lstm_xz = LSTM(84, 128, num_layers=2, batch_first=False)
          -> out_xy = lstm_xy(x_xy)[0][-1]                 # last-time output
          -> out_xz = lstm_xz(x_xz)[0][-1]
          -> y2_2 = conv_xy(out_xy.unsqueeze(1))           # Conv1d(1, 1, k=3, pad=1)
          -> y2_3 = conv_xz(out_xz.unsqueeze(1))
          -> y3 = y2_2 + y2_3                              # sum-fuse (1, 1, 128)
          -> features = tanh(y3) * 0.1                      # (B, 128) penultimate
          -> logits = Linear(128, n_labels)(y3)             # pre-sigmoid
          -> (training) sigmoid + BCE(multilabel one-hot)

    Notes on the batch layout:
      - IMoS forces batch_size=1 at training (train_action_recognition.py:50).
        We support arbitrary B via a (T, B, feat) transpose before LSTM while
        keeping last-timestep extraction, Conv1d(1→1), sum-fuse, tanh*0.1, and
        Linear head bit-identical to IMoS.
      - NO dropout (IMoS arch has none).
      - batch_first=False LSTM (IMoS default).

    Output dict: `{"logits": (B, n_labels), "features": (B, 128)}`.
    The training loop applies `sigmoid` + `F.binary_cross_entropy` on `logits`
    against a one-hot multi-label target to match IMoS (action_rec.py:82).

    References:
      - IMoS `LSTM_Action_Classifier`: /tmp/IMoS/src/models/action_rec.py:14-82
      - Text2HOI `get_action_classifier_input`: /tmp/Text2HOI/lib/utils/proc_output.py:77-104
    """

    def __init__(
        self,
        n_labels: int,
        num_frames: int = 15,
        joints_per_frame: int = 42,
        feat_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if n_labels <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError(
                f"Text2HOIActionClassifier: sizes must be positive "
                f"(n_labels={n_labels}, hidden_dim={hidden_dim}, num_layers={num_layers})"
            )
        if kernel_size % 2 == 0:
            raise ValueError(
                f"Text2HOIActionClassifier: kernel_size must be odd (got {kernel_size})"
            )

        self.n_labels = n_labels
        self.num_frames = num_frames
        self.joints_per_frame = joints_per_frame
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 2D projection width: 42 joints × 2 coords = 84. IMoS used 44 = 22×2.
        self.proj_dim: int = joints_per_frame * 2

        # Two 2-layer LSTMs (batch_first=False, no dropout). IMoS action_rec.py:22,24.
        self.lstm_xy = nn.LSTM(self.proj_dim, hidden_dim, num_layers=num_layers)
        self.lstm_xz = nn.LSTM(self.proj_dim, hidden_dim, num_layers=num_layers)

        # Post-LSTM Conv1d on last-frame hidden (B, 1, H). IMoS action_rec.py:23,25.
        pad = (kernel_size - 1) // 2
        self.conv_xy = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=pad)
        self.conv_xz = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=pad)

        # Linear head. IMoS action_rec.py:30.
        self.head = nn.Linear(hidden_dim, n_labels)

    def forward(
        self,
        points: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """IMoS-2-stream forward for Text2HOI-shaped input.

        Args:
            points:  (B, num_frames, joints_per_frame, feat_dim) float32.
                     Wrist-centered 3D joint positions from canonical_to_text2hoi_input.
            lengths: IGNORED (fixed num_frames).

        Returns:
            {"logits": (B, n_labels), "features": (B, hidden_dim)}.
            `logits` is pre-sigmoid (training applies sigmoid + BCE).
        """
        if points.ndim != 4:
            raise ValueError(
                f"Text2HOIActionClassifier.forward: expected 4D input "
                f"(B, {self.num_frames}, {self.joints_per_frame}, {self.feat_dim}), "
                f"got {tuple(points.shape)}"
            )
        if points.shape[1] != self.num_frames:
            raise ValueError(
                f"Text2HOIActionClassifier.forward: expected num_frames={self.num_frames}, "
                f"got {points.shape[1]}. Input shape: {tuple(points.shape)}"
            )
        if points.shape[2] != self.joints_per_frame:
            raise ValueError(
                f"Text2HOIActionClassifier.forward: expected joints_per_frame={self.joints_per_frame}, "
                f"got {points.shape[2]}. Input shape: {tuple(points.shape)}"
            )
        if points.shape[3] != self.feat_dim:
            raise ValueError(
                f"Text2HOIActionClassifier.forward: expected feat_dim={self.feat_dim}, "
                f"got {points.shape[3]}. Input shape: {tuple(points.shape)}"
            )

        B, T, J, _ = points.shape  # (B, 15, 42, 3)
        H = self.hidden_dim

        # Two 2D projections (IMoS action_rec.py:51-52).
        xy = points[..., 0:2]                              # (B, T, 42, 2)
        xz = points[..., [0, 2]]                           # (B, T, 42, 2)

        # Flatten to (B, T, 84) then transpose to (T, B, 84) for batch_first=False LSTM.
        xy_flat = xy.reshape(B, T, self.proj_dim)
        xz_flat = xz.reshape(B, T, self.proj_dim)
        xy_tbc = xy_flat.transpose(0, 1).contiguous()      # (T, B, 84)
        xz_tbc = xz_flat.transpose(0, 1).contiguous()

        out_xy, _ = self.lstm_xy(xy_tbc)                   # (T, B, H)
        out_xz, _ = self.lstm_xz(xz_tbc)

        # Last-timestep output (IMoS action_rec.py:55-56 uses `lstm_out[-1]`).
        t_xy = out_xy[-1]                                  # (B, H)
        t_xz = out_xz[-1]

        # Conv1d on (B, 1, H) (IMoS action_rec.py:57-58).
        y_xy = self.conv_xy(t_xy.unsqueeze(1)).squeeze(1)  # (B, H)
        y_xz = self.conv_xz(t_xz.unsqueeze(1)).squeeze(1)

        # Sum-fuse. IMoS action_rec.py:60-61.
        y3 = y_xy + y_xz                                   # (B, H)

        # tanh(y3) * 0.1 — IMoS action_rec.py:63 penultimate feature.
        features = torch.tanh(y3) * 0.1                    # (B, H)

        # Linear head → logits (pre-sigmoid).
        logits = self.head(y3)                             # (B, n_labels)

        return {"logits": logits, "features": features}
