"""Bit-parity test: our v2v_mae == GrabNet's trainer.py:528 formula.

We inline GrabNet's formula as torch code (same operations, same order)
and assert that our numpy port produces the identical value (after
accounting for GrabNet's ``* MESH_SCALER`` conversion to millimetres).
"""
import numpy as np
import pytest
import torch

from umm.eval.handobject_new.metrics import v2v_mae


def _grabnet_v2v_mm(verts_pred: np.ndarray, verts_gt: np.ndarray) -> float:
    """Exact formula from ext/baselines/grabnet/grabnet/train/trainer.py:528.

    Kept verbatim so any change to GrabNet is caught by this test.
    """
    # From ext/baselines/grabnet/grabnet/train/trainer.py:523
    MESH_SCALER = 1000
    # Cast via torch to match the upstream dtype/ops path exactly.
    t_pred = torch.from_numpy(verts_pred)
    t_gt = torch.from_numpy(verts_gt)
    # From trainer.py:528
    return float(torch.mean(torch.abs(t_gt - t_pred) * MESH_SCALER))


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("shape", [(778, 3), (4, 778, 3), (8, 16, 778, 3)])
def test_v2v_parity(seed: int, shape) -> None:
    """Assert |ours * 1000 - grabnet_mm| < 1e-9 across random tensors."""
    rng = np.random.default_rng(seed)
    # MANO-scale vertex coords in metres (roughly O(0.1) m).
    verts_pred = rng.standard_normal(shape).astype(np.float64) * 0.1
    verts_gt = rng.standard_normal(shape).astype(np.float64) * 0.1

    ours_metres = v2v_mae(verts_pred, verts_gt)
    ours_mm = ours_metres * 1000.0  # match GrabNet's reporting unit
    theirs_mm = _grabnet_v2v_mm(verts_pred, verts_gt)

    assert abs(ours_mm - theirs_mm) < 1e-9, (
        f"seed={seed} shape={shape}: ours_mm={ours_mm!r} vs "
        f"theirs_mm={theirs_mm!r} diff={ours_mm - theirs_mm!r}"
    )


def test_v2v_zero_diff() -> None:
    """Identical inputs -> zero error."""
    verts = np.zeros((778, 3), dtype=np.float64)
    assert v2v_mae(verts, verts) == 0.0


def test_v2v_shape_mismatch_raises() -> None:
    """Shape mismatch must raise."""
    with pytest.raises(ValueError, match="shape mismatch"):
        v2v_mae(np.zeros((5, 3)), np.zeros((7, 3)))
