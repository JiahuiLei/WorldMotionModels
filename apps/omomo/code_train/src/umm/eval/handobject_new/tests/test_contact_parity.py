"""Bit-parity test for contact %-error against the paper's formula.

GrabNet's released code does NOT implement a contact-map metric (verified
by grep over ext/baselines/grabnet/). The formula therefore comes from
the paper's Table 1 column header, at
eval/handobject_new/baselines/papers/tex_sources/2008.11200/
GRAB___PAPER___04___grabnet___TAB___01___evaluation.tex:14-16:
    "Contact %" = percentage of vertices whose binary contact state
    disagrees between a prediction and the closest-GT grasp.

Our reference here is a direct, inlined implementation of that formula;
bit-parity means any future refactor of metrics.py cannot silently drift.
"""
import numpy as np
import pytest

from umm.eval.handobject_new.metrics import contact_error


def _paper_contact_pct(
    contact_pred: np.ndarray, contact_gt: np.ndarray
) -> float:
    """Reference formula from the paper Table 1 (see module docstring)."""
    a = contact_pred.astype(bool)
    b = contact_gt.astype(bool)
    return float(np.mean(a != b)) * 100.0


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("shape", [(778,), (4, 778), (8, 16, 778)])
def test_contact_parity(seed: int, shape) -> None:
    """Assert |ours - reference| < 1e-9 across random binary maps."""
    rng = np.random.default_rng(seed)
    a = (rng.random(shape) > 0.5).astype(np.bool_)
    b = (rng.random(shape) > 0.5).astype(np.bool_)

    ours = contact_error(a, b)
    theirs = _paper_contact_pct(a, b)

    assert abs(ours - theirs) < 1e-9, (
        f"seed={seed} shape={shape}: ours={ours!r} vs theirs={theirs!r} "
        f"diff={ours - theirs!r}"
    )


def test_contact_all_agree() -> None:
    """Identical maps -> 0%."""
    a = np.array([1, 0, 1, 0, 1], dtype=np.bool_)
    assert contact_error(a, a) == 0.0


def test_contact_all_disagree() -> None:
    """Complementary maps -> 100%."""
    a = np.array([1, 0, 1, 0, 1], dtype=np.bool_)
    b = ~a
    assert contact_error(a, b) == 100.0


def test_contact_shape_mismatch_raises() -> None:
    """Shape mismatch must raise."""
    with pytest.raises(ValueError, match="shape mismatch"):
        contact_error(np.zeros(5, dtype=bool), np.zeros(7, dtype=bool))


def test_contact_accepts_int_inputs() -> None:
    """Int {0,1} inputs should cast to bool without surprise."""
    a = np.array([1, 0, 1, 0, 0], dtype=np.int64)
    b = np.array([1, 1, 0, 0, 0], dtype=np.int64)
    # Disagreement at positions 1 and 2 -> 2/5 = 40%.
    assert contact_error(a, b) == pytest.approx(40.0)
