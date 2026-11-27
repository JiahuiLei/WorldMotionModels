"""Deduplicated R_precision: exactly one motion window per unique text.

The 482-pred OMOMO test windows collapse to 89 unique text strings
(OMOMO uses template captions; avg 5.4 windows share each text).
Standard Guo R_precision (random-32-groups) is depressed when a
group contains multiple same-text pairs: argsort places the ties
near the top regardless of which motion is the diagonal "correct"
one. Dedup removes the collision.

Loads the embeddings NPZ written by `eval_fid_rprec.py --embeddings_npz …`
and re-runs only the R_prec on the per-text-unique subset. FID is NOT
recomputed because (a) dedup population is too small (N=89) for a
reliable covariance in 512-D, and (b) FID does not suffer the
within-group contamination that R_prec does.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _rprec(
    motion_emb: np.ndarray, text_emb: np.ndarray, topk: int = 3,
    n_repeats: int = 20, base_seed: int = 0,
) -> dict[int, float]:
    """Global-shuffle R_prec, averaged over `n_repeats`."""
    N = motion_emb.shape[0]
    group = 32
    accum: dict[int, list[float]] = {k: [] for k in range(1, topk + 1)}
    for r in range(n_repeats):
        rng = np.random.default_rng(base_seed + r)
        perm = rng.permutation(N)
        me = motion_emb[perm]
        te = text_emb[perm]
        n_groups = N // group
        hits = {k: 0 for k in range(1, topk + 1)}
        total = 0
        for g in range(n_groups):
            s, e = g * group, (g + 1) * group
            mb = me[s:e]
            tb = te[s:e]
            diff = tb[:, None, :] - mb[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            argsrt = np.argsort(dist, axis=1)
            for i in range(group):
                for k in range(1, topk + 1):
                    if i in argsrt[i, :k]:
                        hits[k] += 1
            total += group
        for k in range(1, topk + 1):
            accum[k].append(hits[k] / max(total, 1))
    return {k: float(np.mean(v)) for k, v in accum.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_npz", required=True,
                    help="Output of eval_fid_rprec.py --embeddings_npz (gt_emb, pr_emb, txt_emb, texts)")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--n_repeats", type=int, default=20)
    ap.add_argument("--base_seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.embeddings_npz, allow_pickle=True)
    gt, pr, tx, texts = d["gt_emb"], d["pr_emb"], d["txt_emb"], d["texts"]

    # Keep first occurrence per unique text.
    seen: dict[str, int] = {}
    keep: list[int] = []
    for i, t in enumerate(texts):
        s = str(t)
        if s not in seen:
            seen[s] = i
            keep.append(i)
    keep_arr = np.asarray(keep, dtype=np.int64)
    gtd = gt[keep_arr]
    prd = pr[keep_arr]
    txd = tx[keep_arr]
    print(f"deduped N={len(keep)} unique texts (from {len(texts)} windows)")

    rp_gt = _rprec(gtd, txd, topk=args.topk, n_repeats=args.n_repeats, base_seed=args.base_seed)
    rp_pr = _rprec(prd, txd, topk=args.topk, n_repeats=args.n_repeats, base_seed=args.base_seed)
    print(f"R_prec GT   top1/2/3 = {rp_gt[1]:.4f} / {rp_gt[2]:.4f} / {rp_gt[3]:.4f}")
    print(f"R_prec pred top1/2/3 = {rp_pr[1]:.4f} / {rp_pr[2]:.4f} / {rp_pr[3]:.4f}")

    out = {
        "r_prec_gt_dedup_text": rp_gt,
        "r_prec_chois_dedup_text": rp_pr,
        "n_unique_texts_eval": int(len(keep)),
        "n_original": int(len(texts)),
        "source_embeddings_npz": args.embeddings_npz,
        "n_repeats": args.n_repeats,
        "base_seed": args.base_seed,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
