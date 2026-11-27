"""Compute FID and R_precision@[1,2,3] for CHOIS predictions vs OMOMO GT test.

Loads the trained Stage-2 classifier (text_motion_features/finest.tar), embeds
motions + texts, computes:
  * FID between GT-test and CHOIS-pred motion embeddings
  * R_prec@k: for each test sample i, rank 31 random distractor text prompts
    + the correct one against motion[i] embedding; check if correct is in top-k.

Inputs:
  - Preprocessed FK'd preds/GT at --preds_npz (from `preprocess_chois_preds.py`)
    Keys: pred_jpos (N,120,24,3), gt_jpos (N,120,24,3), texts (N,)
  - Classifier ckpt at --ckpt_dir/model/finest.tar (defaults to ckpt/omomo/text_motion_features)
  - Min-max stats at --stats_pkl (from omomo cano_min_max_mean_std_.._joints24.p)
  - GloVe vocab at --glove_dir

Outputs:
  - JSON at --out_json with all metrics.
  - Optional --embeddings_npz for downstream plotting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import join as pjoin

import joblib
import numpy as np
import torch
from scipy import linalg

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from networks.modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from data.omomo_dataset import _tokenize_and_tag


def _build_encoders(device: torch.device, ckpt_path: str):
    # Hparams must match training opts (default TrainTexMotMatchOptions).
    dim_pose = 72
    dim_movement_enc_hidden = 512
    dim_movement_latent = 512
    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    dim_text_hidden = 512
    dim_coemb = 512
    dim_motion_hidden = 1024

    mov_enc = MovementConvEncoder(dim_pose, dim_movement_enc_hidden, dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(
        word_size=dim_word, pos_size=dim_pos_ohot,
        hidden_size=dim_text_hidden, output_size=dim_coemb, device=device,
    )
    motion_enc = MotionEncoderBiGRUCo(
        input_size=dim_movement_latent, hidden_size=dim_motion_hidden,
        output_size=dim_coemb, device=device,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    # Upstream CHOIS trainer saves under these keys:
    mov_enc.load_state_dict(ckpt["movement_encoder"])
    text_enc.load_state_dict(ckpt["text_encoder"])
    motion_enc.load_state_dict(ckpt["motion_encoder"])
    mov_enc.to(device).eval()
    text_enc.to(device).eval()
    motion_enc.to(device).eval()
    return mov_enc, text_enc, motion_enc


def _normalize_jpos(jpos: np.ndarray, jmin: np.ndarray, jmax: np.ndarray) -> np.ndarray:
    # jpos (N,120,24,3). Flatten last two dims to 72 then normalize with per-dim min/max.
    N, T, J, _ = jpos.shape
    flat = jpos.reshape(N, T, J * 3).astype(np.float32)
    span = np.where((jmax - jmin) > 1e-8, (jmax - jmin), 1.0).astype(np.float32)
    return 2.0 * (flat - jmin) / span - 1.0


def _encode_texts(
    texts: np.ndarray, w_vectorizer: WordVectorizer, max_text_len: int = 30,
):
    """Return (word_embs: N,L,300), (pos_ohot: N,L,15), (sent_lens: N,)."""
    N = len(texts)
    L = max_text_len
    word_embs = np.zeros((N, L, 300), dtype=np.float32)
    pos_ohot = np.zeros((N, L, len(POS_enumerator)), dtype=np.float32)
    sent_lens = np.zeros((N,), dtype=np.int64)
    for i, txt in enumerate(texts):
        tagged = _tokenize_and_tag(str(txt))
        words = [w for w, _ in tagged][:L]
        poses = [p for _, p in tagged][:L]
        if not words:
            words, poses = ["unk"], ["OTHER"]
        sent_lens[i] = len(words)
        for j, (w, p) in enumerate(zip(words, poses)):
            key = f"{w}/{p}"
            try:
                wv, pv = w_vectorizer[key]
            except KeyError:
                wv, pv = w_vectorizer["unk/OTHER"]
            word_embs[i, j] = wv
            pos_ohot[i, j] = pv
    return word_embs, pos_ohot, sent_lens


def _batch_embed_motion(
    mot: np.ndarray, m_lens: np.ndarray, mov_enc, motion_enc, device, batch_size: int,
) -> np.ndarray:
    """mot: (N,120,72) [-1,1] normalized, m_lens: (N,). Returns (N,512).

    Contract: motion_enc uses `pack_padded_sequence` internally, so the batch
    must be sorted by m_lens descending before the call. We sort per-batch
    and invert afterwards.
    """
    N = mot.shape[0]
    out = np.zeros((N, 512), dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = torch.from_numpy(mot[s:e]).to(device).float()
        ml = torch.from_numpy(m_lens[s:e]).to(device).long()
        order = ml.argsort(descending=True)
        inv = order.argsort()
        x_sorted = x[order]
        ml_sorted = ml[order]
        with torch.no_grad():
            movements = mov_enc(x_sorted).detach()
            ml_unit = ml_sorted // 4  # unit_length=4
            emb = motion_enc(movements, ml_unit)
        emb = emb[inv]
        out[s:e] = emb.cpu().numpy()
    return out


def _batch_embed_text(
    word_embs: np.ndarray, pos_ohot: np.ndarray, sent_lens: np.ndarray,
    text_enc, device, batch_size: int,
) -> np.ndarray:
    N = word_embs.shape[0]
    out = np.zeros((N, 512), dtype=np.float32)
    # pack_padded_sequence needs descending-length batches; sort each batch.
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        w = torch.from_numpy(word_embs[s:e]).to(device).float()
        p = torch.from_numpy(pos_ohot[s:e]).to(device).float()
        sl = torch.from_numpy(sent_lens[s:e]).to(device).long()
        # Sort by sl desc.
        order = sl.argsort(descending=True)
        inv = order.argsort()
        w, p, sl2 = w[order], p[order], sl[order]
        with torch.no_grad():
            emb = text_enc(w, p, sl2)
        emb = emb[inv]
        out[s:e] = emb.cpu().numpy()
    return out


def _fid(mu1, sig1, mu2, sig2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sig1.shape[0]) * eps
        covmean = linalg.sqrtm((sig1 + offset).dot(sig2 + offset))
    if np.iscomplexobj(covmean):
        # Guo reference: raise if imaginary component is non-trivial; otherwise drop.
        im_mag = float(np.abs(covmean.imag).max())
        if im_mag > 1e-3:
            raise ValueError(f"sqrtm imaginary component too large: {im_mag}")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1) + np.trace(sig2) - 2.0 * np.trace(covmean))


def _stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Guo et al. `t2m_eval/utils/metrics.py:69` uses `np.cov(..., rowvar=False)`
    # with default bias (unbiased, N-1 divisor) — match that exactly.
    return x.mean(axis=0), np.cov(x, rowvar=False)


def _r_precision(
    motion_emb: np.ndarray, text_emb: np.ndarray, topk: int = 3, seed: int = 0,
) -> dict[int, float]:
    """Guo-et-al R_precision protocol (HumanML3D / CHOIS t2m_eval):

    1. Shuffle ALL samples across the whole test set (critical: distractors
       must be sampled globally, not from same-sequence neighbours — see
       `chois_release/t2m_eval/motion_loaders/model_motion_loaders.py` which
       uses a stock DataLoader with `shuffle=True` before batching).
    2. Partition into groups of 32 (1 correct + 31 distractors = batch).
    3. For each row i (text_i), compute L2 distance to every motion_j in group.
    4. If argsort puts correct index i within top-k → hit.

    Repeat R times with different seeds (default R=1 here for simplicity;
    Guo repeats 20× and averages — we expose `seed` so  can average).
    """
    rng = np.random.default_rng(seed)
    N = motion_emb.shape[0]
    group = 32
    # GLOBAL shuffle — NOT per-group-of-consecutive-indices.
    perm = rng.permutation(N)
    motion_emb = motion_emb[perm]
    text_emb = text_emb[perm]
    n_groups = N // group
    dropped_tail = N - n_groups * group
    if dropped_tail:
        print(f"[r_precision] dropping {dropped_tail} tail samples (need multiple of {group})")
    hits: dict[int, int] = {k: 0 for k in range(1, topk + 1)}
    total = 0
    for g in range(n_groups):
        s, e = g * group, (g + 1) * group
        mot_batch = motion_emb[s:e]  # (32, 512)
        text_batch = text_emb[s:e]
        # dist[i,j] = ||text_i - motion_j||_2
        diff = text_batch[:, None, :] - mot_batch[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)  # (32, 32); diag is the correct pair
        argsrt = np.argsort(dist, axis=1)
        for i in range(group):
            for k in range(1, topk + 1):
                if i in argsrt[i, :k]:
                    hits[k] += 1
        total += group
    return {k: hits[k] / max(total, 1) for k in hits}


def _r_precision_repeated(
    motion_emb: np.ndarray, text_emb: np.ndarray, topk: int = 3,
    n_repeats: int = 20, base_seed: int = 0,
) -> dict[int, float]:
    """Average R_prec over `n_repeats` different shuffles to reduce variance."""
    accum: dict[int, list[float]] = {k: [] for k in range(1, topk + 1)}
    for r in range(n_repeats):
        d = _r_precision(motion_emb, text_emb, topk=topk, seed=base_seed + r)
        for k, v in d.items():
            accum[k].append(v)
    return {k: float(np.mean(vs)) for k, vs in accum.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_npz", required=True)
    ap.add_argument("--ckpt_dir", default="./ckpt/omomo/text_motion_features")
    ap.add_argument("--stats_pkl", required=True,
                    help="cano_min_max_mean_std_data_window_120_joints24.p")
    ap.add_argument("--glove_dir", default="./glove")
    ap.add_argument("--glove_prefix", default="our_vab")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--embeddings_npz", default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- Load preprocessed preds + GT ---
    data = np.load(args.preds_npz, allow_pickle=True)
    pred_jpos = data["pred_jpos"]  # (N,120,24,3)
    gt_jpos = data["gt_jpos"]
    texts = data["texts"]
    actual_lens = data["actual_lens"].astype(np.int64)  # all 120
    N = pred_jpos.shape[0]
    print(f"loaded {N} pairs; T={pred_jpos.shape[1]} J={pred_jpos.shape[2]}")

    # --- Min-max normalize ---
    stats = joblib.load(args.stats_pkl)
    jmin = np.asarray(stats["global_jpos_min"], dtype=np.float32).reshape(72)
    jmax = np.asarray(stats["global_jpos_max"], dtype=np.float32).reshape(72)
    pred_mot = _normalize_jpos(pred_jpos, jmin, jmax)  # (N,120,72)
    gt_mot = _normalize_jpos(gt_jpos, jmin, jmax)
    # Round m_lens to unit_length multiple.
    m_lens = (actual_lens // 4) * 4
    print(f"normalized motions. m_len unique: {np.unique(m_lens)}")

    # --- Build encoders ---
    ckpt = pjoin(args.ckpt_dir, "model", "finest.tar")
    if not os.path.isfile(ckpt):
        ckpt = pjoin(args.ckpt_dir, "model", "latest.tar")
    mov_enc, text_enc, motion_enc = _build_encoders(device, ckpt)
    ckpt_meta = torch.load(ckpt, map_location="cpu")
    ckpt_epoch = int(ckpt_meta.get("epoch", -1))
    ckpt_iter = int(ckpt_meta.get("iter", -1))
    print(f"loaded ckpt: {ckpt}  epoch={ckpt_epoch} iter={ckpt_iter}")

    # --- Embed motions ---
    gt_emb = _batch_embed_motion(gt_mot, m_lens, mov_enc, motion_enc, device, args.batch_size)
    pr_emb = _batch_embed_motion(pred_mot, m_lens, mov_enc, motion_enc, device, args.batch_size)
    print(f"motion embs: gt {gt_emb.shape}, pred {pr_emb.shape}")

    # --- Embed texts ---
    w_vec = WordVectorizer(args.glove_dir, args.glove_prefix)
    word_embs, pos_ohot, sent_lens = _encode_texts(texts, w_vec)
    txt_emb = _batch_embed_text(word_embs, pos_ohot, sent_lens, text_enc, device, args.batch_size)
    print(f"text embs: {txt_emb.shape}")

    # --- FID: GT vs CHOIS-pred ---
    mu_gt, sig_gt = _stats(gt_emb)
    mu_pr, sig_pr = _stats(pr_emb)
    fid = _fid(mu_gt, sig_gt, mu_pr, sig_pr)
    print(f"FID(GT, CHOIS-pred) = {fid:.4f}")

    # --- R_prec: text → motion, averaged over 20 global shuffles ---
    rp_gt = _r_precision_repeated(gt_emb, txt_emb, topk=3, n_repeats=20, base_seed=args.seed)
    rp_pr = _r_precision_repeated(pr_emb, txt_emb, topk=3, n_repeats=20, base_seed=args.seed)
    print(f"R_prec GT   top1/2/3 = {rp_gt[1]:.4f}/{rp_gt[2]:.4f}/{rp_gt[3]:.4f}  (avg 20 shuffles)")
    print(f"R_prec pred top1/2/3 = {rp_pr[1]:.4f}/{rp_pr[2]:.4f}/{rp_pr[3]:.4f}  (avg 20 shuffles)")

    # --- Matching score diag (optional) ---
    # Mean diagonal pairwise distance: an intra-sample agreement metric.
    match_gt = float(np.linalg.norm(gt_emb - txt_emb, axis=-1).mean())
    match_pr = float(np.linalg.norm(pr_emb - txt_emb, axis=-1).mean())

    metrics = {
        "n_test_windows": int(N),
        "fid_gt_vs_chois": float(fid),
        "r_prec_gt": rp_gt,
        "r_prec_chois": rp_pr,
        "matching_score_gt": match_gt,
        "matching_score_chois": match_pr,
        "classifier_ckpt_path": os.path.abspath(ckpt),
        "classifier_ckpt_epoch": ckpt_epoch,
        "classifier_ckpt_iter": ckpt_iter,
        "preds_npz": os.path.abspath(args.preds_npz),
        "stats_pkl": os.path.abspath(args.stats_pkl),
        "seed": args.seed,
        "r_prec_n_shuffle_repeats": 20,
    }
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {args.out_json}")

    if args.embeddings_npz:
        np.savez(
            args.embeddings_npz,
            gt_emb=gt_emb, pr_emb=pr_emb, txt_emb=txt_emb,
            seq_names=data["seq_names"], texts=texts,
        )
        print(f"wrote {args.embeddings_npz}")


if __name__ == "__main__":
    main()
