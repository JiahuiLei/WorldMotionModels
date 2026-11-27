"""Convert glove.840B.300d.txt -> {our_vab_data.npy, our_vab_words.pkl, our_vab_idx.pkl}.

Idempotent: skips if outputs already exist. Uses chunked numpy parsing for speed.
"""
import io
import os
import pickle as pkl
import sys
import time

import numpy as np


def main() -> None:
    glove_dir = sys.argv[1] if len(sys.argv) > 1 else "./glove"
    glove_txt = os.path.join(glove_dir, "glove.840B.300d.txt")
    out_data = os.path.join(glove_dir, "our_vab_data.npy")
    out_idx = os.path.join(glove_dir, "our_vab_idx.pkl")
    out_words = os.path.join(glove_dir, "our_vab_words.pkl")

    if all(os.path.isfile(p) for p in (out_data, out_idx, out_words)):
        print(f"[build_glove_vocab] Vocab already present in {glove_dir}, skipping.")
        return

    if not os.path.isfile(glove_txt):
        raise FileNotFoundError(
            f"GloVe TXT not found at {glove_txt}. Unzip glove.840B.300d.zip first."
        )

    print(f"[build_glove_vocab] Reading {glove_txt} …")
    t0 = time.time()

    # Preallocate. GloVe 840B has ~2.196M tokens. We'll grow dynamically.
    words: list[str] = []
    # Use a flat numpy buffer that grows in chunks of 100k rows.
    CHUNK = 100_000
    buf = np.empty((CHUNK, 300), dtype=np.float32)
    row = 0
    capacity = CHUNK

    with io.open(glove_txt, "rb") as f:
        for i, raw in enumerate(f):
            # Find first space (end of word).
            sp = raw.find(b" ")
            if sp <= 0:
                continue
            word = raw[:sp].decode("utf-8", errors="replace")
            rest = raw[sp + 1:]
            # Fast parse: use numpy.fromstring on the tail.
            try:
                vec = np.frombuffer(
                    np.char.strip(np.array([rest])).tobytes(),
                    dtype=np.float32,
                    count=-1,
                ) if False else np.fromstring(rest, sep=" ", dtype=np.float32)  # type: ignore
            except Exception:
                continue
            if vec.shape[0] != 300:
                continue
            if row >= capacity:
                buf = np.concatenate([buf, np.empty((CHUNK, 300), dtype=np.float32)], axis=0)
                capacity += CHUNK
            buf[row] = vec
            words.append(word)
            row += 1
            if (i + 1) % 200_000 == 0:
                print(f"  … scanned {i+1} lines, vocab={row}, elapsed={time.time()-t0:.1f}s")

    vecs = buf[:row].copy()
    idx = {w: i for i, w in enumerate(words)}
    print(
        f"[build_glove_vocab] Final vocab={len(words)}, vecs shape={vecs.shape}, "
        f"elapsed={time.time()-t0:.1f}s"
    )
    np.save(out_data, vecs)
    with open(out_idx, "wb") as f:
        pkl.dump(idx, f)
    with open(out_words, "wb") as f:
        pkl.dump(words, f)
    print(f"[build_glove_vocab] Wrote: {out_data}, {out_idx}, {out_words}")


if __name__ == "__main__":
    main()
