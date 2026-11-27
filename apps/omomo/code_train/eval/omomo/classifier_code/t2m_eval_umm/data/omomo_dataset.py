"""Minimal shim to unblock ``eval_fid_rprec.py``.

Upstream CHOIS's classifier training code imports ``_tokenize_and_tag`` from
a ``data.omomo_dataset`` module that is not distributed with the rest of
the t2m_eval package. This shim provides the ONE symbol the eval path
requires: ``_tokenize_and_tag(text) -> list[(word, universal_POS)]``.

The classifier vectorizer (``utils/word_vectorizer.WordVectorizer``) does
the heavy lifting of lexicon-based VIP overrides (Loc/Body/Obj/Act/Desc)
at ``__getitem__`` time, so this shim only needs to produce the base
universal POS tag per token. Uses NLTK's ``pos_tag(..., tagset='universal')``.

Tagset mapping per POS_enumerator (utils/word_vectorizer.py):
    VERB, NOUN, DET, ADP, NUM, AUX, PRON, ADJ, ADV — universal
    OTHER — fallback for any tag not in the enumerator (. , X, etc.)
    VIP categories — applied later in WordVectorizer, not here.

This module is TEXT-ONLY (no torch Dataset class). Classifier *training*
with CanoObjectTrajDataset is a separate concern that we do not run in
this repo; R_prec / FID at eval time only needs the text tokenizer.
"""

from __future__ import annotations

from typing import List, Tuple

import nltk

# POS tags emitted by POS_enumerator; anything else collapses to OTHER
# and the VIP overrides are applied later by WordVectorizer.__getitem__.
_UNIVERSAL_TAGS = {"VERB", "NOUN", "DET", "ADP", "NUM", "PRON", "ADJ", "ADV"}

# NLTK does not have a standalone "AUX" tag in its universal mapping — AUX
# collapses to VERB for text-to-motion. Leave as-is; VIP still fires.

_NLTK_READY = False


def _ensure_nltk() -> None:
    """Download NLTK resources on first call. Idempotent."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    # Prefer offline data if NLTK_DATA is set; otherwise download.
    for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng", "universal_tagset"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"taggers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    _NLTK_READY = True


def _tokenize_and_tag(text: str) -> List[Tuple[str, str]]:
    """Return ``[(word_lower, universal_POS_or_OTHER), ...]`` for ``text``.

    Empty / whitespace-only input returns an empty list; the caller
    (``eval_fid_rprec._encode_texts``) already handles that case.
    """
    _ensure_nltk()
    text = str(text).strip()
    if not text:
        return []
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens, tagset="universal")
    return [(w, p if p in _UNIVERSAL_TAGS else "OTHER") for w, p in tagged]
