"""GTIdentitySampler — indexes GT episodes by text prompt and yields them unchanged.

Used as a sanity-check sampler for eval pipeline self-test. A perfect metric score
with GTIdentitySampler proves the eval pipeline is correct end-to-end.
"""

import logging
from typing import Dict, List

import numpy as np

from umm.eval.handobject.canonical import CanonicalSample
from umm.eval.handobject.ummsampler import SamplerConfig, UMMSampler

logger = logging.getLogger(__name__)


class GTIdentitySampler(UMMSampler):
    """Sanity sampler: indexes GT episodes by text prompt and yields them
    unchanged on .sample(text). Used for eval pipeline self-test.

    Multiple episodes may share the same text prompt. If n_repeats_per_prompt > 1,
    samples are drawn with replacement in a deterministic order derived from
    (config.seed + hash(text)).
    """

    def __init__(
        self,
        config: SamplerConfig,
        gt_episodes: List[CanonicalSample],
    ) -> None:
        """Initialize the sampler with a pre-loaded list of GT episodes.

        Args:
            config:      SamplerConfig specifying dataset, seed, n_repeats_per_prompt.
            gt_episodes: All GT CanonicalSamples to index (may share text prompts).

        Raises:
            ValueError: If gt_episodes is empty.
        """
        super().__init__(config)

        if len(gt_episodes) == 0:
            raise ValueError("GTIdentitySampler: gt_episodes must be non-empty")

        # Build text -> list[CanonicalSample] index.
        # Multiple episodes may share the same text prompt.
        self._index: Dict[str, List[CanonicalSample]] = {}
        for ep in gt_episodes:
            if not isinstance(ep, CanonicalSample):
                raise ValueError(
                    f"GTIdentitySampler: gt_episodes entries must be CanonicalSample, "
                    f"got {type(ep).__name__}"
                )
            text = ep.text
            if text not in self._index:
                self._index[text] = []
            self._index[text].append(ep)

        total_prompts = len(self._index)
        total_episodes = len(gt_episodes)
        logger.info(
            "GTIdentitySampler: indexed %d episodes across %d unique prompts",
            total_episodes,
            total_prompts,
        )

    def sample(self, text: str) -> List[CanonicalSample]:
        """Yield n_repeats_per_prompt GT episodes for the given text prompt.

        Draws with replacement if n_repeats_per_prompt > len(candidates).
        Deterministic: seeded by (config.seed + hash(text)).

        Args:
            text: Episode text annotation. Must be a key in the GT index.

        Returns:
            List of CanonicalSample of length config.n_repeats_per_prompt.

        Raises:
            ValueError: If text is not present in the GT index.
            ValueError: If text is empty.
        """
        if not text:
            raise ValueError("GTIdentitySampler.sample: text must be non-empty")

        if text not in self._index:
            available = sorted(self._index.keys())
            raise ValueError(
                f"GTIdentitySampler.sample: text not in GT index: {text!r}. "
                f"Available prompts ({len(available)}): {available[:5]}..."
            )

        candidates = self._index[text]
        n = self.config.n_repeats_per_prompt

        # Deterministic seed: combine base seed with text hash.
        # hash() is not stable across processes in Python 3.3+, so we use
        # a simple but reproducible hash via sum of UTF-8 bytes.
        text_bytes = text.encode("utf-8")
        text_hash = sum(b * (i + 1) for i, b in enumerate(text_bytes)) & 0xFFFFFFFF
        seed = (self.config.seed + text_hash) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)  # noqa: NPY002 — RandomState for reproducibility

        if n <= len(candidates):
            # n <= pool size: sample WITHOUT replacement so identical GT episodes
            # are never drawn twice. This is intentional (maximizes diversity under
            # n_repeats) and stronger than the spec's "with replacement if needed"
            # which only mandates with-replacement when n > pool size.
            chosen_indices = rng.choice(len(candidates), size=n, replace=False)
        else:
            # n > pool size: must use with-replacement to reach n draws.
            chosen_indices = rng.choice(len(candidates), size=n, replace=True)

        return [candidates[int(idx)] for idx in chosen_indices]

    @property
    def unique_prompts(self) -> List[str]:
        """Sorted list of all unique text prompts in the GT index."""
        return sorted(self._index.keys())

    @property
    def n_episodes(self) -> int:
        """Total number of indexed GT episodes (sum across all prompts)."""
        return sum(len(v) for v in self._index.values())
