"""Abstract sampler interface for hand-object interaction generation.

All concrete samplers (GTIdentitySampler, trained UMM, trained baselines) implement
this interface. The eval pipeline calls .sample(text) and receives CanonicalSamples.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from umm.eval.handobject.canonical import CanonicalSample


@dataclass
class SamplerConfig:
    """Configuration shared by all sampler implementations.

    Attributes:
        dataset:            "grab" or "arctic".
        seed:               Base random seed for reproducibility.
        n_repeats_per_prompt: How many samples to produce per .sample(text) call.
    """

    dataset: str          # "grab" or "arctic"
    seed: int = 42
    n_repeats_per_prompt: int = 1

    def __post_init__(self) -> None:
        if self.dataset not in ("grab", "arctic"):
            raise ValueError(
                f"SamplerConfig.dataset must be 'grab' or 'arctic', got {self.dataset!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(
                f"SamplerConfig.seed must be int, got {type(self.seed).__name__}"
            )
        if not isinstance(self.n_repeats_per_prompt, int) or self.n_repeats_per_prompt < 1:
            raise ValueError(
                f"SamplerConfig.n_repeats_per_prompt must be int >= 1, "
                f"got {self.n_repeats_per_prompt!r}"
            )


class UMMSampler(ABC):
    """Abstract sampler. Concrete implementations: GTIdentitySampler,
    future trained UMM models, trained baseline models."""

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config

    @abstractmethod
    def sample(self, text: str) -> List[CanonicalSample]:
        """Produce `n_repeats_per_prompt` samples for the given text prompt.

        Returns a list of CanonicalSample of length n_repeats_per_prompt.
        Each call is deterministic given config.seed + text (reproducibility
        requirement).

        Args:
            text: Episode text annotation (must be a non-empty str).

        Returns:
            List of CanonicalSample, length == config.n_repeats_per_prompt.

        Raises:
            ValueError: If text is empty or the sampler has no episodes for text.
        """
