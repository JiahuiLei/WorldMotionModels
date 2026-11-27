"""World Motion Model (WMM) -- top-level package.

A diffusion-based generative model for general motion synthesis across
multiple embodiments (robot arms, humanoids, hands). Trained with
PyTorch Lightning.

Subpackages:
    models: Model architectures (Agent, MotionMachine, PoseIO, etc.).
    datasets: Dataset loaders for various embodiment domains.
    training_utils: Configuration parsing, trainer setup, logging, and callbacks.
    sims: Simulation environments and evaluation callbacks.
"""

__version__ = "0.1.0"

# Import commonly used modules for convenience
try:
    from . import models
    from . import datasets
    from . import training_utils
    from . import sims
except ImportError:
    # Allow partial imports during development
    pass

__all__ = [
    "__version__",
    "models",
    "datasets",
    "training_utils",
    "sims",
]
