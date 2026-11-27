"""Registry entry-point for maniskill_bs.

The UMM model registry (``src/umm/models/__init__.py::get_model``) loads
``model.py::PLModel``. This module re-exports both :class:`Actor` (the
underlying MLP) and :class:`PLModel` (the Lightning wrapper) so downstream
code can import either from here.
"""

from umm.models.maniskill_bs.actor import Actor  # noqa: F401
from umm.models.maniskill_bs.pl_model import PLModel  # noqa: F401

__all__ = ["Actor", "PLModel"]
