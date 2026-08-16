"""Backbone wrappers for SARL.

A backbone is any pretrained audio encoder, adapted to the benchmark by
subclassing :class:`~models.base.BackboneWrapper`. See ``base.py`` for the
contract and ``template.py`` for a copy-and-fill starting point.

Importing this package registers the weight-free reference backbone (``rawfeat_*``).
Register your own with :func:`~models.registry.register`.
"""

from models.base import BackboneWrapper, FORMAT_CHANNELS
from models.registry import available, build_backbone, register

# Import for their registration side effects.
from models import baselines  # noqa: F401

__all__ = [
    "BackboneWrapper",
    "FORMAT_CHANNELS",
    "register",
    "build_backbone",
    "available",
]
