"""A name-to-backbone registry.

Backbones register a zero-argument factory under a name so the command line and
config files can refer to them as strings. Register your own encoder the same way::

    from models.registry import register
    from models.base import BackboneWrapper

    @register("my_encoder")
    def _build():
        return MyEncoder(weights="/path/to/ckpt.pt")

Import the module that calls ``register`` before looking the name up (for example
from your project's entry point, or by adding it to ``models``).
"""
from __future__ import annotations

from typing import Callable, Dict, List

from models.base import BackboneWrapper

BackboneFactory = Callable[[], BackboneWrapper]

_REGISTRY: Dict[str, BackboneFactory] = {}


def register(name: str) -> Callable[[BackboneFactory], BackboneFactory]:
    """Decorator registering a backbone factory under ``name``."""
    def wrap(factory: BackboneFactory) -> BackboneFactory:
        if name in _REGISTRY:
            raise ValueError(f"Backbone {name!r} is already registered.")
        _REGISTRY[name] = factory
        return factory
    return wrap


def build_backbone(name: str) -> BackboneWrapper:
    """Instantiate the backbone registered under ``name``."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown backbone {name!r}. Registered backbones: "
            f"{', '.join(available()) or '(none)'}."
        ) from None
    return factory()


def available() -> List[str]:
    """Names of all registered backbones, sorted."""
    return sorted(_REGISTRY)
