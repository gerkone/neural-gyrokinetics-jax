"""Token layers by spec, so U-Net and autoencoder stages can swap Swin, ViT and Transolver.

The caller passes the arguments shared by every stage; each factory takes the ones in its signature
(window arguments do not reach the global layers), while the spec options go to the factory as given,
so an option the kind does not have raises.
"""

from __future__ import annotations

from typing import Callable

from neugk_jax.models.spec import Spec, accepted, parse_spec
from neugk_jax.models.swin import BlockStack, swin_layer
from neugk_jax.models.transolver import transolver_layer
from neugk_jax.models.vit import vit_layer

TOKEN_LAYERS: dict[str, Callable[..., BlockStack]] = {
    "swin": swin_layer,
    "vit": vit_layer,
    "transolver": transolver_layer,
}


def token_layer(spec: Spec, dim: int, depth: int, num_heads: int, *, key, **shared) -> BlockStack:
    """``depth`` blocks of the spec's kind on ``(*grid, dim)`` tokens; ``shared`` as for :func:`swin_layer`."""
    kind, options = parse_spec(spec, TOKEN_LAYERS)
    factory = TOKEN_LAYERS[kind]
    return factory(dim, depth, num_heads, key=key, **{**accepted(factory, shared), **options})
