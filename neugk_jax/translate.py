"""Torch → Equinox checkpoint translation (weights only; templates come from ``models.build``).

- ``load_torch_state(.pth)`` → ``dict[str, np.ndarray]``
- ``translate_ae`` / ``translate_dit`` / ``translate_gyroswin(model, state)`` → ``(model,
  missing, unused)``: every array leaf takes the first torch key among its candidate names
  with a matching shape
- ``load_or_translate(template, ckpt_path)`` — dispatches on suffix + template type
"""

from __future__ import annotations

import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

_LAYERS_RE = re.compile(r"\.layers\.(\d+)")
_NON_PERSISTENT = (".attn_mask", ".rel_pos", ".rpb", ".rpb_idx", ".omega")


def _unimportable(e: Exception) -> bool:
    # a pickled class whose module is missing or fails at import (deepspeed without cuda)
    return (
        isinstance(e, (ImportError, AttributeError)) or type(e).__name__ == "MissingCUDAException"
    )


def _stub_pickle_module():
    """``pickle`` shim whose unpickler stubs classes it cannot import.

    Replaces any class that fails to import (e.g. deepspeed's ``LossScaler``,
    which needs a CUDA toolchain) with a permissive placeholder, so tensor
    data can still be read out.
    """
    import pickle
    import types

    mod = types.ModuleType("neugk_jax_stub_pickle")
    mod.__dict__.update(pickle.__dict__)
    mod.__name__ = "neugk_jax_stub_pickle"

    class _StubUnpickler(pickle.Unpickler):
        def find_class(self, mod_name, name):
            try:
                return super().find_class(mod_name, name)
            except Exception as e:
                if not _unimportable(e):
                    raise
                # permissive ctor: enums/scalers are rebuilt as ``Cls(value)``
                return type(
                    name,
                    (),
                    {
                        "__init__": lambda self, *a, **k: None,
                        "__setstate__": lambda self, state: None,
                    },
                )

    mod.Unpickler = _StubUnpickler
    return mod


def load_torch_state(path: str) -> dict[str, np.ndarray]:
    """Open a torch ``.pth`` on CPU and return a flat numpy dict."""
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        if not _unimportable(e):
            raise
        # trainer-side objects (e.g. deepspeed loss scalers) whose modules do not import here
        blob = torch.load(
            path, map_location="cpu", weights_only=False, pickle_module=_stub_pickle_module()
        )
    sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    out = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        out[k] = t.numpy()
    return out


def _key_name(k) -> str:
    for attr in ("name", "idx", "key"):
        if hasattr(k, attr):
            return str(getattr(k, attr))
    return str(k)


def named_leaves(model) -> list[tuple[str, jax.Array]]:
    flat, _ = jax.tree_util.tree_flatten_with_path(model)
    return [(".".join(map(_key_name, p)), leaf) for p, leaf in flat if eqx.is_array(leaf)]


def _is_non_persistent(name: str) -> bool:
    return any(name.endswith(s) for s in _NON_PERSISTENT)


# jax -> torch renames: equinox's wrapped Linear/LayerNorm, the u-net modules, DiT modulation
_INNER = ((".inner.", "."),)
_UNET = (
    (".swin.", ".swin_att."),
    (".downsample.proj.", ".downsample.reduction."),
    (".gate.proj.", ".gate.gate.1."),
)
_DIT = ((".mod.proj.", ".dit.modulation."),)
# torch wraps proj_concat in an nn.Sequential, so the param sits at proj_concat.0.*
_PROJ_CONCAT = ((".proj_concat.", ".proj_concat.0."),)


def _name_map(*renames, strip_prefix: str = ""):
    """Candidate torch keys of a jax leaf name: itself, then ``renames`` applied in order
    (after the ``.inner.`` unwrap and ``strip_prefix``), with ``layers.i`` -> ``mlp.3i``."""

    def candidates(jax_name: str) -> list[str]:
        base = jax_name.replace(*_INNER[0]).removeprefix(strip_prefix)
        for old, new in renames:
            base = base.replace(old, new)
        return [jax_name, _LAYERS_RE.sub(lambda m: f".mlp.{int(m.group(1)) * 3}", base)]

    return candidates


_ae_name_map = _name_map(*_UNET, strip_prefix="backbone.")
_dit_name_map = _name_map(*_DIT)
_gyroswin_name_map = _name_map(*_UNET, *_DIT, *_PROJ_CONCAT)


def _translate(model, torch_state, name_map, *, strict: bool = False):
    leaves = named_leaves(model)
    used = set()
    missing = []
    replacements = {}
    for name, leaf in leaves:
        matched = False
        for cand in name_map(name):
            if cand in torch_state:
                tw = torch_state[cand]
                if tuple(tw.shape) == tuple(leaf.shape):
                    replacements[name] = tw
                    used.add(cand)
                    matched = True
                    break
                # torch ape has a leading singleton batch axis — squeeze it
                if (
                    tw.ndim == leaf.ndim + 1
                    and tw.shape[0] == 1
                    and tuple(tw.shape[1:]) == tuple(leaf.shape)
                ):
                    replacements[name] = np.asarray(tw).squeeze(0)
                    used.add(cand)
                    matched = True
                    break
        if not matched and not _is_non_persistent(name):
            missing.append((name, tuple(leaf.shape)))
    unused = sorted(set(torch_state) - used)
    if strict and (missing or unused):
        raise RuntimeError(f"translate strict: missing={len(missing)}, unused={len(unused)}")
    flat, treedef = jax.tree_util.tree_flatten_with_path(model)
    new = [
        jnp.asarray(replacements[n], leaf.dtype) if n in replacements else leaf
        for n, leaf in ((".".join(map(_key_name, p)), leaf) for p, leaf in flat)
    ]
    return jax.tree_util.tree_unflatten(treedef, new), missing, unused


def translate_ae(model, torch_state, *, strict: bool = False):
    return _translate(model, torch_state, _ae_name_map, strict=strict)


def translate_dit(model, torch_state, *, strict: bool = False):
    return _translate(model, torch_state, _dit_name_map, strict=strict)


def translate_gyroswin(model, torch_state, *, strict: bool = False):
    return _translate(model, torch_state, _gyroswin_name_map, strict=strict)


def load_or_translate(template, ckpt_path: str):
    """``.eqx`` → load; ``.pth`` → on-the-fly translate. Returns the model."""
    from neugk_jax.diffusion.dit import DiT
    from neugk_jax.gyroswin.models.gyroswin import GyroSwinMultitask
    from neugk_jax.training.checkpoint import load_model_only

    if ckpt_path.endswith(".eqx"):
        return load_model_only(ckpt_path, template)
    state = load_torch_state(ckpt_path)
    if isinstance(template, GyroSwinMultitask):
        fn = translate_gyroswin
    elif isinstance(template, DiT):
        fn = translate_dit
    else:
        fn = translate_ae
    model, missing, unused = fn(template, state)
    print(f"  translated torch -> jax: {len(missing)} missing, {len(unused)} unused")
    return model


def report(model, torch_state: dict, missing, unused, limit: int = 20) -> None:
    """Print the translation coverage and the first ``limit`` missing / unused names."""
    total = len(named_leaves(model))
    print(f"translated leaves: {total - len(missing)} / {total}")
    if missing:
        print(f"missing JAX leaves ({len(missing)}):")
        for n, s in missing[:limit]:
            print(f"  {n}  shape={s}")
    if unused:
        print(f"unused torch keys ({len(unused)}):")
        for n in unused[:limit]:
            print(f"  {n}  shape={torch_state[n].shape}")
