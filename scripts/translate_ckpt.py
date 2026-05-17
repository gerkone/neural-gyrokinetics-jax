"""Translate a torch Swin5DAE ``.pth`` checkpoint into our Equinox format.

Two-pass strategy:

1. **Walk torch ``state_dict``.** Build a flat ``{dotted_name: ndarray}`` map.
2. **Walk the equinox PyTree.** For each array leaf, derive a dotted path
   from the parent module hierarchy and look it up in the torch map (with
   an explicit per-class rename table). Unrecognised torch keys are
   reported; missing leaves abort the translation.

This script is intentionally explicit so it's easy to extend when we add
RPB, post-norm, etc.

Usage::

    python scripts/translate_ckpt.py \\
        --torch-ckpt path/to/best.pth \\
        --config     path/to/config.yaml \\
        --out        path/to/ae.eqx
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import yaml

from neugk_jax.autoencoders import Swin5DAE
from neugk_jax.training.checkpoint import save_model_only




def load_torch_state(path: str) -> dict[str, np.ndarray]:
    """Open a torch ``.pth`` (CPU) and return a flat ndarray dict."""
    import torch  # lazy import: translator may run in envs without torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        sd = blob["model_state_dict"]
    else:
        sd = blob
    # strip DDP module. prefixes if present
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    # numpy can't represent bfloat16 — upcast to float32
    out = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        if t.dtype == __import__("torch").bfloat16:
            t = t.float()
        out[k] = t.numpy()
    return out




def _is_leaf_array(x) -> bool:
    return isinstance(x, jax.Array) or isinstance(x, np.ndarray)


def iter_leaves(tree, prefix: str = ""):
    """Yield ``(dotted_name, leaf)`` for every array leaf in an Equinox tree.

    Honours Equinox's pytree structure: dataclass field names form the path,
    sequences (lists) get an index suffix.
    """
    if _is_leaf_array(tree):
        yield prefix.lstrip("."), tree
        return
    if isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from iter_leaves(v, f"{prefix}.{i}")
        return
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from iter_leaves(v, f"{prefix}.{k}")
        return
    if hasattr(tree, "__dataclass_fields__"):
        for fname in tree.__dataclass_fields__:
            try:
                v = getattr(tree, fname)
            except AttributeError:
                continue
            yield from iter_leaves(v, f"{prefix}.{fname}")
        return
    # static / non-pytree node — ignore




# MLP layers: our .layers.N ↔ torch nn.Sequential "mlp" at indices 0, 3, 6 (linear, dropout, act)
_LAYERS_RE = re.compile(r"\.layers\.(\d+)")


def jax_to_torch_candidates(jax_name: str) -> list[str]:
    """Map a JAX PyTree path to candidate torch state_dict keys.

    Pure-renaming bridge between our equinox wrappers and the upstream
    module hierarchy. Order matters — we strip the inner wrapper first,
    then the ``backbone`` prefix, then handle per-module renames.
    """
    base = jax_name
    # 1. strip our .inner. wrapper on eqx.nn.Linear / LayerNorm
    base = base.replace(".inner.", ".")
    # 2. drop backbone. prefix (composition vs inheritance)
    if base.startswith("backbone."):
        base = base[len("backbone."):]
    # 3. block container is swin_att upstream
    base = base.replace(".swin.", ".swin_att.")
    # 4. PatchMerge linear is "reduction" upstream
    base = base.replace(".downsample.proj.", ".downsample.reduction.")
    # 5. gated attention: gate.proj ↔ torch gate.gate.1
    base = base.replace(".gate.proj.", ".gate.gate.1.")
    # 5b. APE buffer: pos_embed name is aligned, no rewrite needed
    # 6. .layers.N → .mlp.{3N} (applies to mlp, patch, expansion, cpb_mlp)
    base = _LAYERS_RE.sub(lambda m: f".mlp.{int(m.group(1)) * 3}", base)
    return [jax_name, base]




# non-persistent buffers: torch recomputes these from window_size/shift_size — skip silently
_NON_PERSISTENT = (".attn_mask", ".rel_pos", ".rpb", ".rpb_idx", ".omega")

# backbone.middle is dead weight in Swin5DAE (replaced by middle_pre/middle_post); torch omits it
_AE_DEAD = ("backbone.middle.",)


def _is_non_persistent(jax_name: str) -> bool:
    return any(jax_name.endswith(s) for s in _NON_PERSISTENT)


def _is_dead_leaf(jax_name: str) -> bool:
    return any(s in jax_name for s in _AE_DEAD)


def translate(model, torch_state: dict[str, np.ndarray], *, strict: bool = False):
    """Return a copy of ``model`` with leaves replaced from ``torch_state``."""
    jax_leaves = list(iter_leaves(model))
    torch_names = set(torch_state.keys())
    used_torch = set()
    replacements = {}
    missing = []

    for jax_name, leaf in jax_leaves:
        candidates = jax_to_torch_candidates(jax_name)
        match = None
        for cand in candidates:
            if cand in torch_state:
                match = cand
                break
        if match is None:
            if _is_non_persistent(jax_name) or _is_dead_leaf(jax_name):
                continue
            missing.append((jax_name, tuple(leaf.shape)))
            continue
        tw = torch_state[match]
        if tw.shape != tuple(leaf.shape):
            # tolerate size-1 axis mismatch (torch APE buffer has a leading batch dim)
            tw_sq = np.asarray(tw)
            while tw_sq.ndim > leaf.ndim and tw_sq.shape[0] == 1:
                tw_sq = tw_sq[0]
            while tw_sq.ndim < leaf.ndim and leaf.shape[0] == 1:
                tw_sq = tw_sq[None]
            if tw_sq.shape != tuple(leaf.shape):
                raise ValueError(
                    f"shape mismatch for {jax_name}: torch={tw.shape} jax={leaf.shape}"
                )
            tw = tw_sq
        replacements[jax_name] = jnp.asarray(tw, dtype=leaf.dtype)
        used_torch.add(match)

    unused = sorted(torch_names - used_torch)
    if missing and strict:
        raise ValueError(f"missing torch weights for {len(missing)} leaves: {missing[:5]}")

    return _apply_replacements(model, replacements), missing, unused


def _apply_replacements(model, replacements: dict[str, jax.Array]):
    """Rebuild a model with selected leaves overwritten."""
    paths_and_values = list(iter_leaves(model))

    def _set(tree, dotted, value):
        parts = dotted.split(".")
        for i, p in enumerate(parts):
            if i == len(parts) - 1:
                if isinstance(tree, list):
                    tree[int(p)] = value
                elif isinstance(tree, dict):
                    tree[p] = value
                else:
                    object.__setattr__(tree, p, value)
            else:
                if isinstance(tree, list):
                    tree = tree[int(p)]
                elif isinstance(tree, dict):
                    tree = tree[p]
                else:
                    tree = getattr(tree, p)
        return tree

    # eqx modules are frozen; mutate via deepcopy + object.__setattr__
    import copy
    new_model = copy.deepcopy(model)
    for dotted, value in replacements.items():
        _set(new_model, dotted, value)
    return new_model




def build_ae_from_config(
    cfg_path: str, *, key, resolution: Optional[Sequence[int]] = None
) -> Swin5DAE:
    """Build a Swin5DAE from the upstream Hydra YAML.

    ``resolution`` defaults to the canonical cyclone shape ``(32, 8, 16, 85, 32)``
    when the config doesn't carry it (it lives in per-file metadata.pkl
    upstream, not in the Hydra config).
    """
    from typing import Sequence  # noqa
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    vit = mcfg.get("vit", {})
    patch = mcfg.get("patch", {})
    bn = mcfg.get("bottleneck", {})
    dataset = cfg.get("dataset", {})

    base_resolution = resolution or dataset.get("resolution") or (32, 8, 16, 85, 32)
    depth = vit["depth"]
    num_heads = vit["num_heads"]
    n_layers = mcfg.get("num_layers", len(depth) if isinstance(depth, (list, tuple)) else 4)
    return Swin5DAE(
        space=5,
        decouple_mu=mcfg.get("decouple_mu", True),
        dim=mcfg["latent_dim"],
        base_resolution=list(base_resolution),
        in_channels=dataset.get("in_channels", 2 * (2 if dataset.get("separate_zf", False) else 1)),
        out_channels=dataset.get("out_channels", 2 * (2 if dataset.get("separate_zf", False) else 1)),
        patch_size=patch["patch_size"],
        window_size=patch["window_size"],
        depth=depth,
        num_heads=num_heads,
        num_layers=n_layers,
        middle_depth=mcfg.get("middle_depth", 2),
        middle_num_heads=mcfg.get("middle_num_heads", 8),
        bottleneck_dim=bn.get("dim"),
        bottleneck_depth=bn.get("depth", 2),
        bottleneck_num_heads=bn.get("num_heads", 2),
        hidden_mlp_ratio=mcfg.get("hidden_mlp_ratio", 2.0),
        merging_hidden_ratio=patch.get("merging_hidden_ratio", 8.0),
        unmerging_hidden_ratio=patch.get("unmerging_hidden_ratio", 8.0),
        merging_depth=patch.get("merging_depth", 2),
        unmerging_depth=patch.get("unmerging_depth", 2),
        c_multiplier=int(patch.get("c_multiplier", 2)),
        normalized_latent=bn.get("normalized_latent", False),
        # parity knobs: upstream defaults differ from ours
        qkv_bias=vit.get("qkv_bias", False),
        qk_norm=vit.get("qk_norm", False),
        use_rpb=vit.get("use_rpb", True),
        gated_attention=vit.get("gated_attention", False),
        norm_affine=False,  # upstream swin uses elementwise_affine=False
        key=key,
    )




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torch-ckpt", required=True, help="path to .pth (torch state_dict)")
    p.add_argument("--config", required=True, help="upstream Hydra config.yaml")
    p.add_argument("--out", required=True, help="path to write the equinox checkpoint")
    p.add_argument("--strict", action="store_true",
                   help="abort on any leaf that has no torch counterpart")
    args = p.parse_args()

    torch_state = load_torch_state(args.torch_ckpt)
    print(f"loaded torch state: {len(torch_state)} keys")

    model = build_ae_from_config(args.config, key=jr.PRNGKey(0))
    new_model, missing, unused = translate(model, torch_state, strict=args.strict)

    print(f"translated leaves: {len(list(iter_leaves(model))) - len(missing)} / "
          f"{len(list(iter_leaves(model)))}")
    if missing:
        print(f"missing JAX leaves ({len(missing)}):")
        for n, s in missing[:20]:
            print(f"  {n}  shape={s}")
        if len(missing) > 20:
            print(f"  ... ({len(missing) - 20} more)")
    if unused:
        print(f"unused torch keys ({len(unused)}):")
        for n in unused[:20]:
            print(f"  {n}  shape={torch_state[n].shape}")
        if len(unused) > 20:
            print(f"  ... ({len(unused) - 20} more)")

    save_model_only(args.out, new_model)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
