"""Translate an upstream DiT (``DIFF_FLOW``) checkpoint into our equinox DiT.

Our JAX DiT now mirrors upstream's field layout 1:1
(``encoder``/``ape``/``backbone``/``decoder``/``time_embed``/``cond_embed``),
so the rename table is mostly stripping the ``.inner.`` infix introduced
by our ``Linear`` wrapper plus a few list-vs-Sequential index fixes
(``encoder.0.inner.w`` → ``encoder.0.w`` etc.). The DiT modulation
projection still lives at ``mod.proj`` on our side and ``dit.modulation``
on torch's; that rename is preserved below.

Usage::

    python scripts/translate_dit_ckpt.py \\
        --torch-ckpt /restricteddata/.../DIFF_FLOW/.../best.pth \\
        --config     /restricteddata/.../DIFF_FLOW/.../config.yaml \\
        --ae-ckpt    /tmp/ae_translated.eqx \\
        --ae-config  /restricteddata/.../AE_noCond/.../config.yaml \\
        --out        /tmp/dit_translated.eqx
"""

from __future__ import annotations

import argparse
import re

import jax.random as jr
import yaml

from neugk_jax.diffusion.dit import DiT
from neugk_jax.training.checkpoint import load_model_only, save_model_only
from scripts.translate_ckpt import (
    iter_leaves,
    load_torch_state,
    translate as _translate_base,
    _is_non_persistent,
)


_LAYERS_RE = re.compile(r"\.layers\.(\d+)")


def dit_jax_to_torch(jax_name: str) -> list[str]:
    """JAX-name → list of candidate torch state-dict names for the DiT."""
    base = jax_name.replace(".inner.", ".")
    candidates = [base]

    # backbone.blocks.X.* matches torch; DiT modulation: mod.proj.{w,b} ↔ torch dit.modulation.{w,b}
    base = base.replace(".mod.proj.", ".dit.modulation.")
    # MLP stride (.layers.N → .mlp.{3N}) — applies inside SwinBlock/DiT MLPs
    base = _LAYERS_RE.sub(lambda m: f".mlp.{int(m.group(1)) * 3}", base)
    if base not in candidates:
        candidates.append(base)
    return candidates


def build_dit_from_config(cfg_path: str, ae, *, key) -> DiT:
    """Construct our equinox DiT with dims matching the upstream config + AE."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    vit = mcfg["vit"]
    dim = mcfg["latent_dim"]
    grid = tuple(ae.bottleneck_grid_size)
    z_dim = int(ae.bottleneck_dim)
    n_cond = len(mcfg.get("conditioning", []) or [])
    return DiT(
        space=len(grid),
        z_dim=z_dim,
        dim=dim,
        grid_size=grid,
        depth=vit["depth"],
        num_heads=vit["num_heads"],
        n_cond=n_cond,
        key=key,
        mlp_ratio=vit.get("mlp_ratio", 2.0),  # upstream default
        drop_path=vit.get("drop_path", 0.1),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torch-ckpt", required=True)
    p.add_argument("--config", required=True, help="DIFF_FLOW config.yaml")
    p.add_argument("--ae-ckpt", required=True, help="translated AE checkpoint")
    p.add_argument("--ae-config", required=True, help="AE config.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    # build the AE to extract the latent shape
    from scripts.translate_ckpt import build_ae_from_config
    ae_template = build_ae_from_config(args.ae_config, key=jr.PRNGKey(0))
    ae = load_model_only(args.ae_ckpt, ae_template)

    torch_state = load_torch_state(args.torch_ckpt)
    print(f"loaded torch DiT state: {len(torch_state)} keys")
    model = build_dit_from_config(args.config, ae, key=jr.PRNGKey(0))
    print(f"built DiT: latent_shape={model.latent_shape}, cond_dim={model.cond_dim}")

    # walk leaves + apply remap
    leaves = list(iter_leaves(model))
    used = set()
    missing = []
    replacements = {}
    for name, leaf in leaves:
        matched = False
        for cand in dit_jax_to_torch(name):
            if cand in torch_state:
                tw = torch_state[cand]
                if tuple(tw.shape) == tuple(leaf.shape):
                    replacements[name] = tw
                    used.add(cand)
                    matched = True
                    break
                # APE pos_embed has a leading singleton in torch; squeeze before comparing
                if (
                    tw.ndim == leaf.ndim + 1
                    and tw.shape[0] == 1
                    and tuple(tw.shape[1:]) == tuple(leaf.shape)
                ):
                    replacements[name] = tw.squeeze(0)
                    used.add(cand)
                    matched = True
                    break
        if not matched and not _is_non_persistent(name):
            missing.append((name, tuple(leaf.shape)))

    unused = sorted(set(torch_state) - used)
    print(f"translated leaves: {len(replacements)} / {len(leaves)}")
    if missing:
        print(f"missing JAX leaves ({len(missing)}):")
        for n, s in missing[:10]:
            print(f"  {n}  shape={s}")
        if len(missing) > 10:
            print(f"  ... ({len(missing) - 10} more)")
    if unused:
        print(f"unused torch keys ({len(unused)}):")
        for n in unused[:10]:
            print(f"  {n}  shape={torch_state[n].shape}")
        if len(unused) > 10:
            print(f"  ... ({len(unused) - 10} more)")

    # apply replacements via deepcopy + setattr (same mechanism as translate_ckpt)
    import copy
    import jax.numpy as jnp
    new_model = copy.deepcopy(model)
    for dotted, value in replacements.items():
        parts = dotted.split(".")
        node = new_model
        for p_ in parts[:-1]:
            node = (
                node[int(p_)] if isinstance(node, list)
                else (node[p_] if isinstance(node, dict) else getattr(node, p_))
            )
        leaf = getattr(node, parts[-1])
        object.__setattr__(
            node, parts[-1],
            jnp.asarray(value, dtype=leaf.dtype),
        )

    save_model_only(args.out, new_model)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
