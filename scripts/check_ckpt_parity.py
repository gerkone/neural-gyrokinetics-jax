"""Translation-parity harness across the neurips26 checkpoints.

For each checkpoint: build the JAX template from the checkpoint's own config,
run the matching translate_*, and report (#missing JAX params, #unused torch
keys). Goal: gyroswin -> 0 missing / only-intentional unused; AE/DiT stay at
their known-good baseline (regression guard while editing the shared U-Net).

Usage:
  python scripts/check_ckpt_parity.py [ae|diff|tiny|cold|warm|all]
"""
import os, sys, yaml, tempfile, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax, jax.random as jr, numpy as np, equinox as eqx

CKPT_ROOT = "/restricteddata/ukaea/checkpoints/neurips26"
RES = [32, 8, 16, 85, 32]

def _find(name, fname="config.yaml"):
    import glob
    hits = glob.glob(os.path.join(CKPT_ROOT, name, "**", fname), recursive=True)
    return sorted(hits, key=len)[0] if hits else None

def _pth(name):
    import glob
    for f in ("best.pth", "ckp.pth"):
        hits = glob.glob(os.path.join(CKPT_ROOT, name, "**", f), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    return None

def _shaped_cfg(cfg_path, extra_ds=None):
    full = yaml.safe_load(open(cfg_path))
    ds = {"separate_zf": full.get("dataset", {}).get("separate_zf", True), "resolution": RES}
    if extra_ds:
        ds.update(extra_ds)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"model": full["model"], "dataset": ds}, f)
        return f.name, full

def _report(tag, model, sd, fn):
    n_leaves = len(list(jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))))
    _, missing, unused = fn(model, sd, strict=False)
    # torch registers shared modules under several names (df_up_blocks =
    # df_unet.up_blocks, phi_middle = phi_unet.middle, ...), so the state_dict
    # holds byte-identical duplicates. Only count an unused key as REAL if no
    # used key carries the same bytes.
    used_bytes = {sd[k].tobytes() for k in set(sd) - set(unused)}
    real_unused = [k for k in unused if sd[k].tobytes() not in used_bytes]
    print(f"[{tag}] jax_leaves={n_leaves} torch_keys={len(sd)} -> "
          f"MISSING={len(missing)} UNUSED={len(unused)} "
          f"({len(unused) - len(real_unused)} aliases, {len(real_unused)} real)")
    return missing, real_unused

def check_gyroswin(name):
    from neugk_jax.gyroswin.models.gyroswin import build_gyroswin_from_config
    from neugk_jax.translate import translate_gyroswin, load_torch_state
    cfg, _ = _shaped_cfg(_find(name))
    model = build_gyroswin_from_config(cfg, key=jr.PRNGKey(0), resolution=RES)
    sd = load_torch_state(_pth(name))
    miss, real_unused = _report(name, model, sd, translate_gyroswin)
    # show a few of each for debugging
    for m in miss[:12]:
        print("    MISSING", m)
    # cold/warm train with flux loss weight 0.0 -> torch still builds the
    # (conditioned) flux head but it never gets a gradient; dead weights.
    dead_flux = [u for u in real_unused if u.startswith("flux_head")]
    other = [u for u in real_unused if not u.startswith("flux_head")]
    if dead_flux:
        print(f"    dead flux_head weights (loss weight 0.0): {len(dead_flux)}")
    print(f"    non-trivial UNUSED: {len(other)}")
    for u in other[:15]:
        print("    UNUSED", u)

def check_ae(name="AE_noCond"):
    from neugk_jax.translate import build_ae_from_config, translate_ae, load_torch_state
    cfg, _ = _shaped_cfg(_find(name))
    try:
        model = build_ae_from_config(cfg, key=jr.PRNGKey(0), resolution=RES)
    except Exception as e:
        print(f"[{name}] BUILD FAILED: {type(e).__name__}: {e}")
        return
    sd = load_torch_state(_pth(name))
    _report(name, model, sd, translate_ae)

def check_dit(name="DIFF_FLOW", ae_name="AE_noCond"):
    from neugk_jax.translate import (build_ae_from_config, build_dit_from_config,
                                     translate_dit, load_torch_state)
    # only the AE's shapes matter for the DiT template — no AE weights needed
    ae = build_ae_from_config(_find(ae_name), key=jr.PRNGKey(0), resolution=RES)
    model = build_dit_from_config(_find(name), ae, key=jr.PRNGKey(0))
    sd = load_torch_state(_pth(name))
    miss, unused = _report(name, model, sd, translate_dit)
    for m in miss[:12]:
        print("    MISSING", m)
    for u in unused[:12]:
        print("    UNUSED", u)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all")
    a = ap.parse_args()
    sel = a.which
    if sel in ("ae", "all"):
        check_ae("AE_noCond")
    if sel in ("diff", "all"):
        check_dit("DIFF_FLOW")
    if sel in ("tiny", "all"):
        check_gyroswin("GyroSwin_tiny")
    if sel in ("cold", "all"):
        check_gyroswin("GyroSwin_cold")
    if sel in ("warm", "all"):
        check_gyroswin("GyroSwin_warm")

if __name__ == "__main__":
    main()
