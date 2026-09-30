"""Forward parity: torch Swin5DAE vs translated JAX port on AE_noCond.

``--pre-fix-residual`` puts both sides on the pre-e79b021 doubled swin residual the
neurips26 checkpoints were trained with; the default compares the corrected forward.
"""
import argparse
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.append(os.environ.get("NEUGK_TORCH_REPO", os.path.dirname(_ROOT)))

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
from omegaconf import OmegaConf

from neugk_jax.translate import build_ae_from_config, load_torch_state, translate_ae

ap = argparse.ArgumentParser()
ap.add_argument("name", nargs="?", default="AE_noCond")
ap.add_argument("--pre-fix-residual", action="store_true")
args = ap.parse_args()

CK = sorted(glob.glob(f"/restricteddata/ukaea/checkpoints/neurips26/{args.name}/*/config.yaml"),
            key=len)[0].rsplit("/", 1)[0]
RES = (32, 8, 16, 85, 32)
print("checkpoint:", CK)

if args.pre_fix_residual:
    from neugk.models.nd_vit import swin_layers as _sl

    def _pre_fix_forward(self, x):
        shortcut = self.skip(x)
        x = shortcut + self.drop_path(self.forward_part1(x))
        shortcut = x
        x = x + self.forward_part2(x)
        return shortcut + x

    _sl.SwinTransformerBlock.forward = _pre_fix_forward
print("residual:", "PRE-fix (doubled)" if args.pre_fix_residual else "POST-fix (single)")

cfg = OmegaConf.load(CK + "/config.yaml")
cfg.dataset.separate_zf = cfg.dataset.get("separate_zf", True)


class _StubDS:
    active_keys = ["re", "im"]
    resolution = RES


from neugk.pinc.autoencoders import get_autoencoder  # noqa: E402

tmodel = get_autoencoder(cfg, _StubDS())
state = load_torch_state(CK + "/best.pth")
sd = {k: torch.from_numpy(np.asarray(v)) for k, v in state.items()}
missing, unexpected = tmodel.load_state_dict(sd, strict=False)
print(f"torch load: {len(missing)} missing, {len(unexpected)} unexpected")
tmodel.eval()

jmodel = build_ae_from_config(CK + "/config.yaml", key=jr.PRNGKey(0),
                              legacy_double_shortcut=args.pre_fix_residual)
jmodel, miss, unused = translate_ae(jmodel, state)
print(f"translate: {len(miss)} missing, {len(unused)} unused")

in_ch = 4 if cfg.dataset.separate_zf else 2
x = np.random.default_rng(0).standard_normal((in_ch, *RES)).astype(np.float32) * 0.5
with torch.no_grad():
    tout = tmodel(torch.from_numpy(x)[None], return_latent=True)
jout = jmodel(jnp.asarray(x), return_latent=True)


def cmp(name, a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
    print(f"  {name}: cos={cos:.6f}  relL2={rel:.3e}  max|diff|={np.abs(a - b).max():.3e}")


cmp("latent", tout["latent"].numpy()[0], jout["latent"])
cmp("df", tout["df"].numpy()[0], jout["df"])
