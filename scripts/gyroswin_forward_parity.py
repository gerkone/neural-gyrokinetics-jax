"""Forward parity: torch GyroSwinMultitask vs translated JAX port on GyroSwin_tiny.

Builds both from the checkpoint config, loads best.pth into torch and translates
it into JAX, runs the same random input + conditioning through both, and reports
cosine similarity / max|diff| / MSE per output (df, phi, flux).

IMPORTANT — residual semantics: the JAX SwinBlock now implements ONLY the
corrected single residual (post-e79b021); the doubled-residual path was removed
by decision. Upstream commit e79b021 (2026-07-01) fixed
``SwinTransformerBlock.forward`` summing the MLP shortcut twice. Every neurips26
checkpoint (April 2026) was trained with the OLD doubled-residual semantics,
which the JAX ``SwinBlock`` intentionally replicates. To compare like-for-like,
cross-framework parity is checked with BOTH sides on the corrected forward
(default). Pass ``--pre-fix-residual`` to monkeypatch the torch block back to
the pre-fix doubled forward — that shows the training-time outputs the legacy
checkpoints no longer reproduce, and will NOT match the JAX side.
"""
import argparse
import glob
import os
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/system/user/publicwork/galletti/git/neural-gyrokinetics-gitlab")  # torch repo

import numpy as np, torch, jax, jax.numpy as jnp, jax.random as jr

ap = argparse.ArgumentParser()
ap.add_argument("name", nargs="?", default="GyroSwin_tiny")
ap.add_argument("--pre-fix-residual", action="store_true",
                help="monkeypatch the torch block to the pre-e79b021 doubled residual "
                     "(training-time semantics of the neurips26 ckpts; will NOT match jax)")
args = ap.parse_args()

CK = sorted(glob.glob(f"/restricteddata/ukaea/checkpoints/neurips26/{args.name}/*/config.yaml"), key=len)[0].rsplit("/", 1)[0]
print("checkpoint:", CK)
RES = (32, 8, 16, 85, 32)  # (vp, mu, s, x, y)
full = yaml.safe_load(open(CK + "/config.yaml"))
mcfg = full["model"]
COND_KEYS = sorted(mcfg["conditioning"])  # torch sorts alphabetically
print("cond order:", COND_KEYS)

# ---- torch model ----
def _stub_ds():
    class S: pass
    s = S(); s.active_keys = ["re", "im"]; s.resolution = RES
    s.phi_resolution = (RES[2], RES[3], RES[4]); return s
from omegaconf import OmegaConf
tcfg = OmegaConf.create({"model": mcfg,
    "dataset": {"input_fields": ["df"], "separate_zf": True, "real_potens": True,
                "active_keys": ["re", "im"]}, "logging": {"model_summary": False}})
from neugk.gyroswin.models import get_model

if args.pre_fix_residual:
    # restore the pre-e79b021 doubled MLP shortcut the checkpoints were trained with
    from neugk.models.nd_vit import swin_layers as _sl

    def _pre_fix_forward(self, x):
        shortcut = self.skip(x)
        x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)
        shortcut = x
        x = x + self.forward_part2(x)
        x = shortcut + x
        return x

    _sl.SwinTransformerBlock.forward = _pre_fix_forward
    print("torch reference: PRE-fix (doubled MLP shortcut, matches ckpt training)")
else:
    print("torch reference: POST-fix (e79b021 single shortcut)")

tmodel = get_model(tcfg, dataset=_stub_ds())
sd = torch.load(CK + "/best.pth", map_location="cpu", weights_only=False)["model_state_dict"]
missing, unexpected = tmodel.load_state_dict(sd, strict=False)
print(f"torch load: {len(missing)} missing, {len(unexpected)} unexpected")
tmodel.eval()

# ---- jax model ----
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
    yaml.safe_dump({"model": mcfg, "dataset": {"separate_zf": True, "resolution": list(RES)}}, f)
    cfgp = f.name
from neugk_jax.gyroswin.models.gyroswin import build_gyroswin_from_config
from neugk_jax.translate import translate_gyroswin, load_torch_state
jmodel = build_gyroswin_from_config(cfgp, key=jr.PRNGKey(0), resolution=list(RES))
jmodel, miss, unused = translate_gyroswin(jmodel, load_torch_state(CK + "/best.pth"))
print(f"translate: {len(miss)} missing, {len(unused)} unused (torch aliases/dead heads)")

# ---- shared input ----
rng = np.random.default_rng(0)
in_ch = 4  # 2 active_keys * (2 if separate_zf)
df_np = rng.standard_normal((in_ch, *RES)).astype(np.float32) * 0.1
cond_np = rng.standard_normal(len(COND_KEYS)).astype(np.float32)

# torch forward (batched)
df_t = torch.from_numpy(df_np)[None]  # (1, C, vp, mu, s, x, y)
cond_kw = {k: torch.tensor([[cond_np[i]]]) for i, k in enumerate(COND_KEYS)}
with torch.no_grad():
    tout = tmodel(df_t, **cond_kw)
def _t2n(x): return x.detach().cpu().numpy()
tdf = _t2n(tout[0] if isinstance(tout, (tuple, list)) else tout["df"])[0]
tphi = _t2n(tout[1] if isinstance(tout, (tuple, list)) else tout["phi"])[0]
tflux = tout.get("flux") if isinstance(tout, dict) else None

# jax forward (unbatched)
jout = jmodel(jnp.asarray(df_np), jnp.asarray(cond_np), inference=True)
jdf = np.asarray(jout["df"]); jphi = np.asarray(jout["phi"])

def cmp(name, a, b):
    a = np.asarray(a).ravel().astype(np.float64); b = np.asarray(b).ravel().astype(np.float64)
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH torch{a.shape} vs jax{b.shape}"); return
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f"  {name}: cos={cos:.6f}  max|diff|={np.abs(a-b).max():.3e}  mse={np.mean((a-b)**2):.3e}  "
          f"(torch range [{a.min():.2f},{a.max():.2f}])")

print("\n=== forward parity (fp32) ===")
print(f"  shapes: torch df{tdf.shape} phi{tphi.shape} | jax df{jdf.shape} phi{jphi.shape}")
cmp("df", tdf, jdf)
cmp("phi", tphi, jphi)
if tflux is not None and "flux" in jout:
    tf = float(np.ravel(_t2n(tflux))[0]); jf = float(np.ravel(np.asarray(jout["flux"]))[0])
    print(f"  flux: torch={tf:.6f}  jax={jf:.6f}  |diff|={abs(tf - jf):.3e}")
