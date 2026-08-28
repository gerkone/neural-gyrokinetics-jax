"""Linear-field conditioning: metadata pairing, dataset sharing, model, runner.

Mirrors ``test_runners_smoke``'s synthetic dataset, plus a synthetic ``_Lin`` raw dir
holding one float64 Fortran-order ``FDS`` snapshot per trajectory.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from omegaconf import OmegaConf

RESOLUTION = (4, 4, 4, 16, 8)


def _make_traj(root: Path, name: str, *, n_t: int, resolution=RESOLUTION):
    traj = root / f"{name}_ifft_realpotens"
    data = traj / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(name)) & 0xFFFFFFFF)
    for t in range(n_t):
        df = rng.standard_normal((2, *resolution)).astype(np.float32)
        df.tofile(data / f"timestep_{t:05d}.bin")
    meta = {
        "timesteps": np.arange(n_t, dtype=np.float64),
        "flux": rng.standard_normal(n_t).astype(np.float32),
        "ion_temp_grad": np.array([2.3], dtype=np.float32),
        "density_grad": np.array([1.1], dtype=np.float32),
        "s_hat": np.array([0.8], dtype=np.float32),
        "q": np.array([1.4], dtype=np.float32),
        "resolution": np.array(resolution),
        "geometry": {k: np.ones((1,), dtype=np.float64)
                     for k in ("krho", "ints", "intmu", "intvp", "vpgr", "mugr",
                               "bn", "efun", "rfun", "bt_frac", "parseval",
                               "mas", "tmp", "d2X", "signz", "signB", "kxrh", "little_g")},
    }
    with open(traj / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)
    return traj


def _make_lin(raw_root: Path, name: str, *, resolution=RESOLUTION, scale=1.0):
    lin = raw_root / f"{name}_Lin"
    lin.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash("lin" + name)) & 0xFFFFFFFF)
    arr = (scale * rng.standard_normal((2, *resolution))).astype(np.float64)
    arr.ravel(order="F").tofile(lin / "FDS")
    return lin, arr


@pytest.fixture
def lincond_dir(tmp_path):
    root, raw = tmp_path / "pre", tmp_path / "raw"
    for name in ("iteration_0", "iteration_1"):
        _make_traj(root, name, n_t=8)
        # a big amplitude difference between the two runs: rms normalization must remove it
        _make_lin(raw, name, scale=1.0 if name.endswith("0") else 1e5)
    return root, raw


def test_preprocess_links_and_materializes(lincond_dir):
    from neugk_jax.dataset.linear import k_to_real
    from neugk_jax.dataset.preprocess import run_link_linear

    root, raw = lincond_dir
    run_link_linear(path=str(root), trajs="all", raw_root=str(raw),
                    materialize=False, num_workers=2)
    meta = pickle.load(open(root / "iteration_0_ifft_realpotens" / "metadata.pkl", "rb"))
    assert meta["linear_fds"] == str(raw / "iteration_0_Lin" / "FDS")
    assert "linear_bin" not in meta

    run_link_linear(path=str(root), trajs="all", raw_root=str(raw),
                    materialize=True, num_workers=2)
    traj = root / "iteration_0_ifft_realpotens"
    meta = pickle.load(open(traj / "metadata.pkl", "rb"))
    assert meta["linear_bin"] == "data/linear.bin"
    assert tuple(meta["linear_shape"]) == (2, *RESOLUTION)
    assert meta["linear_space"] == "real"

    stored = np.fromfile(traj / "data" / "linear.bin", dtype=np.float32).reshape(2, *RESOLUTION)
    raw_fds = np.fromfile(raw / "iteration_0_Lin" / "FDS", dtype=np.float64)
    expect = k_to_real(np.reshape(raw_fds, (2, *RESOLUTION), order="F").astype(np.float32))
    assert np.allclose(stored, expect, atol=1e-6)


def _dataset(root, raw, **kw):
    from neugk_jax.dataset import LinearCondCycloneDataset, NumpyBackend
    kw.setdefault("linear_normalize", "rms")   # cky needs the metadata moments
    return LinearCondCycloneDataset(
        path=str(root), split="train", trajectories="iteration_{0-1}",
        backend=NumpyBackend(), raw_root=str(raw), normalization=None, **kw,
    )


def test_linear_cky_profile_is_global_and_inheritable(lincond_dir):
    """(2, nky) profile pooled over the split; val can inherit the train one."""
    from neugk_jax.dataset.preprocess import run_link_linear

    root, raw = lincond_dir
    run_link_linear(path=str(root), trajs="all", raw_root=str(raw),
                    materialize=True, space="raw", stats=True, num_workers=2)
    meta = pickle.load(open(root / "iteration_0_ifft_realpotens" / "metadata.pkl", "rb"))
    assert meta["linear_rms_cky"].shape == (2, RESOLUTION[-1])
    assert meta["linear_rms_ckxky"].shape == (2, RESOLUTION[-2], RESOLUTION[-1])

    ds = _dataset(root, raw, linear_to_real=False, linear_separate_zf=False,
                  linear_normalize="cky")
    assert ds.linear_profile.shape == (2, RESOLUTION[-1])
    assert len(ds.linear_k_bins) == 2 and ds.linear_channels == 2
    # after dividing by the pooled profile the per-(channel, ky) rms sits near 1
    per = np.stack([np.sqrt(np.mean(np.square(ds.get_linear(f), dtype=np.float64),
                                    axis=(1, 2, 3, 4))) for f in ds.metadata])
    assert 0.2 < per.mean() < 5.0, per.mean()
    # ...but not identically 1 per trajectory: the deviation is the conditioning signal
    assert per.std() > 1e-3

    # an injected profile wins (this is how the val split inherits the training one)
    other = ds.linear_profile * 2.0
    ds2 = _dataset(root, raw, linear_to_real=False, linear_separate_zf=False,
                   linear_normalize="cky", linear_profile=other)
    assert np.allclose(ds2.linear_profile, other)
    assert np.allclose(ds2.get_linear(0), ds.get_linear(0) / 2.0, atol=1e-6)

    # zf splitting is real-space only and must be rejected in the spectral path
    with pytest.raises(ValueError, match="real-space construct"):
        _dataset(root, raw, linear_to_real=False, linear_separate_zf=True,
                 linear_normalize="cky")


def test_dataset_shares_one_field_per_trajectory(lincond_dir):
    root, raw = lincond_dir
    ds = _dataset(root, raw)
    assert len(ds.linear_files) == 2                      # one entry per trajectory, not per sample
    s0, s1 = ds[0], ds[1]
    assert s0.linear.shape == (2, *RESOLUTION)
    assert s0.linear is s1.linear                        # same trajectory → same array, no copy
    assert abs(float(np.sqrt(np.mean(s0.linear ** 2))) - 1.0) < 1e-5

    # the 1e5 amplitude gap between the two runs is normalized away
    other = next(ds[i] for i in range(len(ds)) if int(ds[i].file_index) == 1)
    assert other.linear is not s0.linear
    assert abs(float(np.sqrt(np.mean(other.linear ** 2))) - 1.0) < 1e-5

    from neugk_jax.dataset.cyclone import collate
    batch = collate([s0, other])
    assert batch.linear.shape == (2, 2, *RESOLUTION)


def test_dataset_reads_materialized_bin(lincond_dir):
    from neugk_jax.dataset.preprocess import run_link_linear

    root, raw = lincond_dir
    run_link_linear(path=str(root), trajs="all", raw_root=str(raw),
                    materialize=True, num_workers=2)
    ds_bin = _dataset(root, raw)
    assert len(ds_bin.linear_bins) == 2
    # reading the materialized bin must match parsing the raw FDS
    ds_raw = _dataset(root, raw, linear_normalize="none")
    ds_raw.linear_bins = {}
    ds_raw._linear_cache.clear()
    assert np.allclose(ds_bin.get_linear(0),
                       ds_raw.get_linear(0) / np.sqrt(np.mean(ds_raw.get_linear(0) ** 2)),
                       atol=1e-5)


def test_separate_zf_channels(lincond_dir):
    root, raw = lincond_dir
    ds = _dataset(root, raw, separate_zf=True)
    assert ds.linear_channels == 4
    assert ds[0].linear.shape == (4, *RESOLUTION)


def _tiny_encoder(key, in_channels=2, code_dim=8):
    from neugk_jax.diffusion.lincond_dit import LinearFieldEncoder
    return LinearFieldEncoder(
        base_resolution=list(RESOLUTION), in_channels=in_channels,
        patch_size=[2, 0, 2, 4, 2], window_size=[2, 0, 2, 2, 2],
        dim=16, depth=[1, 1], num_heads=[2, 2], num_layers=2,
        code_dim=code_dim, c_multiplier=1, merging_depth=1,
        use_rpb=False, gated_attention=False, qk_norm=False, key=key,
    )


def test_encoder_pools_to_a_code():
    enc = _tiny_encoder(jr.PRNGKey(0))
    field = jr.normal(jr.PRNGKey(1), (2, *RESOLUTION))
    code = enc(field)
    assert code.shape == (8,)
    assert jnp.all(jnp.isfinite(code))
    # a batch of fields vmaps cleanly (the training path)
    codes = jax.vmap(enc)(jnp.stack([field, field * 2.0]))
    assert codes.shape == (2, 8)


def test_no_scalar_conditioning_path_exists():
    """The physical parameters must not be able to reach the model at all."""
    import inspect

    from neugk_jax.diffusion import lincond_dit as m

    enc = _tiny_encoder(jr.PRNGKey(0))
    grid = (2, 2, 4)
    model = m.LinearCondDiT(space=len(grid), z_dim=6, dim=16, grid_size=grid, depth=1,
                            num_heads=2, linear_encoder=enc, key=jr.PRNGKey(2))
    # no scalar embedding module, no way to pass scalars in
    assert not hasattr(model, "cond_embed")
    params = inspect.signature(m.LinearCondDiT.__init__).parameters
    assert "n_cond" not in params and "cond_embed_dim" not in params
    call = inspect.signature(m.LinearCondDiT.__call__).parameters
    assert list(call) == ["self", "x", "tstep", "condition", "key", "inference"]
    assert model.cond_dim == model.time_embed.cond_dim + model.lin_proj.weight.shape[0]

    # and the dataset's scalars do not influence a forward pass
    x = jr.normal(jr.PRNGKey(3), (*grid, 6))
    field = jr.normal(jr.PRNGKey(4), (2, *RESOLUTION))
    out = model(x, jnp.float32(0.3), field)
    assert jnp.allclose(out, model(x, jnp.float32(0.3), field))


def test_cross_attention_cond_mode(lincond_dir):
    """cross mode routes the tokens through cross-attention, adaln through modulation."""
    from neugk_jax.diffusion.lincond_dit import LinearCondDiT
    from neugk_jax.models.vit import CrossAttnDiTLayer, DiTLayer

    enc = _tiny_encoder(jr.PRNGKey(0))
    grid = (2, 2, 4)
    x = jr.normal(jr.PRNGKey(3), (*grid, 6))
    field = jr.normal(jr.PRNGKey(4), (2, *RESOLUTION))
    built = {}
    for mode in ("adaln", "cross"):
        model = LinearCondDiT(space=len(grid), z_dim=6, dim=16, grid_size=grid, depth=2,
                              num_heads=2, linear_encoder=enc, cond_mode=mode,
                              key=jr.PRNGKey(2))
        out = model(x, jnp.float32(0.3), field)
        assert out.shape == (*grid, 6)
        # precomputed conditioning must reproduce the inline path
        assert jnp.allclose(out, model(x, jnp.float32(0.3), model.encode_cond(field)),
                            atol=1e-5)
        # and it must actually depend on the field
        assert not jnp.allclose(
            out, model(x, jnp.float32(0.3), model.encode_cond(field * -1.0)), atol=1e-4)
        built[mode] = model

    assert isinstance(built["adaln"].backbone, DiTLayer)
    assert isinstance(built["cross"].backbone, CrossAttnDiTLayer)
    # adaln pools to a code, cross keeps the grid tokens
    assert built["adaln"].encode_cond(field).ndim == 1
    tok = built["cross"].encode_cond(field)
    assert tok.ndim == 2 and tok.shape[0] == np.prod(enc.grid_sizes[-1])
    # cross mode's modulation is driven by the timestep alone
    assert built["cross"].cond_dim == built["cross"].time_embed.cond_dim
    assert built["cross"].lin_proj is None and built["adaln"].ctx_proj is None


def test_dit_accepts_field_or_precomputed_code():
    from neugk_jax.diffusion.lincond_dit import LinearCondDiT
    enc = _tiny_encoder(jr.PRNGKey(0))
    grid = (2, 2, 4)
    model = LinearCondDiT(
        space=len(grid), z_dim=6, dim=16, grid_size=grid, depth=1, num_heads=2,
        linear_encoder=enc, key=jr.PRNGKey(2),
    )
    x = jr.normal(jr.PRNGKey(3), (*grid, 6))
    field = jr.normal(jr.PRNGKey(4), (2, *RESOLUTION))
    t = jnp.float32(0.3)
    from_field = model(x, t, field)
    from_code = model(x, t, model.encode_cond(field))
    assert from_field.shape == (*grid, 6)
    assert jnp.allclose(from_field, from_code, atol=1e-5)
    # conditioning actually moves the output
    other = model(x, t, model.encode_cond(field * -1.0))
    assert not jnp.allclose(from_field, other, atol=1e-4)


def _tiny_ae_cfg(path, resolution):
    return OmegaConf.create({
        "workflow": "ae", "seed": 0, "output_path": str(path / "out"),
        "model": {
            "name": "ae", "decouple_mu": True, "latent_dim": 16,
            "patch": {"patch_size": [2, 0, 2, 4, 2], "window_size": [2, 0, 2, 2, 2],
                      "merging_depth": 1, "unmerging_depth": 1,
                      "merging_hidden_ratio": 2.0, "unmerging_hidden_ratio": 2.0,
                      "c_multiplier": 1},
            "vit": {"num_heads": [2], "depth": [1], "use_rpb": False,
                    "gated_attention": False, "qk_norm": False, "qkv_bias": False},
            "bottleneck": {"dim": 8, "depth": 1, "num_heads": 2, "normalized_latent": False},
            "middle_depth": 1, "middle_num_heads": 2, "hidden_mlp_ratio": 2.0,
        },
        "dataset": {
            "name": "cyclone", "path": str(path), "backend": "numpy",
            "resolution": list(resolution),
            "training_trajectories": "iteration_0",
            "validation_trajectories": "iteration_1",
            "input_fields": ["df"], "conditions": ["itg", "dg", "s_hat", "q"],
            "separate_zf": False, "offset": 0, "normalization": None,
        },
        "training": {"batch_size": 1, "n_epochs": 1, "learning_rate": 3e-4,
                     "final_learning_rate": 1e-6, "weight_decay": 0.0,
                     "clip_grad": True, "clip_to": 1.0, "exclude_from_wd": []},
        "validation": {"validate_every_n_epochs": 1},
        "logging": {"mode": "disabled", "tqdm": False},
        "distributed": {"enable": False, "n_nodes": 1},
    })


def test_build_lincond_dit_from_config(lincond_dir, tmp_path):
    """The eval path rebuilds the model from a config + AE (checkpoint template)."""
    import yaml

    from neugk_jax.diffusion.lincond_dit import LinearCondDiT
    from neugk_jax.translate import build_ae_from_config, build_lincond_dit_from_config

    root, _ = lincond_dir
    cfg = OmegaConf.to_container(_tiny_ae_cfg(root, RESOLUTION))
    cfg["model"] = {
        "name": "lincond_dit", "model_type": "lincond_dit", "latent_dim": 32,
        "vit": {"num_heads": 2, "depth": 1, "mlp_ratio": 2.0, "drop_path": 0.0},
        "linear_encoder": {
            "dim": 16, "depth": [1, 1], "num_heads": [2, 2], "num_layers": 2,
            "code_dim": 8, "patch_size": [2, 0, 2, 4, 2], "window_size": [2, 0, 2, 2, 2],
            "c_multiplier": 1, "merging_depth": 1, "use_rpb": False,
            "gated_attention": False, "qk_norm": False,
        },
    }
    ae_cfg_path = tmp_path / "ae.yaml"
    OmegaConf.save(_tiny_ae_cfg(root, RESOLUTION), ae_cfg_path)
    cfg_path = tmp_path / "lincond.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    ae = build_ae_from_config(str(ae_cfg_path), key=jr.PRNGKey(0), resolution=RESOLUTION)
    dit = build_lincond_dit_from_config(str(cfg_path), ae, key=jr.PRNGKey(1),
                                       resolution=RESOLUTION)
    assert isinstance(dit, LinearCondDiT)
    assert dit.latent_shape == (*ae.bottleneck_grid_size, ae.bottleneck_dim)
    assert dit.lin_encoder.in_channels == 2   # separate_zf false in the tiny config
    out = dit(jnp.zeros(dit.latent_shape), jnp.float32(0.5),
              jnp.zeros((2, *RESOLUTION)))
    assert out.shape == dit.latent_shape


def test_lincond_runner_constructs_and_steps(lincond_dir, tmp_path):
    from neugk_jax.training.checkpoint import save_model_only
    from neugk_jax.translate import build_ae_from_config

    root, raw = lincond_dir
    ae_cfg = _tiny_ae_cfg(root, RESOLUTION)
    ae_dir = tmp_path / "ae_ckpt"
    ae_dir.mkdir(exist_ok=True)
    OmegaConf.save(ae_cfg, ae_dir / "config.yaml")
    ae = build_ae_from_config(str(ae_dir / "config.yaml"), key=jr.PRNGKey(0),
                             resolution=RESOLUTION)
    save_model_only(ae_dir / "ae.eqx", ae)

    cfg = OmegaConf.create(OmegaConf.to_container(ae_cfg))
    cfg.workflow = "diffusion_lincond"
    cfg.ae_checkpoint = str(ae_dir / "ae.eqx")
    cfg.dataset.training_trajectories = "iteration_{0-1}"
    cfg.dataset.validation_trajectories = "iteration_1"
    cfg.dataset.raw_root = str(raw)
    cfg.dataset.linear_preload = True
    cfg.model = OmegaConf.create({
        "name": "lincond_dit", "model_type": "lincond_dit",
        "latent_dim": 32, "minibatch_ot": False,
        "vit": {"num_heads": 2, "depth": 1, "mlp_ratio": 2.0, "drop_path": 0.0},
        "linear_encoder": {
            "dim": 16, "depth": [1, 1], "num_heads": [2, 2], "num_layers": 2,
            "code_dim": 8, "pool": "max", "decouple_mu": True,
            "patch_size": [2, 0, 2, 4, 2], "window_size": [2, 0, 2, 2, 2],
            "c_multiplier": 1, "merging_depth": 1, "use_rpb": False,
            "gated_attention": False, "qk_norm": False,
        },
    })
    cfg.training.batch_size = 4
    cfg.training.cond_group_size = 2

    from neugk_jax.diffusion.runner_lincond import LinearCondFlowMatchingRunner
    r = LinearCondFlowMatchingRunner(cfg, output_path=cfg.output_path)
    assert len(r.train_ds) > 0
    assert r.latent_shape == (*r.ae.bottleneck_grid_size, r.ae.bottleneck_dim)

    # grouped batches: each block of cond_group_size shares a trajectory
    sel = next(iter(r._grouped_batches(r.train_ds, 4, 2, jr.PRNGKey(0))))
    fids = [r.train_ds.flat_index_to_file_and_tstep[int(i)][0] for i in sel]
    assert fids[0] == fids[1] and fids[2] == fids[3]

    logs = r.train_epoch(1, jr.PRNGKey(0))
    assert np.isfinite(logs["loss"])

    out = r.sample(key=jr.PRNGKey(0), batch=2, steps=2,
                   cond=jnp.stack([jnp.asarray(r.val_ds[0].linear)] * 2))
    assert out["df"].shape == (2, 2, *RESOLUTION)

def test_swin_legacy_double_shortcut_and_rms_norm():
    """Guards the two port bugs that made pre-e79b021 checkpoints unusable."""
    import equinox as eqx

    from neugk_jax.models.swin import SwinBlock, SwinLayer
    from neugk_jax.models.utils import RMSNorm

    grid, win, dim = (4, 8), (2, 4), 8
    x = jr.normal(jr.PRNGKey(0), (*grid, dim))
    kw = dict(mlp_ratio=2.0, drop_path=0.0, rms_norm=True)
    single = SwinBlock(dim, 2, grid, win, key=jr.PRNGKey(1), **kw)
    legacy = SwinBlock(dim, 2, grid, win, key=jr.PRNGKey(1),
                       legacy_double_shortcut=True, **kw)

    # zero the MLP so the block output IS the post-attention residual x_res1:
    # single -> x_res1, legacy -> 2*x_res1 (the pre-e79b021 upstream topology)
    def _zero_mlp(blk):
        zeroed = jax.tree_util.tree_map(
            lambda a: jnp.zeros_like(a) if eqx.is_array(a) else a, blk.mlp)
        return eqx.tree_at(lambda b: b.mlp, blk, zeroed)
    x_res1 = _zero_mlp(single)(x, inference=True)
    assert jnp.allclose(_zero_mlp(legacy)(x, inference=True), 2.0 * x_res1, atol=1e-6)
    # with the MLP live the two differ by exactly x_res1
    assert jnp.allclose(legacy(x, inference=True) - single(x, inference=True),
                        x_res1, atol=1e-5)

    # SwinLayer must forward rms_norm to its blocks (it used to land in **_unused)
    layer = SwinLayer(2, dim, depth=2, num_heads=2, grid_size=grid, window_size=win,
                      key=jr.PRNGKey(2), rms_norm=True)
    assert all(isinstance(b.norm1, RMSNorm) and isinstance(b.norm2, RMSNorm)
               for b in layer.blocks)
    layer_ln = SwinLayer(2, dim, depth=1, num_heads=2, grid_size=grid, window_size=win,
                         key=jr.PRNGKey(2), rms_norm=False)
    assert not isinstance(layer_ln.blocks[0].norm1, RMSNorm)
    layer_legacy = SwinLayer(2, dim, depth=1, num_heads=2, grid_size=grid, window_size=win,
                             key=jr.PRNGKey(2), legacy_double_shortcut=True)
    assert layer_legacy.blocks[0].legacy_double_shortcut
