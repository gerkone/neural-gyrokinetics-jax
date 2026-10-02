"""Flux regressors conditioned on local operational parameters.

Two surrogates sharing one npz format and one predict path: a small MLP and
an exact-GP posterior mean (kind="gpr"). A direct alternative to the
quasilinear model rather than a part of it -- it maps a local feature vector
straight to per-channel GKW gyroBohm fluxes (default targets Q_i, Q_e,
Gamma_e), trained on nonlinear gyaradax runs by
scripts/train_flux_regressor.py. Weights ship as an npz carrying feature and
target names, input standardization, the target transform and the grid
metadata, so a TORAX transport model can adopt one the way it adopts a Cn
head.
"""

import functools
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_TARGETS = ("q_i", "q_e", "pfe")


def init_mlp(sizes, key):
    """He-initialized (W, b) list for layer sizes like (10, 64, 64, 3)."""
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        params.append((w, jnp.zeros(n_out)))
    return params


def mlp_apply(params, x):
    """Forward pass, gelu hidden activations, linear output."""
    for w, b in params[:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def gpr_apply(model, x):
    """Exact GP posterior mean: RBF kernel dot with precomputed alpha."""
    xt = model["x_train"]
    d2 = jnp.sum(((x[..., None, :] - xt) / model["length_scale"]) ** 2, axis=-1)
    k = model["sigma_f"] ** 2 * jnp.exp(-0.5 * d2)
    return k @ model["alpha"]


@functools.lru_cache(maxsize=4)
def _sklearn_gp(x_train, alpha, length_scale, sigma_f, n_targets):
    """The fitted sklearn GaussianProcessRegressor rebuilt from the saved arrays."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    gp = GaussianProcessRegressor(normalize_y=False, alpha=0.0)
    gp.kernel_ = ConstantKernel(float(sigma_f) ** 2) * RBF(np.asarray(length_scale))
    gp.X_train_ = np.asarray(x_train)
    gp.alpha_ = np.asarray(alpha)
    gp._y_train_mean = np.zeros(n_targets)
    gp._y_train_std = np.ones(n_targets)
    return gp


def gpr_apply_sklearn(model, x):
    """Posterior mean from sklearn's own predict, through a host callback."""
    n_targets = model["alpha"].shape[-1]
    n_features = model["x_train"].shape[-1]
    gp = _sklearn_gp(
        _hashable(model["x_train"]), _hashable(model["alpha"]),
        _hashable(model["length_scale"]), float(model["sigma_f"]), n_targets,
    )

    def host(xv):
        flat = np.asarray(xv, dtype=np.float64).reshape(-1, n_features)
        return gp.predict(flat).reshape(np.shape(xv)[:-1] + (n_targets,))

    return jax.pure_callback(
        host,
        jax.ShapeDtypeStruct(jnp.shape(x)[:-1] + (n_targets,), jnp.float64),
        x,
        vmap_method="expand_dims",
    )


class _hashable(tuple):
    """Hashable view of an array so the rebuilt estimator can be cached."""

    def __new__(cls, a):
        a = np.asarray(a)
        return super().__new__(cls, (a.shape, a.tobytes()))

    def __array__(self, dtype=None):
        return np.frombuffer(self[1]).reshape(self[0])


def predict(model, features):
    """Standardize -> surrogate (mlp or gpr) -> inverse target transform."""
    x = (jnp.asarray(features) - model["x_mean"]) / model["x_std"]
    if model.get("kind") == "gpr":
        # sklearn fitted the gp, so sklearn evaluates it; jax form kept for ad
        y = (gpr_apply(model, x) if os.environ.get("GYARADAX_GPR_BACKEND") == "jax"
             else gpr_apply_sklearn(model, x))
    else:
        y = mlp_apply(model["params"], x)
    if model["y_transform"] == "asinh":
        y = jnp.sinh(y) * model["y_scale"]
    return y


def save_model(path, params, *, feature_names, target_names, x_mean, x_std,
               y_transform, y_scale, grid=None, meta=None, kind="mlp", gpr=None):
    arrays = {"kind": np.asarray(kind)}
    if kind == "gpr":
        arrays.update({k: np.asarray(v) for k, v in gpr.items()})
        arrays["n_layers"] = np.asarray(0)
    else:
        for i, (w, b) in enumerate(params):
            arrays[f"w{i}"] = np.asarray(w)
            arrays[f"b{i}"] = np.asarray(b)
        arrays["n_layers"] = np.asarray(len(params))
    arrays["feature_names"] = np.asarray(feature_names)
    arrays["target_names"] = np.asarray(target_names)
    arrays["x_mean"] = np.asarray(x_mean)
    arrays["x_std"] = np.asarray(x_std)
    arrays["y_transform"] = np.asarray(y_transform)
    arrays["y_scale"] = np.asarray(y_scale)
    arrays["grid"] = np.asarray(json.dumps(grid or {}))
    arrays["meta"] = np.asarray(json.dumps(meta or {}))
    np.savez(path, **arrays)


def load_model(path):
    """Load an MLP surrogate npz into a jax-ready dict."""
    d = np.load(path, allow_pickle=True)
    n = int(d["n_layers"])
    params = [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(n)]
    extra = {}
    if "kind" in d.files and str(d["kind"]) == "gpr":
        extra = {k: jnp.asarray(d[k]) for k in ("x_train", "alpha", "length_scale", "sigma_f")}
    return {
        "kind": str(d["kind"]) if "kind" in d.files else "mlp",
        **extra,
        "params": params,
        "feature_names": tuple(str(s) for s in d["feature_names"]),
        "target_names": tuple(str(s) for s in d["target_names"]),
        "x_mean": jnp.asarray(d["x_mean"]),
        "x_std": jnp.asarray(d["x_std"]),
        "y_transform": str(d["y_transform"]),
        "y_scale": jnp.asarray(d["y_scale"]),
        "grid": json.loads(str(d["grid"])),
        "meta": json.loads(str(d["meta"])),
    }
