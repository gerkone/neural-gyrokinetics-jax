"""Train the MLP flux surrogate on collected NL datasets.

Inputs: dataset npz files from collect_campaign.py (F, Y_sp, valid, kinetic,
feature_names, grid_meta) and/or the legacy adiabatic dataset (F, Y).
Targets: asinh-scaled (Q_i, Q_e, Gamma_e) in GKW gyroBohm units.

Usage:
  python scripts/train_mlp.py dataset_v2.npz [more.npz...] --out mlp_v2.npz \
      [--hidden 64 64] [--epochs 2000] [--lr 1e-3] [--seed 0]
"""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

from neugk_jax.flux_regressor import init_mlp, mlp_apply, save_model


def load_rows(paths):
    """Stack (F, Y_sp(3,), feature_names, grid) from mixed dataset files."""
    feats, targs, names, grid = [], [], None, {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        n = [str(s) for s in d["feature_names"]]
        names = names or n
        assert n == names, f"feature schema mismatch in {p}"
        F = np.asarray(d["F"], float)
        if "Y_sp" in d.files:
            y = np.asarray(d["Y_sp"], float)
            # per-species channels: Q_i, Q_e, Gamma_e (species 0=ion, 1=electron)
            t = np.stack([y[:, 0, 1], y[:, 1, 1], y[:, 1, 0]], axis=1)
        else:
            # legacy adiabatic set: only species-summed eflux; qe/pfe zeroed
            yy = np.asarray(d["Y"], float)
            t = np.stack([yy, np.zeros_like(yy), np.zeros_like(yy)], axis=1)
        ok = np.isfinite(F).all(axis=1) & np.isfinite(t).all(axis=1)
        # blown NL runs leave absurd tails; heat fluxes are also non-negative
        ok &= (np.abs(t) < 1e4).all(axis=1) & (t[:, :2] >= 0.0).all(axis=1)
        if "valid" in d.files:
            ok &= np.asarray(d["valid"], bool)
        feats.append(F[ok])
        targs.append(t[ok])
        if "grid_meta" in d.files:
            grids = list(json.loads(str(d["grid_meta"])).values())
            if grids:
                grid = grids[0]
        print(f"{p}: {ok.sum()} usable rows")
    return np.concatenate(feats), np.concatenate(targs), names, grid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="mlp", choices=["mlp", "gpr"])
    ap.add_argument("--features", nargs="+", default=None, help="feature-name subset (default: all)")
    args = ap.parse_args()

    F, T, names, grid = load_rows(args.datasets)
    if args.features:
        cols = [names.index(n) for n in args.features]
        F, names = F[:, cols], list(args.features)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(F))
    n_val = max(int(len(F) * args.val_frac), 1)
    vi, ti = idx[:n_val], idx[n_val:]

    x_mean, x_std = F[ti].mean(0), F[ti].std(0) + 1e-12
    # robust scale: outliers must not compress the bulk of the asinh range
    y_scale = np.quantile(np.abs(T[ti]), 0.9, axis=0) / 3.0 + 1e-12
    X = jnp.asarray((F - x_mean) / x_std)
    Y = jnp.asarray(np.arcsinh(T / y_scale))

    if args.model == "gpr":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        kern = ConstantKernel(1.0) * RBF(np.ones(F.shape[1])) + WhiteKernel(1e-2)
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=False, alpha=0.0)
        gp.fit(np.asarray(X[jnp.asarray(ti)]), np.asarray(Y[jnp.asarray(ti)]))
        k = gp.kernel_
        gpr = dict(
            x_train=gp.X_train_,
            alpha=gp.alpha_,
            length_scale=np.atleast_1d(k.k1.k2.length_scale),
            sigma_f=np.sqrt(k.k1.k1.constant_value),
        )
        yv = np.sinh(np.asarray(gpr["sigma_f"] ** 2 * np.exp(
            -0.5 * (((np.asarray(X[jnp.asarray(vi)])[:, None, :] - gpr["x_train"]) / gpr["length_scale"]) ** 2).sum(-1)
        ) @ gpr["alpha"])) * y_scale
        for c, name in enumerate(("Q_i", "Q_e", "Gamma_e")):
            t = T[vi][:, c]
            if np.abs(t).max() > 0:
                ss = 1 - np.sum((t - yv[:, c]) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-30)
                print(f"val R2 {name}: {ss:+.3f}")
        save_model(
            args.out, None, kind="gpr", gpr=gpr,
            feature_names=names, target_names=("q_i", "q_e", "pfe"),
            x_mean=x_mean, x_std=x_std, y_transform="asinh", y_scale=y_scale,
            grid=grid, meta={"datasets": args.datasets, "model": "gpr",
                             "kernel": str(k), "n_train": int(len(ti))},
        )
        print(f"saved -> {args.out}")
        return

    sizes = (F.shape[1], *args.hidden, T.shape[1])
    params = init_mlp(sizes, jax.random.PRNGKey(args.seed))

    def loss(p, i):
        return jnp.mean((mlp_apply(p, X[i]) - Y[i]) ** 2)

    tx = optax.adam(args.lr)
    opt_state = tx.init(params)
    ti_j = jnp.asarray(ti)
    vi_j = jnp.asarray(vi)

    @jax.jit
    def step_fn(p, s):
        loss_val, g = jax.value_and_grad(loss)(p, ti_j)
        updates, s = tx.update(g, s, p)
        return optax.apply_updates(p, updates), s, loss_val

    for step in range(1, args.epochs + 1):
        params, opt_state, loss_val = step_fn(params, opt_state)
        if step % max(args.epochs // 10, 1) == 0:
            print(f"step {step:5d}  train {float(loss_val):.4e}  val {float(loss(params, vi_j)):.4e}")

    yv = np.sinh(np.asarray(mlp_apply(params, X[vi_j]))) * y_scale
    for c, name in enumerate(("Q_i", "Q_e", "Gamma_e")):
        t = T[vi][:, c]
        if np.abs(t).max() > 0:
            ss = 1 - np.sum((t - yv[:, c]) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-30)
            print(f"val R2 {name}: {ss:+.3f}")

    save_model(
        args.out, params,
        feature_names=names, target_names=("q_i", "q_e", "pfe"),
        x_mean=x_mean, x_std=x_std, y_transform="asinh", y_scale=y_scale,
        grid=grid, meta={"datasets": args.datasets, "hidden": args.hidden,
                         "epochs": args.epochs, "n_train": int(len(ti)), "n_val": int(n_val)},
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
