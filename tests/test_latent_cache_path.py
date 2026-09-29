"""Latent cache naming matches torch and is independent of how the AE checkpoint is given."""

from __future__ import annotations

from types import SimpleNamespace

from neugk_jax.diffusion.latents import latent_cache_path


def test_cache_path_dir_and_file_agree(tmp_path):
    run = tmp_path / "20260405_022851_327"
    run.mkdir()
    (run / "best.pth").write_bytes(b"")
    ds = SimpleNamespace(files=["/d/iteration_1_ifft_realpotens", "/d/iteration_0_ifft_realpotens"],
                         offset=80, path=str(tmp_path))
    by_dir = latent_cache_path(ds, "train", run, decouple_mu=True)
    by_file = latent_cache_path(ds, "train", run / "best.pth", decouple_mu=True)
    assert by_dir == by_file and by_dir.parent == tmp_path
    assert by_dir.name.startswith("diff_train_latents_offset80_mu_")
    assert by_dir.name.endswith("_latents_ae327.pkl")
