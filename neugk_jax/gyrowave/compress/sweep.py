"""Compress a list of trajectories (both versions) in ONE process — the JIT fit graph
is built once and reused, avoiding per-process recompilation. Resumable: skips caches
that already exist. Usage: python -m neugk_jax.gyrowave.compress.sweep <traj_json> <out_dir> [versions]"""
import os, sys, json, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # grow on-demand -> share GPUs, no preallocation
import jax                                                        # persistent compile cache: compile the
jax.config.update("jax_compilation_cache_dir", "/local00/bioinf/galletti/jax_cache")  # heavy fit graph ONCE,
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)  # reuse across shards/trajectories instead
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # of recompiling per process.
from neugk_jax.gyrowave.compress import process_trajectory as PT

traj_json = sys.argv[1]
out_dir = sys.argv[2]
versions = sys.argv[3].split(",") if len(sys.argv) > 3 else ["real", "semispectral"]
shard_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
n_shards = int(sys.argv[5]) if len(sys.argv) > 5 else 1
trajs = json.load(open(traj_json))[shard_idx::n_shards]        # this shard's slice
os.makedirs(out_dir, exist_ok=True)
print(f"sweep shard {shard_idx}/{n_shards}: {len(trajs)} trajs x {versions} -> {out_dir}", flush=True)
for it in trajs:
    for ver in versions:
        out = os.path.join(out_dir, f"tokens_iteration_{it}_{ver}.npz")
        if os.path.exists(out):
            print(f"skip iteration_{it} {ver} (exists)", flush=True)
            continue
        t0 = time.time()
        try:
            res = PT.run(ver, f"iteration_{it}", "per_snapshot", (1.0, 0.0, 0.0), out, verbose=False)
            summ = res[0] if isinstance(res, tuple) else res
            summ = summ if isinstance(summ, dict) else {}
            print(f"DONE iteration_{it} {ver}  df={summ.get('df_psnr', float('nan')):.1f} "
                  f"flux={summ.get('flux_relerr', float('nan')):.3f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"FAIL iteration_{it} {ver}: {e}", flush=True)
print("sweep complete", flush=True)
