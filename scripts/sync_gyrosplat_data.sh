#!/usr/bin/env bash
# idempotent local mirror of the gyrosplat latents + cache rebuild.
# re-run whenever more iteration dirs become readable (needs g+rX from the owner).
set -euo pipefail

SRC=${SRC:-/restricteddata/ukaea/gyrokinetics/gyrosplats/data}
DST=${DST:-/local00/bioinf/galletti/gyrosplats/data}
CACHE=${CACHE:-/local00/bioinf/galletti/gyrosplats_cache}

mkdir -p "$DST"
cp -n "$SRC/_train_trajs_valid.json" "$DST/" 2>/dev/null || true

n_ok=0
for d in "$SRC"/iteration_*; do
    name=$(basename "$d")
    if [ -r "$d/params.npy" ]; then
        rsync -a --ignore-existing "$d/" "$DST/$name/"
        n_ok=$((n_ok + 1))
    fi
done
echo "mirrored $n_ok readable trajectories -> $DST"

# rebuild the training cache over everything mirrored (refits channel stats)
cd "$(dirname "$0")/.."
python scripts/convert_gyrosplat_latents.py \
    --data "$DST" \
    --cache "$CACHE" \
    --trajs $(ls -d "$DST"/iteration_* | xargs -n1 basename) \
    --skip-unreadable
