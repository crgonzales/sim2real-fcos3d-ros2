#!/usr/bin/env bash
# Download and extract the nuScenes v1.0-mini split (~4 GB).
#
#   bash scripts/get_nuscenes_mini.sh [URL]
#
# The download URL is account-gated: register free at https://nuscenes.org,
# then Downloads -> Full dataset (v1.0) -> Mini, and copy the link. Pass it as
# the first argument or set NUSCENES_MINI_URL. The links are signed and expire,
# so this script does not hardcode one.
#
# Contributor: Carlos Gonzales
set -euo pipefail

WS=${WS:-/workspace}
DATA=$WS/data
DEST=$DATA/nuscenes
TGZ=$DATA/v1.0-mini.tgz
URL=${1:-${NUSCENES_MINI_URL:-}}

mkdir -p "$DEST"

if [ ! -s "$TGZ" ]; then
  [ -n "$URL" ] || { echo "No URL. Pass one or set NUSCENES_MINI_URL." >&2; exit 2; }
  echo "==> downloading nuScenes v1.0-mini"
  wget -q --show-progress -O "$TGZ" "$URL"
else
  echo "==> archive already present: $(du -h "$TGZ" | cut -f1)"
fi

echo "==> extracting"
# --no-same-owner is required. The archive records uid/gid 1035, and an
# unprivileged container cannot chown to it, so plain `tar -xzf` aborts with
#   tar: Cannot change ownership to uid 1035 ... Operation not permitted
#   tar: Exiting with failure status due to previous errors
# even though the file data extracts fine.
tar --no-same-owner -xzf "$TGZ" -C "$DEST"

# mmdet3d configs use a relative data_root of 'data/nuscenes/', and
# update_infos_to_v2 hardcodes './data/nuscenes', so the repo must see the
# dataset at that path.
if [ -d "$WS/mmdetection3d" ]; then
  mkdir -p "$WS/mmdetection3d/data"
  ln -sfn "$DEST" "$WS/mmdetection3d/data/nuscenes"
  echo "==> linked into mmdetection3d/data/nuscenes"
fi

echo "==> contents"
ls "$DEST"
du -sh "$DEST"

cat <<EOF

Next: generate the info files (run from the mmdetection3d repo root, whose
configs expect the relative data path):

  cd $WS/mmdetection3d
  PYTHONPATH=$WS/mmdetection3d python3 $WS/sim2real/scripts/prepare_nuscenes.py \\
      --root-path ./data/nuscenes --version v1.0-mini
EOF
