#!/usr/bin/env python3
"""Generate the mmdet3d info .pkl files for nuScenes, for FCOS3D evaluation.

Why not `tools/create_data.py`
------------------------------
That script imports lyft_dataset_sdk and waymo converters at module import
time, so it cannot run unless those are installed. We deliberately skipped
them (mmdet3d was installed with --no-deps to avoid open3d/Flask/blinker),
and installing lyft-dataset-sdk risks disturbing the numpy-1.x pinned stack.

So we call the nuScenes converter directly and reproduce exactly what
`nuscenes_data_prep` does, minus `create_groundtruth_database` -- that step
only builds a sampling database for training-time augmentation and is not
used by evaluation.

Usage (on the pod):
    PYTHONPATH=/workspace/mmdetection3d python3 scripts/prepare_nuscenes.py \
        --root-path /workspace/data/nuscenes \
        --version   v1.0-mini

Contributor: Carlos Gonzales
"""

import argparse
import os.path as osp
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root-path', required=True)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--info-prefix', default='nuscenes')
    ap.add_argument('--max-sweeps', type=int, default=10)
    args = ap.parse_args()

    out_dir = args.out_dir or args.root_path

    from tools.dataset_converters import nuscenes_converter
    from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

    print(f'[1/2] create_nuscenes_infos  version={args.version}')
    nuscenes_converter.create_nuscenes_infos(
        args.root_path,
        args.info_prefix,
        version=args.version,
        max_sweeps=args.max_sweeps)

    # v1.0-test has no val split; mini and trainval both produce train+val.
    splits = ['test'] if args.version == 'v1.0-test' else ['train', 'val']
    for split in splits:
        pkl = osp.join(out_dir, f'{args.info_prefix}_infos_{split}.pkl')
        if not osp.exists(pkl):
            print(f'  !! missing {pkl}', file=sys.stderr)
            continue
        print(f'[2/2] update_pkl_infos -> {osp.basename(pkl)}')
        update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=pkl)

    print('\nInfo files ready:')
    for split in splits:
        pkl = osp.join(out_dir, f'{args.info_prefix}_infos_{split}.pkl')
        if osp.exists(pkl):
            print(f'  {pkl}  ({osp.getsize(pkl)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
