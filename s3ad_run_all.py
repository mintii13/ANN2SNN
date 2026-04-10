"""
s3ad_run_all.py — Chạy S3AD trên tất cả 15 categories MVTec
============================================================
Dùng sau khi validate hypothesis trên 1-2 categories.

Chạy:
  python s3ad_run_all.py --data_path /home/minhtringuyen/ANN2SNN/mvtec
  python s3ad_run_all.py --data_path /path/to/mvtec --timesteps 8 16
"""

import subprocess
import sys
import os
import time
import argparse

MVTEC_CATEGORIES = [
    'leather', 'wood', 'tile', 'carpet', 'grid',
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str,
                   default='/home/minhtringuyen/ANN2SNN/mvtec')
    p.add_argument('--timesteps', type=int, nargs='+', default=[4, 8, 16, 32])
    p.add_argument('--save_dir',  type=str, default='./s3ad_results')
    p.add_argument('--categories', type=str, nargs='+',
                   default=MVTEC_CATEGORIES,
                   help='Subset of categories to run')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("S3AD — Full MVTec evaluation")
    print(f"Categories: {args.categories}")
    print(f"Timesteps:  {args.timesteps}")
    print("=" * 60)

    results_summary = []
    for i, cat in enumerate(args.categories):
        print(f"\n[{i+1}/{len(args.categories)}] {cat}")
        start = time.time()
        cmd = [
            sys.executable, 's3ad_validate.py',
            '--name', cat,
            '--data_path', args.data_path,
            '--timesteps', *[str(t) for t in args.timesteps],
            '--save_dir', args.save_dir,
        ]
        ret = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - start
        status = 'OK' if ret.returncode == 0 else 'FAILED'
        results_summary.append((cat, status, elapsed))
        print(f"  {cat}: {status} ({elapsed/60:.1f} min)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cat, status, elapsed in results_summary:
        print(f"  {cat:<15} {status}  ({elapsed/60:.1f} min)")
    print(f"\nAll results saved in: {args.save_dir}/")


if __name__ == '__main__':
    main()