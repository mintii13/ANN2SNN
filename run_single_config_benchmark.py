#!/usr/bin/env python
"""
Run a single S2AD configuration on all categories of a dataset and print summary.
Usage:
  python run_single_config_benchmark.py --dataset mvtec --backbone vgg11 --layers layer123 --snn_mode 0.99 --calib_samples 10 --timesteps 4 8 16 32 64
"""

import argparse
import subprocess
import sys
import os
import pandas as pd
from io import StringIO

# Các categories
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

VISA_CATEGORIES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]

def run_s2ad(category, dataset, args):
    """Run s2ad_validate.py for one category and return the results dict."""
    cmd = [
        sys.executable, 's2ad_validate.py',
        '--dataset', dataset,
        '--name', category,
        '--data_path', args.data_root,
        '--backbone', args.backbone,
        '--layers', args.layers,
        '--timesteps'] + [str(t) for t in args.timesteps] + [
        '--img_size', str(args.img_size),
        '--batch_size', str(args.batch_size),
        '--calib_samples', str(args.calib_samples),
        '--snn_mode', args.snn_mode,
        '--combine_method', args.combine_method,
        '--save_dir', './temp_results',
    ]
    if args.use_membrane:
        cmd.append('--use_membrane')
    
    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse AUC from stdout (last lines of summary)
    lines = result.stdout.split('\n')
    # Find the summary table
    in_table = False
    results = {}
    for line in lines:
        if 'Timestep' in line and 'Image AUC' in line:
            in_table = True
            continue
        if in_table and line.startswith('-'):
            continue
        if in_table and line.strip() and not line.startswith('='):
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    t = int(parts[0].strip())
                    img = float(parts[1].strip())
                    pix = float(parts[2].strip())
                    results[t] = {'img': img, 'pix': pix}
                except:
                    pass
            if line.startswith('='):
                break
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['mvtec', 'visa'])
    parser.add_argument('--data_root', default='/home/minhtringuyen/ANN2SNN/datasets')
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--layers', required=True)
    parser.add_argument('--timesteps', type=int, nargs='+', default=[4,8,16,32,64])
    parser.add_argument('--calib_samples', type=int, default=100)
    parser.add_argument('--snn_mode', default='max')
    parser.add_argument('--combine_method', default='mad_weighted')
    parser.add_argument('--use_membrane', action='store_true')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    categories = MVTEC_CATEGORIES if args.dataset == 'mvtec' else VISA_CATEGORIES
    
    all_results = {}
    for cat in categories:
        print(f"Running {cat}...")
        res = run_s2ad(cat, args.dataset, args)
        all_results[cat] = res
    
    # Build dataframes for image and pixel AUC
    timesteps = args.timesteps
    img_data = {t: [] for t in timesteps}
    pix_data = {t: [] for t in timesteps}
    
    for cat in categories:
        for t in timesteps:
            if t in all_results[cat]:
                img_data[t].append(all_results[cat][t]['img'])
                pix_data[t].append(all_results[cat][t]['pix'])
            else:
                img_data[t].append(float('nan'))
                pix_data[t].append(float('nan'))
    
    # Create summary dataframe
    df_img = pd.DataFrame(img_data, index=categories)
    df_pix = pd.DataFrame(pix_data, index=categories)
    
    # Add mean row
    df_img.loc['MEAN'] = df_img.mean(axis=0)
    df_pix.loc['MEAN'] = df_pix.mean(axis=0)
    
    print("\n" + "="*80)
    print(f"RESULTS for {args.backbone}, layers={args.layers}, combine={args.combine_method}, snn_mode={args.snn_mode}, calib={args.calib_samples}")
    print("="*80)
    print("\nIMAGE AUC (per category):")
    print(df_img.round(4).to_string())
    print("\nPIXEL AUC (per category):")
    print(df_pix.round(4).to_string())
    
    # Save to CSV
    os.makedirs('benchmark_results', exist_ok=True)
    img_csv = f"benchmark_results/{args.dataset}_{args.backbone}_{args.layers}_img.csv"
    pix_csv = f"benchmark_results/{args.dataset}_{args.backbone}_{args.layers}_pix.csv"
    df_img.to_csv(img_csv)
    df_pix.to_csv(pix_csv)
    print(f"\nSaved to {img_csv} and {pix_csv}")

if __name__ == '__main__':
    main()