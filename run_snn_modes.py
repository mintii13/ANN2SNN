#!/usr/bin/env python
"""
run_snn_modes.py - Run S2AD with multiple snn_modes for VGG16 layer123
and generate comparison table (7 metrics + mAD) across modes.

Usage:
  python run_snn_modes.py --dataset mvtec --modes 0.78 0.75 0.73 0.7 --save_anomaly_maps --resume
  python run_snn_modes.py --dataset visa --modes 0.78 0.75 0.73 0.7 --wandb
"""

import argparse
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
S2AD_SCRIPT = 's2ad_validate.py'
DEFAULT_DATA_ROOT = '/home/minhtringuyen/ANN2SNN/datasets'
DEFAULT_SAVE_DIR = './s2ad_results_snn_modes'
DEFAULT_MAPS_ROOT = './anomaly_maps_snn_modes'

# MVTec categories
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

# VisA categories
VISA_CATEGORIES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]

# Common parameters
COMMON_ARGS = {
    'img_size': 256,
    'batch_size': 16,
    'calib_samples': -1,
    'timesteps': [4, 8, 16, 32, 64],
    'backbone': 'vgg16',
    'layers': 'layer123',
    'combine_method': 'mad_weighted',
    'use_membrane': False,
}

# ============================================================
# FUNCTIONS
# ============================================================
def get_categories(dataset):
    if dataset == 'mvtec':
        return MVTEC_CATEGORIES
    elif dataset == 'visa':
        return VISA_CATEGORIES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def parse_result_file(filepath):
    """Parse result file to extract metrics for each timestep (7 metrics)."""
    results = {}
    if not os.path.exists(filepath):
        return results
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    in_table = False
    for line in lines:
        line = line.strip()
        if 'Timestep' in line and 'Img AUC' in line:
            in_table = True
            continue
        if in_table and line.startswith('-'):
            continue
        if in_table and line and not line.startswith('='):
            parts = line.split('|')
            if len(parts) >= 8:
                try:
                    t = int(parts[0].strip())
                    img_auc = float(parts[1].strip())
                    img_ap = float(parts[2].strip())
                    img_f1 = float(parts[3].strip())
                    pix_auc = float(parts[4].strip())
                    pix_ap = float(parts[5].strip())
                    pix_f1 = float(parts[6].strip())
                    pro = float(parts[7].strip())
                    results[t] = {
                        'img_auc': img_auc,
                        'img_ap': img_ap,
                        'img_f1': img_f1,
                        'pix_auc': pix_auc,
                        'pix_ap': pix_ap,
                        'pix_f1': pix_f1,
                        'pro': pro
                    }
                except Exception as e:
                    pass
            if line.startswith('='):
                break
    return results

def create_summary_csv(config_save_dir, config_name, categories, timesteps):
    """Create a single summary CSV file for a specific config (mode) containing all 7 metrics."""
    data = {'Category': []}
    for T in timesteps:
        data[f'T{T}_img_auc'] = []
        data[f'T{T}_img_ap'] = []
        data[f'T{T}_img_f1'] = []
        data[f'T{T}_pix_auc'] = []
        data[f'T{T}_pix_ap'] = []
        data[f'T{T}_pix_f1'] = []
        data[f'T{T}_pro'] = []
    
    for category in categories:
        result_file = os.path.join(config_save_dir, f'{category}_results.txt')
        if os.path.exists(result_file):
            results = parse_result_file(result_file)
            data['Category'].append(category)
            for T in timesteps:
                if T in results:
                    r = results[T]
                    data[f'T{T}_img_auc'].append(r['img_auc'])
                    data[f'T{T}_img_ap'].append(r['img_ap'])
                    data[f'T{T}_img_f1'].append(r['img_f1'])
                    data[f'T{T}_pix_auc'].append(r['pix_auc'])
                    data[f'T{T}_pix_ap'].append(r['pix_ap'])
                    data[f'T{T}_pix_f1'].append(r['pix_f1'])
                    data[f'T{T}_pro'].append(r['pro'])
                else:
                    data[f'T{T}_img_auc'].append('')
                    data[f'T{T}_img_ap'].append('')
                    data[f'T{T}_img_f1'].append('')
                    data[f'T{T}_pix_auc'].append('')
                    data[f'T{T}_pix_ap'].append('')
                    data[f'T{T}_pix_f1'].append('')
                    data[f'T{T}_pro'].append('')
        else:
            # Nếu không có file, bỏ qua category (không thêm vào data)
            pass
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(config_save_dir, f's2ad_summary_{config_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Summary CSV saved: {csv_path}")
    return df

def create_comparison_table(base_save_dir_dataset, dataset_name, timesteps, categories):
    """Create a comparison table of all configs (modes) with mean of 7 metrics + mAD."""
    config_dirs = [d for d in os.listdir(base_save_dir_dataset) 
                   if os.path.isdir(os.path.join(base_save_dir_dataset, d)) and d.startswith('snn_')]
    results = []
    
    metric_names = ['img_auc', 'img_ap', 'img_f1', 'pix_auc', 'pix_ap', 'pix_f1', 'pro']
    mad_metrics = ['img_auc', 'img_ap', 'img_f1', 'pix_auc', 'pix_f1', 'pro']  # bỏ pix_ap
    
    for config_dir in config_dirs:
        config_path = os.path.join(base_save_dir_dataset, config_dir)
        summary_csv = os.path.join(config_path, f's2ad_summary_{config_dir}.csv')
        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv)
            mean_metrics = {}
            for T in timesteps:
                for metric in metric_names:
                    col = f'T{T}_{metric}'
                    if col in df.columns:
                        vals = df[col].dropna().values
                        mean_metrics[col] = np.mean(vals) if len(vals) > 0 else 0
                    else:
                        mean_metrics[col] = 0
                mad_vals = [mean_metrics[f'T{T}_{m}'] for m in mad_metrics]
                mean_metrics[f'T{T}_mAD'] = np.mean(mad_vals)
            results.append({'config': config_dir, 'metrics': mean_metrics})
    
    if not results:
        print("  No summary CSV files found. Comparison table not created.")
        return None
    
    results.sort(key=lambda x: x['config'])
    
    comparison_data = []
    for res in results:
        row = {'Config': res['config'].replace('snn_', '').replace('_', '.')}
        for T in timesteps:
            row[f'T{T}_img_auc'] = round(res['metrics'].get(f'T{T}_img_auc', 0), 4)
            row[f'T{T}_img_ap']  = round(res['metrics'].get(f'T{T}_img_ap', 0), 4)
            row[f'T{T}_img_f1']  = round(res['metrics'].get(f'T{T}_img_f1', 0), 4)
            row[f'T{T}_pix_auc'] = round(res['metrics'].get(f'T{T}_pix_auc', 0), 4)
            row[f'T{T}_pix_ap']  = round(res['metrics'].get(f'T{T}_pix_ap', 0), 4)
            row[f'T{T}_pix_f1']  = round(res['metrics'].get(f'T{T}_pix_f1', 0), 4)
            row[f'T{T}_pro']     = round(res['metrics'].get(f'T{T}_pro', 0), 4)
            row[f'T{T}_mAD']     = round(res['metrics'].get(f'T{T}_mAD', 0), 4)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(base_save_dir_dataset, f'comparison_table_{dataset_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Comparison table (7 metrics + mAD) saved: {csv_path}")
    return df

def run_s2ad(category, snn_mode, args):
    """Run S2AD for a single category with given snn_mode."""
    mode_str = str(snn_mode).replace('.', '_')
    save_dir = os.path.join(args.save_dir, args.dataset, f'snn_{mode_str}')
    os.makedirs(save_dir, exist_ok=True)
    
    result_file = os.path.join(save_dir, f'{category}_results.txt')
    if args.resume and os.path.exists(result_file):
        print(f"  SKIP: {category} already completed")
        return True
    
    run_name = f"{args.dataset}_{category}_snn{mode_str}"
    
    cmd = [
        sys.executable, S2AD_SCRIPT,
        '--dataset', args.dataset,
        '--name', category,
        '--data_path', args.data_root,
        '--backbone', COMMON_ARGS['backbone'],
        '--layers', COMMON_ARGS['layers'],
        '--timesteps'] + [str(t) for t in COMMON_ARGS['timesteps']] + [
        '--img_size', str(COMMON_ARGS['img_size']),
        '--batch_size', str(COMMON_ARGS['batch_size']),
        '--calib_samples', str(COMMON_ARGS['calib_samples']),
        '--snn_mode', str(snn_mode),
        '--save_dir', save_dir,
        '--combine_method', COMMON_ARGS['combine_method'],
    ]
    
    if COMMON_ARGS['use_membrane']:
        cmd.append('--use_membrane')
    
    # --- Thêm phần lưu anomaly maps ---
    if args.save_anomaly_maps:
        cmd.append('--save_anomaly_maps')
        # Thêm cấp dataset vào maps_root để tách riêng mvtec và visa
        maps_root = os.path.join(args.maps_root, args.dataset)
        cmd.extend(['--maps_root', maps_root])
    # ---------------------------------
    
    if args.wandb:
        cmd.append('--wandb')
        cmd.append('--wandb_project')
        cmd.append(args.wandb_project)
        cmd.append('--wandb_run_name')
        cmd.append(run_name)
        if args.wandb_offline:
            cmd.append('--wandb_offline')
    
    print(f"\n  Running: {category} (snn_mode={snn_mode})")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"  ✓ Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"  ✗ Failed after {elapsed:.1f}s")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Run S2AD with multiple snn_modes and generate comparison table')
    parser.add_argument('--dataset', type=str, required=True, choices=['mvtec', 'visa'],
                        help='Dataset to run')
    parser.add_argument('--modes', type=float, nargs='+', required=True,
                        help='snn_mode values to test (e.g., 0.78 0.75 0.73 0.7)')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Root directory containing mvtec/ and visa/ subfolders')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                        help='Base directory to save results (metrics)')
    parser.add_argument('--maps_root', type=str, default=DEFAULT_MAPS_ROOT,
                        help='Root directory to save anomaly maps (will be extended with dataset name)')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Specific categories (default: all)')
    parser.add_argument('--save_anomaly_maps', action='store_true',
                        help='Save anomaly maps for each test image')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='S2AD_SNN_Modes')
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Skip existing results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    categories = args.categories if args.categories else get_categories(args.dataset)
    timesteps = COMMON_ARGS['timesteps']
    
    print("=" * 70)
    print("S2AD - Multiple snn_modes Experiment")
    print(f"  Dataset: {args.dataset}")
    print(f"  Categories: {len(categories)}")
    print(f"  snn_modes: {args.modes}")
    print(f"  Backbone: {COMMON_ARGS['backbone']}")
    print(f"  Layers: {COMMON_ARGS['layers']}")
    print(f"  Combine: {COMMON_ARGS['combine_method']}")
    print(f"  Timesteps: {timesteps}")
    if args.save_anomaly_maps:
        print(f"  Save anomaly maps: YES (root: {args.maps_root}/{args.dataset})")
    print("=" * 70)
    
    total_runs = len(categories) * len(args.modes)
    run_count = 0
    successful = []
    failed = []
    
    base_save_dir_dataset = os.path.join(args.save_dir, args.dataset)
    os.makedirs(base_save_dir_dataset, exist_ok=True)
    
    for snn_mode in args.modes:
        mode_str = str(snn_mode).replace('.', '_')
        config_name = f'snn_{mode_str}'
        config_save_dir = os.path.join(base_save_dir_dataset, config_name)
        os.makedirs(config_save_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Processing snn_mode = {snn_mode}")
        print(f"{'='*70}")
        
        for category in categories:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {category}")
            
            success = run_s2ad(category, snn_mode, args)
            if success:
                successful.append(f"{config_name}/{category}")
            else:
                failed.append(f"{config_name}/{category}")
                if not args.resume:
                    print("Stopping due to failure")
                    break
        
        print(f"\n📊 Creating summary CSV for {config_name}...")
        create_summary_csv(config_save_dir, config_name, categories, timesteps)
        
        if failed and not args.resume:
            break
    
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON TABLE")
    print("=" * 70)
    comparison_df = create_comparison_table(base_save_dir_dataset, args.dataset, timesteps, categories)
    
    if comparison_df is not None:
        print(f"\nAll comparison files saved under: {base_save_dir_dataset}")
        print(f"  - comparison_table_{args.dataset}.csv (7 metrics + mAD)")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful runs: {len(successful)}")
    print(f"Failed runs: {len(failed)}")
    if failed:
        print("Failed runs:")
        for f in failed:
            print(f"  {f}")
    print(f"\nResults saved under: {base_save_dir_dataset}")
    if args.save_anomaly_maps:
        print(f"Anomaly maps saved under: {os.path.join(args.maps_root, args.dataset)}")
    print("=" * 70)

if __name__ == '__main__':
    main()