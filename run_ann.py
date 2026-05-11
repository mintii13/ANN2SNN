#!/usr/bin/env python
"""
run_ann.py - Run A2AD (ANN-based anomaly detection) with multiple configs
and generate comparison table (7 metrics) across configs.

Usage:
  python run_ann.py --dataset mvtec --configs all --save_anomaly_maps --resume
  python run_ann.py --dataset visa --backbone vgg16 --layers layer123 --wandb
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
A2AD_SCRIPT = 'a2ad_validate.py'  # ANN version script
DEFAULT_DATA_ROOT = '/home/minhtringuyen/ANN2SNN/datasets'
DEFAULT_SAVE_DIR = './a2ad_results'
DEFAULT_MAPS_ROOT = './anomaly_maps_a2ad'

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

# Common parameters for ANN (no timesteps, no snn_mode)
COMMON_ARGS = {
    'img_size': 256,
    'batch_size': 16,
    'calib_samples': 500,
    'backbone': 'vgg16',
    'layers': 'layer123',
    'combine_method': 'mad_weighted',
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
    """Parse result file to extract metrics (7 metrics) for ANN (single result)."""
    results = {}
    if not os.path.exists(filepath):
        return results
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Format: "Image AUC: 0.xxxx", "Pixel AUC: 0.xxxx", etc.
    for line in lines:
        line = line.strip()
        if line.startswith('Image AUC:'):
            parts = line.split(':')
            if len(parts) >= 2:
                results['img_auc'] = float(parts[1].strip())
        elif line.startswith('Image AP:'):
            parts = line.split(':')
            if len(parts) >= 2:
                results['img_ap'] = float(parts[1].strip())
        elif line.startswith('Image F1:'):
            parts = line.split(':')
            if len(parts) >= 2:
                results['img_f1'] = float(parts[1].strip())
        elif line.startswith('Pixel AUC:'):
            parts = line.split(':')
            if len(parts) >= 2:
                results['pix_auc'] = float(parts[1].strip())
        elif line.startswith('Pixel AP:'):
            parts = line.split(':')
            if len(parts) >= 2:
                results['pix_ap'] = float(parts[1].strip())
        elif line.startswith('Pixel F1:'):
            parts = line.split(':')
            if len(parts) >= 2:
                results['pix_f1'] = float(parts[1].strip())
        elif line.startswith('PRO:'):
            parts = line.split(':')
            if len(parts) >= 2:
                results['pro'] = float(parts[1].strip())
    
    # Nếu không có trong file text thì thử parse theo format cũ (bảng)
    if not results:
        # Fallback: parse bảng nếu có (dù ANN chỉ có một kết quả)
        in_table = False
        for line in lines:
            line = line.strip()
            if 'Image AUC' in line and 'Pixel AUC' in line:
                in_table = True
                continue
            if in_table and line and not line.startswith('-') and not line.startswith('='):
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        img_auc = float(parts[1].strip())
                        pix_auc = float(parts[2].strip())
                        results['img_auc'] = img_auc
                        results['pix_auc'] = pix_auc
                    except:
                        pass
                if len(parts) >= 8:
                    try:
                        results['img_auc'] = float(parts[1].strip())
                        results['img_ap'] = float(parts[2].strip())
                        results['img_f1'] = float(parts[3].strip())
                        results['pix_auc'] = float(parts[4].strip())
                        results['pix_ap'] = float(parts[5].strip())
                        results['pix_f1'] = float(parts[6].strip())
                        results['pro'] = float(parts[7].strip())
                    except:
                        pass
                break
    return results

def create_summary_csv(config_save_dir, config_name, categories):
    """Create a single summary CSV file for a specific config containing all 7 metrics."""
    data = {'Category': []}
    metric_names = ['img_auc', 'img_ap', 'img_f1', 'pix_auc', 'pix_ap', 'pix_f1', 'pro']
    for metric in metric_names:
        data[metric] = []
    
    for category in categories:
        result_file = os.path.join(config_save_dir, f'{category}_results.txt')
        if os.path.exists(result_file):
            results = parse_result_file(result_file)
            data['Category'].append(category)
            for metric in metric_names:
                data[metric].append(results.get(metric, ''))
        else:
            # Bỏ qua category không có file
            pass
    
    if not data['Category']:
        print(f"  No result files found for {config_name}")
        return None
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(config_save_dir, f'a2ad_summary_{config_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Summary CSV saved: {csv_path}")
    return df

def create_comparison_table(base_save_dir_dataset, dataset_name, configs_info):
    """Create a comparison table of all configs with mean of 7 metrics."""
    config_dirs = [d for d in os.listdir(base_save_dir_dataset) 
                   if os.path.isdir(os.path.join(base_save_dir_dataset, d))]
    results = []
    
    metric_names = ['img_auc', 'img_ap', 'img_f1', 'pix_auc', 'pix_ap', 'pix_f1', 'pro']
    
    for config_dir in config_dirs:
        config_path = os.path.join(base_save_dir_dataset, config_dir)
        summary_csv = os.path.join(config_path, f'a2ad_summary_{config_dir}.csv')
        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv)
            mean_metrics = {}
            for metric in metric_names:
                if metric in df.columns:
                    vals = df[metric].dropna().values
                    mean_metrics[metric] = np.mean(vals) if len(vals) > 0 else 0
                else:
                    mean_metrics[metric] = 0
            results.append({'config': config_dir, 'metrics': mean_metrics})
    
    if not results:
        print("  No summary CSV files found. Comparison table not created.")
        return None
    
    # Sắp xếp theo tên config
    results.sort(key=lambda x: x['config'])
    
    # Tạo DataFrame
    comparison_data = []
    for res in results:
        row = {'Config': res['config']}
        for metric in metric_names:
            row[metric] = round(res['metrics'].get(metric, 0), 4)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(base_save_dir_dataset, f'comparison_table_{dataset_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Comparison table (7 metrics) saved: {csv_path}")
    return df

def run_a2ad(category, config, args):
    """Run A2AD for a single category with given config."""
    config_name = config['name']
    save_dir = os.path.join(args.save_dir, args.dataset, config_name)
    os.makedirs(save_dir, exist_ok=True)
    
    result_file = os.path.join(save_dir, f'{category}_results.txt')
    if args.resume and os.path.exists(result_file):
        print(f"  SKIP: {category} already completed")
        return True
    
    run_name = f"{args.dataset}_{category}_{config_name}"
    
    cmd = [
        sys.executable, A2AD_SCRIPT,
        '--dataset', args.dataset,
        '--name', category,
        '--data_path', args.data_root,
        '--backbone', config.get('backbone', COMMON_ARGS['backbone']),
        '--layers', config.get('layers', COMMON_ARGS['layers']),
        '--img_size', str(COMMON_ARGS['img_size']),
        '--batch_size', str(config.get('batch_size', COMMON_ARGS['batch_size'])),
        '--calib_samples', str(config.get('calib_samples', COMMON_ARGS['calib_samples'])),
        '--save_dir', save_dir,
        '--combine_method', config.get('combine_method', COMMON_ARGS['combine_method']),
    ]
    
    if args.save_anomaly_maps:
        cmd.append('--save_anomaly_maps')
        maps_root = os.path.join(args.maps_root, args.dataset)
        cmd.extend(['--maps_root', maps_root])
    
    if args.wandb:
        cmd.append('--wandb')
        cmd.append('--wandb_project')
        cmd.append(args.wandb_project)
        cmd.append('--wandb_run_name')
        cmd.append(run_name)
        if args.wandb_offline:
            cmd.append('--wandb_offline')
    
    print(f"\n  Running: {category} (config={config_name})")
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
    parser = argparse.ArgumentParser(description='Run A2AD (ANN) with multiple configs and generate comparison table')
    parser.add_argument('--dataset', type=str, required=True, choices=['mvtec', 'visa'],
                        help='Dataset to run')
    parser.add_argument('--configs', type=str, nargs='+', default=['all'],
                        choices=['all', 'backbone', 'layers', 'combine', 'custom'],
                        help='Which config groups to run (all runs all predefined configs)')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Specific backbone for custom config')
    parser.add_argument('--layers', type=str, default=None,
                        help='Specific layers for custom config')
    parser.add_argument('--combine_method', type=str, default=None,
                        help='Specific combine method for custom config')
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
    parser.add_argument('--wandb_project', type=str, default='A2AD_Config_Exp')
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Skip existing results')
    return parser.parse_args()

def get_configs(args):
    """Return list of configs based on user selection."""
    configs = []
    
    # Kiểm tra nếu có tham số custom (--backbone, --layers, --combine_method)
    if args.configs == ['custom'] or (args.backbone is not None or args.layers is not None or args.combine_method is not None):
        config_name = []
        backbone = args.backbone if args.backbone else COMMON_ARGS['backbone']
        layers = args.layers if args.layers else COMMON_ARGS['layers']
        combine = args.combine_method if args.combine_method else COMMON_ARGS['combine_method']
        config_name.append(backbone)
        config_name.append(layers)
        config_name.append(combine)
        config_name = '_'.join(config_name)
        configs.append({
            'name': config_name,
            'backbone': backbone,
            'layers': layers,
            'combine_method': combine,
        })
    else:
        # Predefined configs based on groups
        if 'all' in args.configs or 'backbone' in args.configs:
            # Các backbone khác nhau
            backbones = ['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2',
                         'vgg11', 'vgg13', 'vgg16', 'alexnet', 'mobilenet_v2', 'densenet121']
            for bb in backbones:
                configs.append({
                    'name': f'{bb}_{COMMON_ARGS["layers"]}_{COMMON_ARGS["combine_method"]}',
                    'backbone': bb,
                    'layers': COMMON_ARGS['layers'],
                    'combine_method': COMMON_ARGS['combine_method'],
                })
        
        if 'all' in args.configs or 'layers' in args.configs:
            layers_list = ['layer1', 'layer2', 'layer3', 'layer12', 'layer23', 'layer123']
            for l in layers_list:
                configs.append({
                    'name': f'{COMMON_ARGS["backbone"]}_{l}_{COMMON_ARGS["combine_method"]}',
                    'backbone': COMMON_ARGS['backbone'],
                    'layers': l,
                    'combine_method': COMMON_ARGS['combine_method'],
                })
        
        if 'all' in args.configs or 'combine' in args.configs:
            combine_list = ['simple', 'mad_weighted']
            for cm in combine_list:
                configs.append({
                    'name': f'{COMMON_ARGS["backbone"]}_{COMMON_ARGS["layers"]}_{cm}',
                    'backbone': COMMON_ARGS['backbone'],
                    'layers': COMMON_ARGS['layers'],
                    'combine_method': cm,
                })
        
        # Remove duplicates
        unique_configs = []
        seen = set()
        for cfg in configs:
            key = (cfg['backbone'], cfg['layers'], cfg['combine_method'])
            if key not in seen:
                seen.add(key)
                unique_configs.append(cfg)
        configs = unique_configs
    
    return configs

def main():
    args = parse_args()
    
    categories = args.categories if args.categories else get_categories(args.dataset)
    configs = get_configs(args)
    
    print("=" * 70)
    print("A2AD - ANN-based Anomaly Detection (multiple configs experiment)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Categories: {len(categories)}")
    print(f"  Configurations: {len(configs)}")
    for i, cfg in enumerate(configs):
        print(f"    {i+1}. {cfg['name']}")
    if args.save_anomaly_maps:
        print(f"  Save anomaly maps: YES (root: {args.maps_root}/{args.dataset})")
    print("=" * 70)
    
    total_runs = len(categories) * len(configs)
    run_count = 0
    successful = []
    failed = []
    
    base_save_dir_dataset = os.path.join(args.save_dir, args.dataset)
    os.makedirs(base_save_dir_dataset, exist_ok=True)
    
    for config in configs:
        config_name = config['name']
        config_save_dir = os.path.join(base_save_dir_dataset, config_name)
        os.makedirs(config_save_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Processing config: {config_name}")
        print(f"{'='*70}")
        
        for category in categories:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {category}")
            
            success = run_a2ad(category, config, args)
            if success:
                successful.append(f"{config_name}/{category}")
            else:
                failed.append(f"{config_name}/{category}")
                if not args.resume:
                    print("Stopping due to failure")
                    break
        
        print(f"\n📊 Creating summary CSV for {config_name}...")
        create_summary_csv(config_save_dir, config_name, categories)
        
        if failed and not args.resume:
            break
    
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON TABLE")
    print("=" * 70)
    comparison_df = create_comparison_table(base_save_dir_dataset, args.dataset, configs)
    
    if comparison_df is not None:
        print(f"\nAll comparison files saved under: {base_save_dir_dataset}")
        print(f"  - comparison_table_{args.dataset}.csv (7 metrics)")
    
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