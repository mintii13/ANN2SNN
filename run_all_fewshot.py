#!/usr/bin/env python
"""
run_all_fewshot.py - Run S2AD with few-shot calibration (1,2,4 samples)
=======================================================================
Run all backbone architectures with calibration samples 1,2,4.
Fixed settings: layers=layer123, combine=mad_weighted, snn_mode=max.

Usage:
  python run_all_fewshot.py --dataset mvtec --wandb
  python run_all_fewshot.py --dataset visa --wandb
"""

import argparse
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# MVTec categories (15)
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

# VisA categories (12)
VISA_CATEGORIES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]

# Paths
DEFAULT_DATA_ROOT = '/home/minhtringuyen/ANN2SNN/datasets'
BASE_SAVE_DIR = './s2ad_fewshot_results'
S2AD_SCRIPT = 's2ad_validate.py'

# Fixed common parameters
COMMON_ARGS = {
    'img_size': 256,
    'batch_size': 16,
    'timesteps': [4, 8, 16, 32, 64],
    'layers': 'layer123',
    'combine_method': 'mad_weighted',
    'snn_mode': 'max',
    'use_membrane': False,
}

# All backbones (same as arch_configs in original)
BACKBONE_LIST = [
    'resnet18', 'resnet34', 'resnet50',
    'wide_resnet50_2', 'wide_resnet101_2',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'alexnet',
    'mobilenet_v2', 'mobilenet_v3_large',
    'densenet121', 'densenet169'
]

# Few‑shot calibration samples
CALIB_VALUES = [1, 2, 4]

# Build list of configs: each backbone + calib value
def build_fewshot_configs():
    configs = []
    for backbone in BACKBONE_LIST:
        for calib in CALIB_VALUES:
            name = f"fewshot_{backbone}_calib{calib}"
            configs.append({
                'backbone': backbone,
                'calib_samples': calib,
                'name': name,
                'layers': COMMON_ARGS['layers'],
                'combine_method': COMMON_ARGS['combine_method'],
                'snn_mode': COMMON_ARGS['snn_mode'],
                'use_membrane': COMMON_ARGS['use_membrane'],
                'batch_size': COMMON_ARGS['batch_size'],
            })
    return configs

FEWSHOT_CONFIGS = build_fewshot_configs()


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_s2ad(category, config, args, dataset):
    """Run S2AD for a single category with given config."""
    data_path = args.data_root
    config_save_dir = os.path.join(BASE_SAVE_DIR, dataset, config['name'])
    os.makedirs(config_save_dir, exist_ok=True)

    run_name = f"{dataset}_{category}_{config['name']}_T{''.join(str(t) for t in COMMON_ARGS['timesteps'])}"

    cmd = [
        sys.executable, S2AD_SCRIPT,
        '--dataset', dataset,
        '--name', category,
        '--data_path', data_path,
        '--backbone', config['backbone'],
        '--layers', config['layers'],
        '--timesteps'] + [str(t) for t in COMMON_ARGS['timesteps']] + [
        '--img_size', str(COMMON_ARGS['img_size']),
        '--batch_size', str(config['batch_size']),
        '--calib_samples', str(config['calib_samples']),
        '--snn_mode', config['snn_mode'],
        '--save_dir', config_save_dir,
        '--combine_method', config['combine_method'],
    ]

    if config['use_membrane']:
        cmd.append('--use_membrane')

    if args.wandb:
        cmd.append('--wandb')
        cmd.extend(['--wandb_project', args.wandb_project])
        cmd.extend(['--wandb_run_name', run_name])
        if args.wandb_offline:
            cmd.append('--wandb_offline')

    print(f"\n{'='*70}")
    print(f"Running: {run_name}")
    print(f"Dataset: {dataset}, Category: {category}")
    print(f"Config: {config}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed_time = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✓ Completed {run_name} in {elapsed_time:.2f} seconds")
        return True
    else:
        print(f"\n✗ Failed {run_name} after {elapsed_time:.2f} seconds")
        return False


def get_categories(dataset):
    if dataset == 'mvtec':
        return MVTEC_CATEGORIES
    elif dataset == 'visa':
        return VISA_CATEGORIES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def print_plan(dataset, categories, configs):
    print("\n" + "=" * 70)
    print("EXECUTION PLAN (Few‑shot: 1,2,4 samples)")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Categories: {len(categories)}")
    for i, cat in enumerate(categories[:10]):
        print(f"  {i+1}. {cat}")
    if len(categories) > 10:
        print(f"  ... and {len(categories)-10} more")
    print(f"\nConfigurations: {len(configs)} (backbones × calib values)")
    total_runs = len(categories) * len(configs)
    print(f"\nTotal runs: {total_runs}")
    print(f"Estimated time: ~{total_runs * 2 / 60:.1f} hours (assuming 2 min/run)")
    print("=" * 70)


def parse_result_file(filepath):
    """Parse result file to extract metrics (6 metrics)."""
    results = {}
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
            if len(parts) >= 7:
                try:
                    t = int(parts[0].strip())
                    img_auc = float(parts[1].strip())
                    img_ap = float(parts[2].strip())
                    img_f1 = float(parts[3].strip())
                    pix_auc = float(parts[4].strip())
                    pix_f1 = float(parts[5].strip())
                    pro = float(parts[6].strip())
                    results[t] = {
                        'img_auc': img_auc,
                        'img_ap': img_ap,
                        'img_f1': img_f1,
                        'pix_auc': pix_auc,
                        'pix_f1': pix_f1,
                        'pro': pro
                    }
                except:
                    pass
            elif len(parts) >= 3:
                try:
                    t = int(parts[0].strip())
                    img_auc = float(parts[1].strip())
                    pix_auc = float(parts[2].strip())
                    results[t] = {
                        'img_auc': img_auc,
                        'img_ap': 0.0,
                        'img_f1': 0.0,
                        'pix_auc': pix_auc,
                        'pix_f1': 0.0,
                        'pro': 0.0
                    }
                except:
                    pass
            if line.startswith('='):
                break
    return results


def create_summary_csv(config_save_dir, config_name, categories, timesteps):
    """Create summary CSV with all metrics."""
    data = {'Category': []}
    for T in timesteps:
        data[f'T{T}_img_auc'] = []
        data[f'T{T}_img_ap'] = []
        data[f'T{T}_img_f1'] = []
        data[f'T{T}_pix_auc'] = []
        data[f'T{T}_pix_f1'] = []
        data[f'T{T}_pro'] = []

    for category in categories:
        result_file = os.path.join(config_save_dir, f'{category}_results.txt')
        if os.path.exists(result_file):
            results = parse_result_file(result_file)
            data['Category'].append(category)
            for T in timesteps:
                if T in results:
                    data[f'T{T}_img_auc'].append(results[T]['img_auc'])
                    data[f'T{T}_img_ap'].append(results[T]['img_ap'])
                    data[f'T{T}_img_f1'].append(results[T]['img_f1'])
                    data[f'T{T}_pix_auc'].append(results[T]['pix_auc'])
                    data[f'T{T}_pix_f1'].append(results[T]['pix_f1'])
                    data[f'T{T}_pro'].append(results[T]['pro'])
                else:
                    for col in [f'T{T}_img_auc', f'T{T}_img_ap', f'T{T}_img_f1',
                                f'T{T}_pix_auc', f'T{T}_pix_f1', f'T{T}_pro']:
                        data[col].append('')
    df = pd.DataFrame(data)
    csv_path = os.path.join(config_save_dir, f's2ad_summary_{config_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Summary CSV saved: {csv_path}")
    return df


def create_comparison_table(base_save_dir_dataset, dataset_name, timesteps, categories):
    """Create comparison table (mean of 6 metrics) for all configs."""
    config_dirs = [d for d in os.listdir(base_save_dir_dataset)
                   if os.path.isdir(os.path.join(base_save_dir_dataset, d))]
    results = []
    for config_name in config_dirs:
        config_path = os.path.join(base_save_dir_dataset, config_name)
        summary_csv = os.path.join(config_path, f's2ad_summary_{config_name}.csv')
        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv)
            mean_metrics = {}
            for T in timesteps:
                for metric in ['img_auc', 'img_ap', 'img_f1', 'pix_auc', 'pix_f1', 'pro']:
                    col = f'T{T}_{metric}'
                    if col in df.columns:
                        vals = df[col].dropna().values
                        mean_metrics[col] = np.mean(vals) if len(vals) > 0 else 0
                    else:
                        mean_metrics[col] = 0
            results.append({'config': config_name, 'metrics': mean_metrics})

    results.sort(key=lambda x: x['metrics'].get('T8_pix_auc', 0), reverse=True)
    comparison_data = []
    for res in results:
        row = {'Config': res['config']}
        for T in timesteps:
            row[f'T{T}_img_auc'] = round(res['metrics'].get(f'T{T}_img_auc', 0), 4)
            row[f'T{T}_img_ap']  = round(res['metrics'].get(f'T{T}_img_ap', 0), 4)
            row[f'T{T}_img_f1']  = round(res['metrics'].get(f'T{T}_img_f1', 0), 4)
            row[f'T{T}_pix_auc'] = round(res['metrics'].get(f'T{T}_pix_auc', 0), 4)
            row[f'T{T}_pix_f1']  = round(res['metrics'].get(f'T{T}_pix_f1', 0), 4)
            row[f'T{T}_pro']     = round(res['metrics'].get(f'T{T}_pro', 0), 4)
        comparison_data.append(row)
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(base_save_dir_dataset, f'comparison_table_fewshot_{dataset_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Comparison table (6 metrics) saved: {csv_path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='S2AD - Few‑shot (1,2,4 samples) benchmark')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'],
                        help='Dataset to run')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Categories to run (default: all)')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Root directory containing mvtec/ and visa/')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='S2AD_FewShot')
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--skip_failed', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine categories
    if args.categories:
        categories = args.categories
    else:
        categories = get_categories(args.dataset)

    selected_configs = FEWSHOT_CONFIGS

    print_plan(args.dataset, categories, selected_configs)

    if not os.path.exists(S2AD_SCRIPT):
        print(f"\nError: {S2AD_SCRIPT} not found!")
        sys.exit(1)

    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    timesteps = COMMON_ARGS['timesteps']

    base_save_dir_dataset = os.path.join(BASE_SAVE_DIR, args.dataset)
    os.makedirs(base_save_dir_dataset, exist_ok=True)

    successful = []
    failed = []
    total_runs = len(categories) * len(selected_configs)
    run_count = 0

    for config in selected_configs:
        config_name = config['name']
        config_save_dir = os.path.join(base_save_dir_dataset, config_name)
        os.makedirs(config_save_dir, exist_ok=True)

        for category in categories:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] Running {args.dataset}/{category} with {config_name}")

            result_file = os.path.join(config_save_dir, f'{category}_results.txt')
            if args.resume and os.path.exists(result_file):
                print(f"  SKIP: Already completed")
                successful.append(f"{config_name}/{category}")
                continue

            success = run_s2ad(category, config, args, args.dataset)
            if success:
                successful.append(f"{config_name}/{category}")
            else:
                failed.append(f"{config_name}/{category}")
                if not args.skip_failed:
                    print(f"\nStopping due to failure")
                    break

        print(f"\n📊 Creating summary CSV for {config_name}...")
        create_summary_csv(config_save_dir, config_name, categories, timesteps)

        if failed and not args.skip_failed:
            break

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Successful: {len(successful)} runs")
    print(f"Failed: {len(failed)} runs")
    if successful:
        print("\nSuccessful runs (first 20):")
        for s in successful[:20]:
            print(f"  {s}")
        if len(successful) > 20:
            print(f"  ... and {len(successful)-20} more")
    if failed:
        print("\nFailed runs:")
        for f in failed:
            print(f"  {f}")
    print(f"\nAll results saved under: {base_save_dir_dataset}")

    # Generate final comparison table
    print("\n" + "=" * 70)
    print("GENERATING FINAL COMPARISON TABLE")
    print("=" * 70)
    comparison_df = create_comparison_table(base_save_dir_dataset, args.dataset, timesteps, categories)
    print(f"\nComparison table saved: {comparison_df}")


if __name__ == '__main__':
    main()