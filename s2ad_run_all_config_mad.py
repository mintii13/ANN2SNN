"""
s2ad_run_all_config.py - Run S2AD with multiple configurations
===============================================================
Run predefined configs for backbone, membrane, calibration, and batch size experiments.

Usage:
  python s2ad_run_all_config.py --wandb
  python s2ad_run_all_config.py --categories bottle leather --wandb
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
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

# MVTec categories (có thể giảm để test nhanh)
ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

# Categories for quick test (chạy nhanh hơn)
QUICK_CATEGORIES = ['bottle', 'leather', 'wood']

# Paths
DEFAULT_DATA_PATH = '/home/minhtringuyen/ANN2SNN/mvtec'
BASE_SAVE_DIR = './s2ad_results_config_mad'
S2AD_SCRIPT = 's2ad_validate.py'

# Common parameters
COMMON_ARGS = {
    'data_path': DEFAULT_DATA_PATH,
    'img_size': 256,
    'snn_mode': 'max',
    'timesteps': [4, 8, 16, 32, 64],
}


# ═══════════════════════════════════════════════════════════════════════════
# DEFINE ALL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS = []

# ========== NHÓM 1: Kiểm tra backbone (layer123, layer23, layer2, layer3) ==========
# membrane=False, calib=500
backbone_configs = [
    {'layers': 'layer123', 'use_membrane': False, 'calib_samples': 500, 'batch_size': 16, 'name': 'backbone_layer123_memFalse'},
    {'layers': 'layer23',  'use_membrane': False, 'calib_samples': 500, 'batch_size': 16, 'name': 'backbone_layer23_memFalse'},
    {'layers': 'layer12',   'use_membrane': False, 'calib_samples': 500, 'batch_size': 16, 'name': 'backbone_layer12_memFalse'},
]

# ========== NHÓM 2: Kiểm tra calibration samples ==========
calib_configs = [
    # calib=50'},
    {'layers': 'layer12',  'use_membrane': False,  'calib_samples': 50, 'batch_size': 16, 'name': 'calib50_layer12_memFalse'},
    {'layers': 'layer123', 'use_membrane': False, 'calib_samples': 50, 'batch_size': 16, 'name': 'calib50_layer123_memFalse'},
    {'layers': 'layer23',  'use_membrane': False, 'calib_samples': 50, 'batch_size': 16, 'name': 'calib50_layer23_memFalse'},
    # calib=100
    {'layers': 'layer12',  'use_membrane': False,  'calib_samples': 100, 'batch_size': 16, 'name': 'calib100_layer12_memFalse'},
    {'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'calib100_layer123_memFalse'},
    {'layers': 'layer23',  'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'calib100_layer23_memFalse'},
]


# Gộp tất cả configs
ALL_CONFIGS = backbone_configs + calib_configs


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_s2ad(category, config, args):
    """Run S2AD for a single category with given config."""
    
    # Tạo save_dir riêng cho từng config
    config_save_dir = os.path.join(BASE_SAVE_DIR, args.backbone, config['name'])  # Thêm args.backbone
    os.makedirs(config_save_dir, exist_ok=True)
    
    # Tạo run name
    run_name = f"{category}_{config['name']}_T{''.join(str(t) for t in COMMON_ARGS['timesteps'])}"
    
    cmd = [
        sys.executable, S2AD_SCRIPT,
        '--name', category,
        '--data_path', COMMON_ARGS['data_path'],
        '--backbone', args.backbone,
        '--layers', config['layers'],
        '--timesteps'] + [str(t) for t in COMMON_ARGS['timesteps']] + [
        '--img_size', str(COMMON_ARGS['img_size']),
        '--batch_size', str(config['batch_size']),
        '--calib_samples', str(config['calib_samples']),
        '--snn_mode', COMMON_ARGS['snn_mode'],
        '--save_dir', config_save_dir,
        '--combine_method', args.combine_method,
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
    print(f"Config: {config['name']}")
    print(f"Category: {category}")
    print(f"Save dir: {config_save_dir}")
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


def print_plan(categories, configs):
    """Print execution plan."""
    print("\n" + "=" * 70)
    print("EXECUTION PLAN")
    print("=" * 70)
    print(f"Categories: {len(categories)}")
    for i, cat in enumerate(categories):
        print(f"  {i+1}. {cat}")
    
    print(f"\nConfigurations: {len(configs)}")
    for i, cfg in enumerate(configs):
        print(f"  {i+1}. {cfg['name']}")
        print(f"     layers={cfg['layers']}, membrane={cfg['use_membrane']}, "
              f"calib={cfg['calib_samples']}, batch={cfg['batch_size']}")
    
    total_runs = len(categories) * len(configs)
    print(f"\nTotal runs: {total_runs}")
    print(f"Estimated time: ~{total_runs * 2 / 60:.1f} hours (assuming 2 min/run)")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='S2AD - Run multiple configurations')
    
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Categories to run (default: all 15 categories)')
    parser.add_argument('--quick', action='store_true',
                        help='Run only on quick test categories (bottle, leather, wood)')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        choices=['backbone', 'membrane', 'calib', 'batch', 'all'],
                        help='Which config groups to run (default: all)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture (default: resnet18)')
    
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='S2AD_mvtec_version2_resnet34')
    parser.add_argument('--wandb_offline', action='store_true')
    
    parser.add_argument('--skip_failed', action='store_true')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--combine_method', type=str, default='mad_weighted',
                    choices=['simple', 'mad_weighted'],
                    help='Combine method for multi-layer deviations')
    
    return parser.parse_args()

def parse_result_file(filepath):
    """Parse result file to extract AUCs for each timestep."""
    results = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    in_table = False
    for line in lines:
        line = line.strip()
        if 'Timestep' in line and 'Image AUC' in line:
            in_table = True
            continue
        if in_table and line.startswith('-'):
            continue
        if in_table and line and not line.startswith('='):
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    timestep = int(parts[0].strip())
                    img_auc = float(parts[1].strip())
                    pix_auc = float(parts[2].strip())
                    results[timestep] = {'img_auc': img_auc, 'pix_auc': pix_auc}
                except:
                    pass
            if line.startswith('='):
                break
    
    return results

def create_summary_csv(config_save_dir, config_name, categories, timesteps):
    """
    Create a single summary CSV file containing both Image AUC and Pixel AUC.
    Format: Category, T4_img, T4_px, T8_img, T8_px, T16_img, T16_px, ...
    """
    
    data = {'Category': []}
    
    # Tạo các cột: T{timestep}_img và T{timestep}_px
    for T in timesteps:
        data[f'T{T}_img'] = []
        data[f'T{T}_px'] = []
    
    for category in categories:
        result_file = os.path.join(config_save_dir, f'{category}_results.txt')
        if os.path.exists(result_file):
            results = parse_result_file(result_file)
            data['Category'].append(category)
            for T in timesteps:
                if T in results:
                    data[f'T{T}_img'].append(results[T]['img_auc'])
                    data[f'T{T}_px'].append(results[T]['pix_auc'])
                else:
                    data[f'T{T}_img'].append('')
                    data[f'T{T}_px'].append('')
    
    # Save single CSV file
    df = pd.DataFrame(data)
    csv_path = os.path.join(config_save_dir, f's2ad_summary_{config_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Summary CSV saved: {csv_path}")
    
    return df

def create_comparison_table(backbone_save_dir, backbone, timesteps, categories):
    """
    Create a comparison table of all configs with mean AUC values.
    Sorted by Pixel AUC at T=8 (descending).
    Each row: Config, T4_img, T4_px, T8_img, T8_px, ...
    """
    config_dirs = [d for d in os.listdir(backbone_save_dir) 
                   if os.path.isdir(os.path.join(backbone_save_dir, d))]
    
    results = []
    
    for config_name in config_dirs:
        config_path = os.path.join(backbone_save_dir, config_name)
        
        # Đọc từ summary CSV (1 file duy nhất)
        summary_csv = os.path.join(config_path, f's2ad_summary_{config_name}.csv')
        
        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv)
            
            mean_img = {}
            mean_pix = {}
            for T in timesteps:
                img_col = f'T{T}_img'
                pix_col = f'T{T}_px'
                if img_col in df.columns:
                    img_vals = df[img_col].dropna().values
                    pix_vals = df[pix_col].dropna().values
                    mean_img[T] = np.mean(img_vals) if len(img_vals) > 0 else 0
                    mean_pix[T] = np.mean(pix_vals) if len(pix_vals) > 0 else 0
                else:
                    mean_img[T] = 0
                    mean_pix[T] = 0
            
            results.append({
                'config': config_name,
                'mean_img': mean_img,
                'mean_pix': mean_pix,
            })
        else:
            print(f"  Warning: No summary CSV found for {config_name}")
    
    # Sắp xếp theo Pixel AUC tại T=8 (giảm dần)
    results.sort(key=lambda x: x['mean_pix'].get(8, 0), reverse=True)
    
    # Tạo DataFrame cho bảng so sánh
    comparison_data = []
    for res in results:
        row = {'Config': res['config'].replace(f'{backbone}_', '')}
        for T in timesteps:
            row[f'T{T}_img'] = round(res['mean_img'].get(T, 0), 4)
            row[f'T{T}_px'] = round(res['mean_pix'].get(T, 0), 4)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Lưu CSV
    csv_path = os.path.join(backbone_save_dir, f'comparison_table_{backbone}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Comparison table saved: {csv_path}")
    
    # In bảng ra console
    print("\n" + "=" * 100)
    print(f"COMPARISON TABLE - {backbone.upper()} (sorted by Pixel AUC at T=8)")
    print("=" * 100)
    
    # Format header
    header = f"{'Config':<35}"
    for T in timesteps:
        header += f" | T{T}_img  T{T}_px "
    print(header)
    print("-" * (35 + 15 * len(timesteps)))
    
    # Print each row
    for res in results:
        config_short = res['config'].replace(f'{backbone}_', '')
        row = f"{config_short:<35}"
        for T in timesteps:
            img_val = res['mean_img'].get(T, 0)
            pix_val = res['mean_pix'].get(T, 0)
            row += f" | {img_val:.4f} {pix_val:.4f}"
        print(row)
    
    print("=" * 100)
    
    return df


def create_best_config_summary(backbone_save_dir, backbone, timesteps):
    """
    Create a summary of best configs for each timestep.
    """
    config_dirs = [d for d in os.listdir(backbone_save_dir) 
                   if os.path.isdir(os.path.join(backbone_save_dir, d))]
    
    best_img = {}
    best_pix = {}
    
    for T in timesteps:
        best_img[T] = {'config': '', 'value': 0}
        best_pix[T] = {'config': '', 'value': 0}
    
    for config_name in config_dirs:
        config_path = os.path.join(backbone_save_dir, config_name)
        
        # Đọc từ summary CSV (1 file duy nhất)
        summary_csv = os.path.join(config_path, f's2ad_summary_{config_name}.csv')
        
        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv)
            
            for T in timesteps:
                img_col = f'T{T}_img'
                pix_col = f'T{T}_px'
                if img_col in df.columns and pix_col in df.columns:
                    img_mean = df[img_col].dropna().mean()
                    pix_mean = df[pix_col].dropna().mean()
                    
                    if img_mean > best_img[T]['value']:
                        best_img[T]['value'] = img_mean
                        best_img[T]['config'] = config_name
                    
                    if pix_mean > best_pix[T]['value']:
                        best_pix[T]['value'] = pix_mean
                        best_pix[T]['config'] = config_name
    
    # Create best config summary DataFrame
    best_data = []
    for T in timesteps:
        best_data.append({
            'Timestep': T,
            'Best Image AUC': best_img[T]['value'],
            'Best Image Config': best_img[T]['config'].replace(f'{backbone}_', ''),
            'Best Pixel AUC': best_pix[T]['value'],
            'Best Pixel Config': best_pix[T]['config'].replace(f'{backbone}_', ''),
        })
    
    df_best = pd.DataFrame(best_data)
    
    # Save CSV
    best_csv_path = os.path.join(backbone_save_dir, f'best_configs_{backbone}.csv')
    df_best.to_csv(best_csv_path, index=False)
    print(f"\n🏆 Best configs summary saved: {best_csv_path}")
    
    # Print to console
    print("\n" + "=" * 80)
    print(f"BEST CONFIGS FOR EACH TIMESTEP - {backbone.upper()}")
    print("=" * 80)
    print(df_best.to_string(index=False))
    print("=" * 80)
    
    return df_best

def main():
    args = parse_args()
    
    # Select categories
    if args.quick:
        categories = QUICK_CATEGORIES
        print(f"Using QUICK mode: {categories}")
    elif args.categories:
        categories = args.categories
    else:
        categories = ALL_CATEGORIES
    
    # Select configs
    selected_configs = []
    config_groups = args.configs if args.configs else ['all']
    
    if 'backbone' in config_groups or 'all' in config_groups:
        selected_configs.extend(backbone_configs)
    if 'calib' in config_groups or 'all' in config_groups:
        selected_configs.extend(calib_configs)

    
    # Print plan
    print_plan(categories, selected_configs)
    
    # Check if script exists
    if not os.path.exists(S2AD_SCRIPT):
        print(f"\nError: {S2AD_SCRIPT} not found!")
        sys.exit(1)
    
    # Create base save directory
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    
    # THÊM DÒNG NÀY - Định nghĩa timesteps
    timesteps = COMMON_ARGS['timesteps']
    
    # Track results
    all_results = {}
    successful = []
    failed = []
    
    total_runs = len(categories) * len(selected_configs)
    run_count = 0
    
    for config in selected_configs:
        config_name = config['name']
        config_save_dir = os.path.join(BASE_SAVE_DIR, args.backbone, config_name)
        os.makedirs(config_save_dir, exist_ok=True)
        all_results[config_name] = {}
        
        for category in categories:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] Running {category} with {config_name}")
            
            # Check if already completed (resume mode)
            result_file = os.path.join(config_save_dir, f'{category}_results.txt')
            
            if args.resume and os.path.exists(result_file):
                print(f"  SKIP: Already completed (result file exists)")
                all_results[config_name][category] = True
                successful.append(f"{config_name}/{category}")
                continue
            
            success = run_s2ad(category, config, args)
            
            if success:
                successful.append(f"{config_name}/{category}")
                all_results[config_name][category] = True
            else:
                failed.append(f"{config_name}/{category}")
                if not args.skip_failed:
                    print(f"\nStopping due to failure")
                    break
        print(f"\n📊 Creating summary CSV for {config_name}...")
        create_summary_csv(config_save_dir, config_name, categories, timesteps)
        
        if failed and not args.skip_failed:
            break
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Successful: {len(successful)} runs")
    print(f"Failed: {len(failed)} runs")
    
    if successful:
        print("\nSuccessful runs:")
        for s in successful[:20]:
            print(f"  {s}")
        if len(successful) > 20:
            print(f"  ... and {len(successful) - 20} more")
    
    if failed:
        print("\nFailed runs:")
        for f in failed:
            print(f"  {f}")
    
    print("\n" + "=" * 70)
    print("All results saved under:", BASE_SAVE_DIR)
    print("Each config has its own subdirectory:")
    for config in selected_configs:
        print(f"  - {config['name']}/")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("GENERATING COMPARISON TABLES")
    print("=" * 70)
    
    backbone_save_dir = os.path.join(BASE_SAVE_DIR, args.backbone)
    os.makedirs(backbone_save_dir, exist_ok=True)
    timesteps = COMMON_ARGS['timesteps']
    
    # Tạo bảng so sánh tất cả configs
    comparison_df = create_comparison_table(backbone_save_dir, args.backbone, timesteps, categories)
    
    # Tạo bảng best configs cho từng timestep
    best_df = create_best_config_summary(backbone_save_dir, args.backbone, timesteps)
    
    print("\n" + "=" * 70)
    print(f"All results saved under: {backbone_save_dir}")
    print("Files generated:")
    print(f"  - comparison_table_{args.backbone}.csv (all configs comparison)")
    print(f"  - best_configs_{args.backbone}.csv (best per timestep)")
    print("=" * 70)

if __name__ == '__main__':
    main()