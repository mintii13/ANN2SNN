#!/usr/bin/env python
"""
s2ad_run_all_config.py - Run S2AD with multiple configurations
===============================================================
Run predefined configs for backbone, combine, calibration, architecture, snn_mode.

Usage:
  python s2ad_run_all_config.py --dataset mvtec --configs backbone --wandb
  python s2ad_run_all_config.py --dataset visa --configs combine calib --wandb
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
BASE_SAVE_DIR = './s2ad_results_config_mad'
S2AD_SCRIPT = 's2ad_validate.py'

# Common parameters (có thể override qua command line nếu muốn, nhưng để cứng)
COMMON_ARGS = {
    'img_size': 256,
    'batch_size': 16,
    'timesteps': [4, 8, 16, 32, 64],
}


# ═══════════════════════════════════════════════════════════════════════════
# DEFINE ALL CONFIGURATIONS (theo từng nhóm)
# ═══════════════════════════════════════════════════════════════════════════

# Nhóm 1: Kiểm tra backbone layers (use_membrane=False, calib=100, combine=mad_weighted)
backbone_configs = [
    {'layers': 'layer1',   'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'backbone_layer1'},
    {'layers': 'layer2',   'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'backbone_layer2'},
    {'layers': 'layer3',   'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'backbone_layer3'},
]

# Nhóm 2: Kiểm tra combine method trên các layer khác nhau
combine_configs = [
    # layer23
    {'combine_method': 'simple',      'layers': 'layer23', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'combine_simple_layer23'},
    {'combine_method': 'mad_weighted','layers': 'layer23', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'combine_mad_layer23'},
    # layer12
    {'combine_method': 'simple',      'layers': 'layer12', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'combine_simple_layer12'},
    {'combine_method': 'mad_weighted','layers': 'layer12', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'combine_mad_layer12'},
    # layer123
    {'combine_method': 'simple',      'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'combine_simple_layer123'},
    {'combine_method': 'mad_weighted','layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'combine_mad_layer123'},
]

# Nhóm 3: Kiểm tra snn_mode (dùng layer123, combine=mad_weighted, calib=100)
snnmode_configs = [
    {'snn_mode': 'max',   'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'snnmode_max_layer123'},
    {'snn_mode': '0.99',  'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'snnmode_099_layer123'},
]

# Nhóm 4: Kiểm tra calibration samples (dùng layer123, combine=mad_weighted)
calib_configs = [
    {'calib_samples': 10,   'layers': 'layer123', 'use_membrane': False, 'batch_size': 16, 'name': 'calib10_layer123'},
    {'calib_samples': 100,  'layers': 'layer123', 'use_membrane': False, 'batch_size': 16, 'name': 'calib100_layer123'},
    {'calib_samples': -1,   'layers': 'layer123', 'use_membrane': False, 'batch_size': 16, 'name': 'calib_all_layer123'},
]

# Nhóm 5: Kiểm tra backbone architecture (dùng layer123, combine=mad_weighted, calib=100)
arch_configs = [
    {'backbone': 'resnet18', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_resnet18_layer123'},
    {'backbone': 'resnet34', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_resnet34_layer123'},
    {'backbone': 'resnet50', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_resnet50_layer123'},
    {'backbone': 'wide_resnet50_2', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_wide_resnet50_layer123'},
    {'backbone': 'wide_resnet101_2', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_wide_resnet101_layer123'},
    {'backbone': 'vgg11', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_vgg11_layer123'},
    {'backbone': 'vgg13', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_vgg13_layer123'},
    {'backbone': 'vgg16', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_vgg16_layer123'},
    {'backbone': 'vgg19', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_vgg19_layer123'},
    {'backbone': 'alexnet', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_alexnet_layer123'},
    {'backbone': 'mobilenet_v2', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_mobilenet_v2_layer123'},
    {'backbone': 'mobilenet_v3_large', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_mobilenet_v3_layer123'},
    {'backbone': 'densenet121', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_densenet121_layer123'},
    {'backbone': 'densenet169', 'layers': 'layer123', 'use_membrane': False, 'calib_samples': 100, 'batch_size': 16, 'name': 'arch_densenet169_layer123'},
]

# Gộp tất cả configs theo nhóm (để dễ dàng chọn lọc)
ALL_CONFIG_GROUPS = {
    'backbone': backbone_configs,
    'combine': combine_configs,
    'calib': calib_configs,
    'arch': arch_configs,
    'snnmode': snnmode_configs,
}


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_s2ad(category, config, args, dataset):
    """Run S2AD for a single category with given config."""
    
    # Đường dẫn dữ liệu: args.data_root + dataset
    data_path = args.data_root
    
    # Tạo save_dir riêng cho từng config
    config_save_dir = os.path.join(BASE_SAVE_DIR, dataset, config['name'])
    os.makedirs(config_save_dir, exist_ok=True)
    
    # Tạo run name
    run_name = f"{dataset}_{category}_{config['name']}_T{''.join(str(t) for t in COMMON_ARGS['timesteps'])}"
    
    # Xử lý calib_samples = -1 (nghĩa là dùng toàn bộ train set)
    calib_samples = config.get('calib_samples', 100)
    
    # Xây dựng command
    cmd = [
        sys.executable, S2AD_SCRIPT,
        '--dataset', dataset,
        '--name', category,
        '--data_path', data_path,
        '--backbone', config.get('backbone', args.backbone),   # ưu tiên config, fallback args.backbone
        '--layers', config['layers'],
        '--timesteps'] + [str(t) for t in COMMON_ARGS['timesteps']] + [
        '--img_size', str(COMMON_ARGS['img_size']),
        '--batch_size', str(config.get('batch_size', COMMON_ARGS['batch_size'])),
        '--calib_samples', str(calib_samples),
        '--snn_mode', config.get('snn_mode', 'max'),
        '--save_dir', config_save_dir,
        '--combine_method', config.get('combine_method', args.combine_method),
    ]
    
    if config.get('use_membrane', False):
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
    """Print execution plan."""
    print("\n" + "=" * 70)
    print("EXECUTION PLAN")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Categories: {len(categories)}")
    for i, cat in enumerate(categories[:10]):
        print(f"  {i+1}. {cat}")
    if len(categories) > 10:
        print(f"  ... and {len(categories)-10} more")
    
    print(f"\nConfigurations: {len(configs)}")
    for i, cfg in enumerate(configs):
        print(f"  {i+1}. {cfg['name']}")
        # In các tham số chính
        info = []
        if 'layers' in cfg: info.append(f"layers={cfg['layers']}")
        if 'combine_method' in cfg: info.append(f"combine={cfg['combine_method']}")
        if 'calib_samples' in cfg: info.append(f"calib={cfg['calib_samples']}")
        if 'backbone' in cfg: info.append(f"arch={cfg['backbone']}")
        if 'snn_mode' in cfg: info.append(f"snn={cfg['snn_mode']}")
        print(f"       ({', '.join(info)})")
    
    total_runs = len(categories) * len(configs)
    print(f"\nTotal runs: {total_runs}")
    print(f"Estimated time: ~{total_runs * 2 / 60:.1f} hours (assuming 2 min/run)")
    print("=" * 70)


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
    """Create a single summary CSV file containing both Image AUC and Pixel AUC."""
    data = {'Category': []}
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
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(config_save_dir, f's2ad_summary_{config_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Summary CSV saved: {csv_path}")
    return df


def create_comparison_table(backbone_save_dir, backbone, timesteps, categories):
    """Create a comparison table of all configs with mean AUC values."""
    config_dirs = [d for d in os.listdir(backbone_save_dir) 
                   if os.path.isdir(os.path.join(backbone_save_dir, d))]
    results = []
    
    for config_name in config_dirs:
        config_path = os.path.join(backbone_save_dir, config_name)
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
            results.append({'config': config_name, 'mean_img': mean_img, 'mean_pix': mean_pix})
    
    results.sort(key=lambda x: x['mean_pix'].get(8, 0), reverse=True)
    comparison_data = []
    for res in results:
        row = {'Config': res['config'].replace(f'{backbone}_', '')}
        for T in timesteps:
            row[f'T{T}_img'] = round(res['mean_img'].get(T, 0), 4)
            row[f'T{T}_px'] = round(res['mean_pix'].get(T, 0), 4)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(backbone_save_dir, f'comparison_table_{backbone}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Comparison table saved: {csv_path}")
    return df


def create_best_config_summary(backbone_save_dir, backbone, timesteps):
    """Create a summary of best configs for each timestep."""
    config_dirs = [d for d in os.listdir(backbone_save_dir) 
                   if os.path.isdir(os.path.join(backbone_save_dir, d))]
    best_img = {T: {'config': '', 'value': 0} for T in timesteps}
    best_pix = {T: {'config': '', 'value': 0} for T in timesteps}
    
    for config_name in config_dirs:
        config_path = os.path.join(backbone_save_dir, config_name)
        summary_csv = os.path.join(config_path, f's2ad_summary_{config_name}.csv')
        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv)
            for T in timesteps:
                img_col = f'T{T}_img'
                pix_col = f'T{T}_px'
                if img_col in df.columns:
                    img_mean = df[img_col].dropna().mean()
                    pix_mean = df[pix_col].dropna().mean()
                    if img_mean > best_img[T]['value']:
                        best_img[T]['value'] = img_mean
                        best_img[T]['config'] = config_name
                    if pix_mean > best_pix[T]['value']:
                        best_pix[T]['value'] = pix_mean
                        best_pix[T]['config'] = config_name
    
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
    best_csv_path = os.path.join(backbone_save_dir, f'best_configs_{backbone}.csv')
    df_best.to_csv(best_csv_path, index=False)
    print(f"\n🏆 Best configs summary saved: {best_csv_path}")
    return df_best


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='S2AD - Run multiple configurations')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'],
                        help='Dataset to run')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Categories to run (default: all categories of the dataset)')
    parser.add_argument('--configs', type=str, nargs='+', default=['backbone'],
                        choices=['backbone', 'combine', 'calib', 'arch', 'snnmode', 'all'],
                        help='Which config groups to run')
    parser.add_argument('--backbone', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2',
                             'vgg11', 'vgg13', 'vgg16', 'vgg19', 'alexnet',
                             'mobilenet_v2', 'mobilenet_v3_large', 'densenet121', 'densenet169'],
                    help='Backbone architecture')
    parser.add_argument('--combine_method', type=str, default='mad_weighted',
                        choices=['simple', 'mad_weighted'],
                        help='Default combine method (used when not specified in config)')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Root directory containing mvtec/ and visa/ subfolders')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='S2AD_Config_Exp')
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--skip_failed', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Xác định categories
    if args.categories:
        categories = args.categories
    else:
        categories = get_categories(args.dataset)
    
    # Chọn các config dựa trên nhóm
    selected_configs = []
    config_groups = args.configs if args.configs else ['backbone']
    if 'all' in config_groups:
        config_groups = ['backbone', 'combine', 'calib', 'arch', 'snnmode']
    
    for grp in config_groups:
        if grp in ALL_CONFIG_GROUPS:
            selected_configs.extend(ALL_CONFIG_GROUPS[grp])
    
    # In kế hoạch
    print_plan(args.dataset, categories, selected_configs)
    
    # Kiểm tra script tồn tại
    if not os.path.exists(S2AD_SCRIPT):
        print(f"\nError: {S2AD_SCRIPT} not found!")
        sys.exit(1)
    
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    timesteps = COMMON_ARGS['timesteps']
    
    # Thư mục lưu kết quả theo dataset và backbone mặc định (args.backbone)
    # Lưu ý: đối với các config có thay đổi backbone, sẽ được lưu trong thư mục con riêng của config đó,
    # nhưng file comparison_table vẫn được gộp chung dưới thư mục backbone mặc định? Hơi phức tạp.
    # Để đơn giản, ta sẽ lưu tất cả kết quả dưới BASE_SAVE_DIR/dataset/ (không phân biệt backbone mặc định),
    # vì mỗi config đã có tên riêng (bao gồm cả thông tin backbone nếu có).
    base_save_dir_dataset = os.path.join(BASE_SAVE_DIR, args.dataset)
    os.makedirs(base_save_dir_dataset, exist_ok=True)
    
    successful = []
    failed = []
    total_runs = len(categories) * len(selected_configs)
    run_count = 0
    
    for config in selected_configs:
        config_name = config['name']
        # Thư mục lưu của config này: base_save_dir_dataset/config_name
        config_save_dir = os.path.join(base_save_dir_dataset, config_name)
        os.makedirs(config_save_dir, exist_ok=True)
        
        for category in categories:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] Running {args.dataset}/{category} with {config_name}")
            
            # Kiểm tra resume
            result_file = os.path.join(config_save_dir, f'{category}_results.txt')
            if args.resume and os.path.exists(result_file):
                print(f"  SKIP: Already completed (result file exists)")
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
        
        # Sau khi chạy xong tất cả categories cho config này, tạo summary CSV
        print(f"\n📊 Creating summary CSV for {config_name}...")
        create_summary_csv(config_save_dir, config_name, categories, timesteps)
        
        if failed and not args.skip_failed:
            break
    
    # In tổng kết
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
    
    # Tạo bảng so sánh tổng hợp cho tất cả các config đã chạy
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON TABLES")
    print("=" * 70)
    # Bảng so sánh sẽ được tạo trong thư mục base_save_dir_dataset
    comparison_df = create_comparison_table(base_save_dir_dataset, args.dataset, timesteps, categories)
    best_df = create_best_config_summary(base_save_dir_dataset, args.dataset, timesteps)
    
    print(f"\nAll comparison files saved under: {base_save_dir_dataset}")
    print("  - comparison_table_{dataset}.csv")
    print("  - best_configs_{dataset}.csv")
    print("=" * 70)


if __name__ == '__main__':
    main()