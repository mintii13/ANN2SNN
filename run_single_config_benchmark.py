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
import shutil
from tqdm import tqdm
import time

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
    # Tạo thư mục tạm thời cho từng category để tránh xung đột
    timestamp = int(time.time())
    temp_dir = f'./temp_results_{category}_{timestamp}'
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
        '--save_dir', temp_dir,
    ]
    if args.use_membrane:
        cmd.append('--use_membrane')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse output
        lines = result.stdout.split('\n')
        in_table = False
        results = {}
        for line in lines:
            if 'Timestep' in line and 'Img AUC' in line:
                in_table = True
                continue
            if in_table and line.startswith('-'):
                continue
            if in_table and line.strip() and not line.startswith('='):
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
                        results[t] = {'img_auc': img_auc, 'img_ap': img_ap, 'img_f1': img_f1,
                                      'pix_auc': pix_auc, 'pix_f1': pix_f1, 'pro': pro}
                    except:
                        pass
                elif len(parts) >= 3:  # fallback nếu dùng bảng cũ
                    try:
                        t = int(parts[0].strip())
                        img_auc = float(parts[1].strip())
                        pix_auc = float(parts[2].strip())
                        results[t] = {'img_auc': img_auc, 'img_ap': 0.0, 'img_f1': 0.0,
                                      'pix_auc': pix_auc, 'pix_f1': 0.0, 'pro': 0.0}
                    except:
                        pass
                if line.startswith('='):
                    break
    except subprocess.CalledProcessError as e:
        print(f"Error running {category}: {e}")
        print(f"STDERR: {e.stderr}")
        return {}
    finally:
        # Xóa thư mục tạm
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
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
    for cat in tqdm(categories, desc="Processing categories"):
        res = run_s2ad(cat, args.dataset, args)
        all_results[cat] = res
    
    # Tạo DataFrame cho image AUC, pixel AUC, v.v.
    timesteps = args.timesteps
    img_auc_data = {t: [] for t in timesteps}
    img_ap_data = {t: [] for t in timesteps}
    img_f1_data = {t: [] for t in timesteps}
    pix_auc_data = {t: [] for t in timesteps}
    pix_f1_data = {t: [] for t in timesteps}
    pro_data = {t: [] for t in timesteps}
    
    valid_cats = []
    for cat in categories:
        if not all_results[cat]:
            continue
        valid_cats.append(cat)
        for t in timesteps:
            if t in all_results[cat]:
                img_auc_data[t].append(all_results[cat][t]['img_auc'])
                img_ap_data[t].append(all_results[cat][t]['img_ap'])
                img_f1_data[t].append(all_results[cat][t]['img_f1'])
                pix_auc_data[t].append(all_results[cat][t]['pix_auc'])
                pix_f1_data[t].append(all_results[cat][t]['pix_f1'])
                pro_data[t].append(all_results[cat][t]['pro'])
            else:
                # Nếu thiếu, thêm NaN
                img_auc_data[t].append(float('nan'))
                img_ap_data[t].append(float('nan'))
                img_f1_data[t].append(float('nan'))
                pix_auc_data[t].append(float('nan'))
                pix_f1_data[t].append(float('nan'))
                pro_data[t].append(float('nan'))
    
    # Tạo DataFrame với index là category
    df_img_auc = pd.DataFrame(img_auc_data, index=valid_cats)
    df_img_ap = pd.DataFrame(img_ap_data, index=valid_cats)
    df_img_f1 = pd.DataFrame(img_f1_data, index=valid_cats)
    df_pix_auc = pd.DataFrame(pix_auc_data, index=valid_cats)
    df_pix_f1 = pd.DataFrame(pix_f1_data, index=valid_cats)
    df_pro = pd.DataFrame(pro_data, index=valid_cats)
    
    # Thêm dòng MEAN
    df_img_auc.loc['MEAN'] = df_img_auc.mean(axis=0)
    df_img_ap.loc['MEAN'] = df_img_ap.mean(axis=0)
    df_img_f1.loc['MEAN'] = df_img_f1.mean(axis=0)
    df_pix_auc.loc['MEAN'] = df_pix_auc.mean(axis=0)
    df_pix_f1.loc['MEAN'] = df_pix_f1.mean(axis=0)
    df_pro.loc['MEAN'] = df_pro.mean(axis=0)
    
    # In kết quả
    print("\n" + "="*80)
    print(f"RESULTS for {args.backbone}, layers={args.layers}, combine={args.combine_method}, snn_mode={args.snn_mode}, calib={args.calib_samples}")
    print("="*80)
    print("\nIMAGE AUC (per category):")
    print(df_img_auc.round(4).to_string())
    print("\nIMAGE AUPR (per category):")
    print(df_img_ap.round(4).to_string())
    print("\nIMAGE F1 (per category):")
    print(df_img_f1.round(4).to_string())
    print("\nPIXEL AUC (per category):")
    print(df_pix_auc.round(4).to_string())
    print("\nPIXEL F1 (per category):")
    print(df_pix_f1.round(4).to_string())
    print("\nPRO (per category):")
    print(df_pro.round(4).to_string())
    
    # Lưu CSV
    os.makedirs('benchmark_results', exist_ok=True)
    df_img_auc.to_csv(f"benchmark_results/{args.dataset}_{args.backbone}_{args.layers}_img_auc.csv")
    df_img_ap.to_csv(f"benchmark_results/{args.dataset}_{args.backbone}_{args.layers}_img_ap.csv")
    df_img_f1.to_csv(f"benchmark_results/{args.dataset}_{args.backbone}_{args.layers}_img_f1.csv")
    df_pix_auc.to_csv(f"benchmark_results/{args.dataset}_{args.backbone}_{args.layers}_pix_auc.csv")
    df_pix_f1.to_csv(f"benchmark_results/{args.dataset}_{args.backbone}_{args.layers}_pix_f1.csv")
    df_pro.to_csv(f"benchmark_results/{args.dataset}_{args.backbone}_{args.layers}_pro.csv")
    
    print(f"\nSaved CSVs to benchmark_results/")

if __name__ == '__main__':
    main()