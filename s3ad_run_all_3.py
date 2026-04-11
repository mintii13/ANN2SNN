"""
s3ad_run_all.py - Chạy S3AD trên tất cả 15 categories MVTec
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
import re
from pathlib import Path

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
    p.add_argument('--save_dir',  type=str, default='./s3ad_results_layer3')
    p.add_argument('--categories', type=str, nargs='+',
                   default=MVTEC_CATEGORIES,
                   help='Subset of categories to run')
    return p.parse_args()


def parse_result_file(result_path, timesteps):
    """
    Parse file results.txt để lấy AUROC cho từng timestep
    Trả về dict: {timestep: (img_auc, pix_auc)}
    """
    results = {}
    if not os.path.exists(result_path):
        return results
    
    with open(result_path, 'r') as f:
        content = f.read()
    
    # Tìm bảng kết quả trong file
    # Format: "   T |  Img AUC |  Pix AUC"
    lines = content.split('\n')
    in_table = False
    for line in lines:
        if 'T | Img AUC | Pix AUC' in line or 'T |  Img AUC |  Pix AUC' in line:
            in_table = True
            continue
        if in_table and line.strip().startswith('-'):
            continue
        if in_table and line.strip():
            # Parse dòng như: "   4 |   0.8923 |   0.8745"
            match = re.match(r'\s*(\d+)\s*\|\s*([\d\.]+|N/A)\s*\|\s*([\d\.]+|N/A)', line)
            if match:
                T = int(match.group(1))
                img_auc = float(match.group(2)) if match.group(2) != 'N/A' else None
                pix_auc = float(match.group(3)) if match.group(3) != 'N/A' else None
                results[T] = (img_auc, pix_auc)
        elif in_table and not line.strip():
            break
    
    return results


def create_summary_csv(results_dir, timesteps, categories):
    """
    Tạo file CSV tổng hợp kết quả
    """
    csv_path = os.path.join(results_dir, 'summary_all_categories.csv')
    
    # Chuẩn bị headers
    headers = ['Category']
    for T in timesteps:
        headers.append(f'Img_AUC_T{T}')
        headers.append(f'Pix_AUC_T{T}')
    
    # Thu thập dữ liệu
    rows = []
    for cat in categories:
        result_file = os.path.join(results_dir, f'{cat}_results.txt')
        results = parse_result_file(result_file, timesteps)
        
        row = [cat]
        for T in timesteps:
            if T in results:
                img_auc, pix_auc = results[T]
                row.append(f"{img_auc:.4f}" if img_auc is not None else "N/A")
                row.append(f"{pix_auc:.4f}" if pix_auc is not None else "N/A")
            else:
                row.append("N/A")
                row.append("N/A")
        rows.append(row)
    
    # Ghi CSV
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"\n✓ Summary CSV saved: {csv_path}")
    return csv_path


def create_summary_markdown(results_dir, timesteps, categories):
    """
    Tạo file Markdown tổng hợp kết quả (dễ đọc và copy vào sheet)
    """
    md_path = os.path.join(results_dir, 'summary_all_categories.md')
    
    with open(md_path, 'w') as f:
        f.write("# S3AD Results Summary\n\n")
        f.write(f"**Categories:** {len(categories)}\n\n")
        f.write(f"**Timesteps:** {timesteps}\n\n")
        
        # Bảng Image AUROC
        f.write("## Image AUROC\n\n")
        f.write("| Category | " + " | ".join([f"T={T}" for T in timesteps]) + " |\n")
        f.write("|" + "---|" * (len(timesteps) + 1) + "\n")
        
        for cat in categories:
            result_file = os.path.join(results_dir, f'{cat}_results.txt')
            results = parse_result_file(result_file, timesteps)
            
            row = [f"**{cat}**"]
            for T in timesteps:
                if T in results:
                    img_auc, _ = results[T]
                    row.append(f"{img_auc:.4f}" if img_auc is not None else "N/A")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
        
        # Bảng Pixel AUROC
        f.write("\n## Pixel AUROC\n\n")
        f.write("| Category | " + " | ".join([f"T={T}" for T in timesteps]) + " |\n")
        f.write("|" + "---|" * (len(timesteps) + 1) + "\n")
        
        for cat in categories:
            result_file = os.path.join(results_dir, f'{cat}_results.txt')
            results = parse_result_file(result_file, timesteps)
            
            row = [f"**{cat}**"]
            for T in timesteps:
                if T in results:
                    _, pix_auc = results[T]
                    row.append(f"{pix_auc:.4f}" if pix_auc is not None else "N/A")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
        
        # Bảng Best per category
        f.write("\n## Best Results per Category\n\n")
        f.write("| Category | Best T (Image) | Best Img AUC | Best T (Pixel) | Best Pix AUC |\n")
        f.write("|" + "---|" * 5 + "\n")
        
        for cat in categories:
            result_file = os.path.join(results_dir, f'{cat}_results.txt')
            results = parse_result_file(result_file, timesteps)
            
            best_img_T = None
            best_img_auc = -1
            best_pix_T = None
            best_pix_auc = -1
            
            for T, (img_auc, pix_auc) in results.items():
                if img_auc is not None and img_auc > best_img_auc:
                    best_img_auc = img_auc
                    best_img_T = T
                if pix_auc is not None and pix_auc > best_pix_auc:
                    best_pix_auc = pix_auc
                    best_pix_T = T
            
            f.write(f"| {cat} | {best_img_T} | {best_img_auc:.4f} | {best_pix_T} | {best_pix_auc:.4f} |\n")
        
        # Statistics
        f.write("\n## Statistics (Average over categories)\n\n")
        
        avg_img = {T: [] for T in timesteps}
        avg_pix = {T: [] for T in timesteps}
        
        for cat in categories:
            result_file = os.path.join(results_dir, f'{cat}_results.txt')
            results = parse_result_file(result_file, timesteps)
            for T in timesteps:
                if T in results:
                    img_auc, pix_auc = results[T]
                    if img_auc is not None:
                        avg_img[T].append(img_auc)
                    if pix_auc is not None:
                        avg_pix[T].append(pix_auc)
        
        f.write("| Timestep | Mean Img AUC | Std Img AUC | Mean Pix AUC | Std Pix AUC |\n")
        f.write("|" + "---|" * 5 + "\n")
        
        for T in timesteps:
            mean_img = np.mean(avg_img[T]) if avg_img[T] else 0
            std_img = np.std(avg_img[T]) if avg_img[T] else 0
            mean_pix = np.mean(avg_pix[T]) if avg_pix[T] else 0
            std_pix = np.std(avg_pix[T]) if avg_pix[T] else 0
            f.write(f"| T={T} | {mean_img:.4f} | {std_img:.4f} | {mean_pix:.4f} | {std_pix:.4f} |\n")
    
    print(f"✓ Summary Markdown saved: {md_path}")
    return md_path


def create_summary_excel(results_dir, timesteps, categories):
    """
    Tạo file Excel tổng hợp kết quả (dễ copy vào sheet)
    """
    try:
        import pandas as pd
        
        excel_path = os.path.join(results_dir, 'summary_all_categories.xlsx')
        
        # Thu thập dữ liệu
        data_img = {'Category': categories}
        data_pix = {'Category': categories}
        
        for T in timesteps:
            data_img[f'T={T}'] = []
            data_pix[f'T={T}'] = []
        
        for cat in categories:
            result_file = os.path.join(results_dir, f'{cat}_results.txt')
            results = parse_result_file(result_file, timesteps)
            
            for T in timesteps:
                if T in results:
                    img_auc, pix_auc = results[T]
                    data_img[f'T={T}'].append(img_auc if img_auc is not None else None)
                    data_pix[f'T={T}'].append(pix_auc if pix_auc is not None else None)
                else:
                    data_img[f'T={T}'].append(None)
                    data_pix[f'T={T}'].append(None)
        
        # Tạo DataFrame
        df_img = pd.DataFrame(data_img)
        df_pix = pd.DataFrame(data_pix)
        
        # Ghi Excel
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_img.to_excel(writer, sheet_name='Image AUROC', index=False)
            df_pix.to_excel(writer, sheet_name='Pixel AUROC', index=False)
            
            # Thêm sheet best results
            best_data = []
            for cat in categories:
                result_file = os.path.join(results_dir, f'{cat}_results.txt')
                results = parse_result_file(result_file, timesteps)
                
                best_img_T = None
                best_img_auc = -1
                best_pix_T = None
                best_pix_auc = -1
                
                for T, (img_auc, pix_auc) in results.items():
                    if img_auc is not None and img_auc > best_img_auc:
                        best_img_auc = img_auc
                        best_img_T = T
                    if pix_auc is not None and pix_auc > best_pix_auc:
                        best_pix_auc = pix_auc
                        best_pix_T = T
                
                best_data.append({
                    'Category': cat,
                    'Best T (Image)': best_img_T,
                    'Best Img AUC': best_img_auc,
                    'Best T (Pixel)': best_pix_T,
                    'Best Pix AUC': best_pix_auc
                })
            
            df_best = pd.DataFrame(best_data)
            df_best.to_excel(writer, sheet_name='Best per Category', index=False)
        
        print(f"✓ Summary Excel saved: {excel_path}")
        return excel_path
        
    except ImportError:
        print("⚠ Pandas or openpyxl not installed. Skipping Excel export.")
        return None


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Import numpy for statistics
    global np
    import numpy as np

    print("=" * 60)
    print("S3AD - Full MVTec evaluation")
    print(f"Categories: {args.categories}")
    print(f"Timesteps:  {args.timesteps}")
    print("=" * 60)

    results_summary = []
    for i, cat in enumerate(args.categories):
        print(f"\n[{i+1}/{len(args.categories)}] {cat}")
        start = time.time()
        cmd = [
            sys.executable, 's3ad_validate_3.py',
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
    
    # Tạo file tổng hợp kết quả
    print("\n" + "=" * 60)
    print("CREATING SUMMARY FILES")
    print("=" * 60)
    
    # 1. CSV file
    csv_path = create_summary_csv(args.save_dir, args.timesteps, args.categories)
    
    # 2. Markdown file (dễ đọc)
    md_path = create_summary_markdown(args.save_dir, args.timesteps, args.categories)
    
    # 3. Excel file (nếu có pandas)
    excel_path = create_summary_excel(args.save_dir, args.timesteps, args.categories)
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"\n📁 Results saved in: {args.save_dir}/")
    print(f"   - Individual results: {args.save_dir}/{{category}}_results.txt")
    print(f"   - Summary CSV: {csv_path}")
    print(f"   - Summary Markdown: {md_path}")
    if excel_path:
        print(f"   - Summary Excel: {excel_path}")
    print("\n💡 You can copy the Markdown table directly into Google Sheets or Excel.")


if __name__ == '__main__':
    main()