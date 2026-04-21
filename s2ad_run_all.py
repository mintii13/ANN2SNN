"""
s2ad_run_all.py - Run S2AD on multiple MVTec categories
=========================================================
Statistical SNN-based Anomaly Detection - Batch evaluation.

Usage:
  python s2ad_run_all.py --wandb
  python s2ad_run_all.py --categories bottle cable leather --timesteps 8 16 32
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
import numpy as np
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# MVTec categories (15 categories)
ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

# Default timesteps to test
DEFAULT_TIMESTEPS = [8, 16, 32, 64]

# Default layers
DEFAULT_LAYERS = 'layer23'

# Paths
DEFAULT_DATA_PATH = '/home/minhtringuyen/ANN2SNN/mvtec'
DEFAULT_SAVE_DIR = './s2ad_results'
S2AD_SCRIPT = 's2ad_validate.py'  # Tên script S2AD


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_s2ad(category, args):
    """Run S2AD for a single category."""
    
    # Tạo tên run dựa trên config khác default
    run_name = f"{category}_{args.layers}"
    
    # Membrane
    run_name += "_memTrue" if args.use_membrane else "_memFalse"
    
    # Batch size (nếu khác 16)
    if args.batch_size != 16:
        run_name += f"_b{args.batch_size}"
    
    # Calibration samples (nếu khác 100)
    if args.calib_samples != 100:
        run_name += f"_c{args.calib_samples}"
    
    # Timesteps
    run_name += "_T" + "".join(str(t) for t in args.timesteps)
    
    cmd = [
        sys.executable, S2AD_SCRIPT,
        '--name', category,
        '--data_path', args.data_path,
        '--layers', args.layers,
        '--timesteps'] + [str(t) for t in args.timesteps] + [
        '--img_size', str(args.img_size),
        '--batch_size', str(args.batch_size),
        '--calib_samples', str(args.calib_samples),
        '--snn_mode', args.snn_mode,
        '--save_dir', args.save_dir,
    ]
    
    if args.use_membrane:
        cmd.append('--use_membrane')
    
    if args.wandb:
        cmd.append('--wandb')
        cmd.extend(['--wandb_project', args.wandb_project])
        cmd.extend(['--wandb_run_name', run_name])  # Thêm argument mới
        if args.wandb_offline:
            cmd.append('--wandb_offline')
    
    print(f"\n{'='*60}")
    print(f"Running S2AD on category: {category}")
    print(f"Run name: {run_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ Completed {category} in {elapsed_time:.2f} seconds")
        return True
    else:
        print(f"\n✗ Failed {category} after {elapsed_time:.2f} seconds")
        return False


def collect_results(args):
    """Collect all results from saved files."""
    results = {}
    
    for category in args.categories:
        result_file = os.path.join(args.save_dir, f'{category}_results.txt')
        if os.path.exists(result_file):
            results[category] = parse_result_file(result_file)
        else:
            print(f"Warning: Result file not found for {category}")
            results[category] = None
    
    return results


def parse_result_file(filepath):
    """Parse result file to extract AUCs for each timestep."""
    results = {}
    current_timestep = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the results table
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


def print_summary_table(results, categories):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("S2AD SUMMARY RESULTS")
    print("=" * 80)
    
    # Get all timesteps from all categories
    all_timesteps = set()
    for cat_results in results.values():
        if cat_results:
            all_timesteps.update(cat_results.keys())
    all_timesteps = sorted(all_timesteps)
    
    if not all_timesteps:
        print("No results found!")
        return
    
    # Print Image AUC table
    print("\n📊 IMAGE AUC:")
    print("-" * (8 + 12 * len(all_timesteps)))
    print(f"{'Category':<12}", end="")
    for T in all_timesteps:
        print(f"T={T:<8}", end="")
    print()
    print("-" * (8 + 12 * len(all_timesteps)))
    
    for category in categories:
        print(f"{category:<12}", end="")
        cat_results = results.get(category)
        if cat_results:
            for T in all_timesteps:
                auc = cat_results.get(T, {}).get('img_auc', None)
                if auc is not None:
                    print(f"{auc:<10.4f}", end="")
                else:
                    print(f"{'N/A':<10}", end="")
        else:
            for _ in all_timesteps:
                print(f"{'N/A':<10}", end="")
        print()
    
    # Print Pixel AUC table
    print("\n📊 PIXEL AUC:")
    print("-" * (8 + 12 * len(all_timesteps)))
    print(f"{'Category':<12}", end="")
    for T in all_timesteps:
        print(f"T={T:<8}", end="")
    print()
    print("-" * (8 + 12 * len(all_timesteps)))
    
    for category in categories:
        print(f"{category:<12}", end="")
        cat_results = results.get(category)
        if cat_results:
            for T in all_timesteps:
                auc = cat_results.get(T, {}).get('pix_auc', None)
                if auc is not None:
                    print(f"{auc:<10.4f}", end="")
                else:
                    print(f"{'N/A':<10}", end="")
        else:
            for _ in all_timesteps:
                print(f"{'N/A':<10}", end="")
        print()
    
    print("-" * (8 + 12 * len(all_timesteps)))


def log_to_wandb(results, categories, args):
    """Log summary results to WandB."""
    if not args.wandb or not WANDB_AVAILABLE:
        return
    
    # Tạo tên run summary dựa trên config
    summary_name = f"S2AD_summary_{args.layers}_mem{args.use_membrane}"
    
    # Batch size (nếu khác 16)
    if args.batch_size != 16:
        summary_name += f"_b{args.batch_size}"
    
    # Calibration samples (nếu khác 100)
    if args.calib_samples != 100:
        summary_name += f"_c{args.calib_samples}"
    
    summary_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize WandB for summary
    wandb.init(
        project=args.wandb_project,
        name=summary_name,
        config={
            'categories': categories,
            'timesteps': args.timesteps,
            'layers': args.layers,
            'use_membrane': args.use_membrane,
            'snn_mode': args.snn_mode,
            'batch_size': args.batch_size,
            'calib_samples': args.calib_samples,
        },
        mode='offline' if args.wandb_offline else 'online'
    )
    
    # Collect all timesteps
    all_timesteps = set()
    for cat_results in results.values():
        if cat_results:
            all_timesteps.update(cat_results.keys())
    all_timesteps = sorted(all_timesteps)
    
    # Create summary tables
    img_table_data = []
    pix_table_data = []
    
    for category in categories:
        cat_results = results.get(category)
        if cat_results:
            for T in all_timesteps:
                auc = cat_results.get(T, {})
                img_table_data.append([category, T, auc.get('img_auc', None)])
                pix_table_data.append([category, T, auc.get('pix_auc', None)])
    
    # Log tables
    img_table = wandb.Table(data=img_table_data, 
                            columns=["Category", "Timestep", "Image AUC"])
    pix_table = wandb.Table(data=pix_table_data,
                            columns=["Category", "Timestep", "Pixel AUC"])
    
    wandb.log({"summary_image_auc": img_table})
    wandb.log({"summary_pixel_auc": pix_table})
    
    # Calculate average AUC per timestep
    avg_img_auc = {}
    avg_pix_auc = {}
    for T in all_timesteps:
        img_vals = []
        pix_vals = []
        for cat_results in results.values():
            if cat_results and T in cat_results:
                img_vals.append(cat_results[T]['img_auc'])
                pix_vals.append(cat_results[T]['pix_auc'])
        if img_vals:
            avg_img_auc[T] = np.mean(img_vals)
            avg_pix_auc[T] = np.mean(pix_vals)
    
    # Log average line plots
    if avg_img_auc:
        img_avg_table = wandb.Table(
            data=[[T, avg_img_auc[T]] for T in sorted(avg_img_auc.keys())],
            columns=["Timestep", "Average Image AUC"]
        )
        wandb.log({
            "Average Image AUC vs Timestep": wandb.plot.line(
                img_avg_table, "Timestep", "Average Image AUC",
                title="Average Image AUC across all categories",
                stroke="blue"
            )
        })
    
    if avg_pix_auc:
        pix_avg_table = wandb.Table(
            data=[[T, avg_pix_auc[T]] for T in sorted(avg_pix_auc.keys())],
            columns=["Timestep", "Average Pixel AUC"]
        )
        wandb.log({
            "Average Pixel AUC vs Timestep": wandb.plot.line(
                pix_avg_table, "Timestep", "Average Pixel AUC",
                title="Average Pixel AUC across all categories",
                stroke="red"
            )
        })
    
    # Log summary metrics
    wandb.run.summary['total_categories'] = len(categories)
    wandb.run.summary['completed_categories'] = sum(1 for r in results.values() if r is not None)
    
    print(f"\n  WandB summary run completed")
    wandb.finish()


def save_combined_results(results, categories, args):
    """Save combined results to a CSV file."""
    # Get all timesteps
    all_timesteps = set()
    for cat_results in results.values():
        if cat_results:
            all_timesteps.update(cat_results.keys())
    all_timesteps = sorted(all_timesteps)
    
    # Tạo tên file dựa trên config
    file_suffix = f"{args.layers}_mem{args.use_membrane}"
    
    # Batch size (nếu khác 16)
    if args.batch_size != 16:
        file_suffix += f"_b{args.batch_size}"
    
    # Calibration samples (nếu khác 100)
    if args.calib_samples != 100:
        file_suffix += f"_c{args.calib_samples}"
    
    # Save Image AUC
    img_csv_path = os.path.join(args.save_dir, f's2ad_summary_image_auc_{file_suffix}.csv')
    with open(img_csv_path, 'w') as f:
        f.write("Category," + ",".join([f"T{T}" for T in all_timesteps]) + "\n")
        for category in categories:
            f.write(f"{category},")
            cat_results = results.get(category)
            if cat_results:
                for T in all_timesteps:
                    auc = cat_results.get(T, {}).get('img_auc', '')
                    f.write(f"{auc}," if auc else ",")
            f.write("\n")
    print(f"\nImage AUC results saved: {img_csv_path}")
    
    # Save Pixel AUC
    pix_csv_path = os.path.join(args.save_dir, f's2ad_summary_pixel_auc_{file_suffix}.csv')
    with open(pix_csv_path, 'w') as f:
        f.write("Category," + ",".join([f"T{T}" for T in all_timesteps]) + "\n")
        for category in categories:
            f.write(f"{category},")
            cat_results = results.get(category)
            if cat_results:
                for T in all_timesteps:
                    auc = cat_results.get(T, {}).get('pix_auc', '')
                    f.write(f"{auc}," if auc else ",")
            f.write("\n")
    print(f"Pixel AUC results saved: {pix_csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='S2AD - Run on multiple MVTec categories')
    
    # Categories
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Categories to run (default: all 15 categories)')
    
    # SNN Configuration
    parser.add_argument('--timesteps', type=int, nargs='+', default=DEFAULT_TIMESTEPS,
                        help='Timesteps to test')
    parser.add_argument('--layers', type=str, default=DEFAULT_LAYERS,
                        choices=['layer1', 'layer2', 'layer3', 'layer12', 'layer23', 'layer123'])
    parser.add_argument('--use_membrane', action='store_true',
                        help='Use membrane potential in scoring')
    parser.add_argument('--calib_samples', type=int, default=100)
    parser.add_argument('--snn_mode', type=str, default='max',
                        choices=['max', '0.99', '0.9'])
    
    # Dataset
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    
    # Paths
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR)
    
    # WandB
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='S2AD',
                        help='WandB project name')
    parser.add_argument('--wandb_offline', action='store_true',
                        help='Run WandB in offline mode')
    
    # Execution
    parser.add_argument('--skip_failed', action='store_true',
                        help='Continue even if a category fails')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set categories
    if args.categories is None:
        args.categories = ALL_CATEGORIES
    else:
        # Validate categories
        for cat in args.categories:
            if cat not in ALL_CATEGORIES:
                print(f"Warning: '{cat}' is not a standard MVTec category")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("S2AD - Statistical SNN-based Anomaly Detection")
    print("Batch evaluation on MVTec dataset")
    print("=" * 60)
    print(f"Categories: {len(args.categories)} categories")
    print(f"  {args.categories}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Layers: {args.layers}")
    print(f"Use membrane: {args.use_membrane}")
    print(f"SNN mode: {args.snn_mode}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 60)
    
    # Check if script exists
    if not os.path.exists(S2AD_SCRIPT):
        print(f"\nError: {S2AD_SCRIPT} not found in current directory!")
        sys.exit(1)
    
    # Collect existing results if resuming
    results = {}
    if args.resume:
        print("\nResuming mode: Loading existing results...")
        results = collect_results(args)
        completed = [cat for cat, res in results.items() if res is not None]
        pending = [cat for cat in args.categories if cat not in completed]
        print(f"  Completed: {len(completed)} categories")
        print(f"  Pending: {len(pending)} categories")
        categories_to_run = pending
    else:
        categories_to_run = args.categories
    
    # Run S2AD for each category
    successful = []
    failed = []
    
    for i, category in enumerate(categories_to_run):
        print(f"\n[{i+1}/{len(categories_to_run)}] Processing {category}...")
        
        success = run_s2ad(category, args)
        
        if success:
            successful.append(category)
        else:
            failed.append(category)
            if not args.skip_failed:
                print(f"\nStopping due to failure in {category}")
                break
    
    # Collect all results
    all_results = collect_results(args)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(successful)} categories")
    if successful:
        print(f"  {successful}")
    print(f"Failed: {len(failed)} categories")
    if failed:
        print(f"  {failed}")
    
    # Print results table
    print_summary_table(all_results, args.categories)
    
    # Save combined results
    save_combined_results(all_results, args.categories, args)
    
    # Log to WandB
    if args.wandb and successful:
        log_to_wandb(all_results, args.categories, args)
    
    print("\n" + "=" * 60)
    print("S2AD batch evaluation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()