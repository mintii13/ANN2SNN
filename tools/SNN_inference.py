#!/usr/bin/env python3
"""
SNN UniAD Inference Script with ECMT
Usage: python tools/snn_inference.py --config config.yaml --checkpoint checkpoint.pth --class_name bottle
python tools/snn_inference.py  --config tools/config.yaml --checkpoint tools/checkpoints/Bottle/ckpt.pth.tar --class_name bottle --timesteps 4 --calibrate --benchmark
"""

import os
import sys
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

# Add project root to path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from utils.misc_helper import load_state, set_random_seed, update_config
from datasets.data_builder import build_dataloader
from models.model_helper import ModelHelper
from models.SNN_model import convert_uniad_to_ecmt, calibrate_ecmt_thresholds, ecmt_temporal_inference

def parse_args():
    parser = argparse.ArgumentParser(description='SNN UniAD Inference with ECMT')
    parser.add_argument('--config', type=str, default='./tools/config.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='ANN checkpoint path')
    parser.add_argument('--class_name', type=str, default=None, help='Single class to test (e.g., bottle)')
    parser.add_argument('--timesteps', type=int, default=4, help='Number of timesteps (ECMT uses 4)')
    parser.add_argument('--tau', type=float, default=2.0, help='LIF time constant')
    parser.add_argument('--num_thresholds', type=int, default=4, help='Multi-threshold neuron thresholds')
    parser.add_argument('--compensation_window', type=int, default=4, help='ECM history window')
    parser.add_argument('--output_dir', type=str, default='./snn_results', help='Output directory')
    parser.add_argument('--calibrate', action='store_true', help='Run threshold calibration')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark vs ANN baseline')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        config = update_config(config)
    
    # Set random seed
    set_random_seed(config.get('random_seed', 133), reproduce=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("🚀 SNN UniAD with ECMT Conversion")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Class: {args.class_name or 'All classes'}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # 1. Load ANN model
    print("📦 Loading ANN model...")
    ann_model = ModelHelper(config.net)
    ann_model.to(device)
    ann_model.device = torch.device(device)
    load_state(args.checkpoint, ann_model)
    ann_model.eval()
    print("✅ ANN model loaded successfully")
    
    # 2. Convert to ECMT SNN
    print("🧠 Converting to ECMT SNN...")
    ecmt_model = convert_uniad_to_ecmt(
        ann_model,
        tau=args.tau,
        num_thresholds=args.num_thresholds,
        compensation_window=args.compensation_window
    )
    ecmt_model.to(device)
    ecmt_model.eval()
    print("✅ ECMT conversion completed")
    
    # 3. Build dataloader
    print("📊 Building dataloader...")
    _, test_loader = build_dataloader(
        config.dataset, 
        distributed=False, 
        class_name=args.class_name
    )
    print(f"✅ Dataloader ready: {len(test_loader)} batches")
    
    # 4. Threshold calibration (optional)
    if args.calibrate:
        print("🎯 Calibrating thresholds...")
        threshold_dict = calibrate_ecmt_thresholds(ann_model, ecmt_model, test_loader)
        print(f"✅ Calibrated {len(threshold_dict)} thresholds")
    
    # 5. Benchmark (optional)
    if args.benchmark:
        print("📈 Benchmarking performance...")
        # Get sample batch for benchmarking
        sample_batch = next(iter(test_loader))
        test_input = {
            'image': sample_batch['image'][:1].to(device),
            'clsname': [sample_batch['clsname'][0]] if 'clsname' in sample_batch else ['bottle'],
            'filename': [sample_batch['filename'][0]] if 'filename' in sample_batch else ['test.png']
        }
        
        # ANN baseline
        with torch.no_grad():
            ann_output = ann_model(test_input)
            ann_pred = ann_output['pred']
        
        # ECMT inference
        ecmt_model.reset()
        with torch.no_grad():
            ecmt_output = ecmt_model(test_input, timesteps=args.timesteps)
            ecmt_pred = ecmt_output['pred']
        
        # Compute metrics
        mse_loss = torch.nn.functional.mse_loss(ecmt_pred, ann_pred)
        relative_error = (mse_loss / ann_pred.var()).item()
        
        print(f"📊 Benchmark Results:")
        print(f"   MSE Loss: {mse_loss.item():.6f}")
        print(f"   Relative Error: {relative_error:.4f} ({relative_error*100:.2f}%)")
        print(f"   Expected (Paper): ~1% accuracy loss")
    
    # 6. Run SNN inference
    print("🔥 Running SNN inference...")
    results = ecmt_temporal_inference(ecmt_model, test_loader, timesteps=args.timesteps, device=device)
    
    # 7. Save results
    print("💾 Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save predictions
    all_preds = []
    all_filenames = []
    all_clsnames = []
    
    for batch_result in results:
        all_preds.extend(batch_result['pred'])
        all_filenames.extend(batch_result['filename'])
        all_clsnames.extend(batch_result['clsname'])
    
    # Save as numpy arrays for evaluation
    np.savez(
        os.path.join(args.output_dir, f'snn_results_{args.class_name or "all"}.npz'),
        predictions=np.array([p.numpy() for p in all_preds]),
        filenames=np.array(all_filenames),
        classnames=np.array(all_clsnames),
        timesteps=args.timesteps,
        config=dict(
            tau=args.tau,
            num_thresholds=args.num_thresholds,
            compensation_window=args.compensation_window
        )
    )
    
    print("=" * 60)
    print("🎉 SNN Inference Completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📊 Processed {len(all_preds)} samples")
    print("=" * 60)

if __name__ == '__main__':
    main()