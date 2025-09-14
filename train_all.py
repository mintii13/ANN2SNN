#!/usr/bin/env python3
"""
Simple multi-class training script for MVTec AD dataset
Usage: python train_all.py --epochs 100
"""

import argparse
import subprocess
import sys
import time
import os

# Define all MVTec AD classes
MVTEC_CLASSES = [
    # Objects
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
    # 'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
    # # Textures  
    # 'carpet', 'grid', 'leather', 'tile', 'wood'
]

# Class-specific configurations (from README)
CLASS_CONFIGS = {
    'bottle': {
        'train': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'do_aug': True, 'p_rotate': 0.0},
        'test': {'bg_mask': 'W'}
    },
    'cable': {
        'train': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'do_aug': True, 'p_rotate': 0.0, 'p_horizonal_flip': 0.0, 'p_vertical_flip': 0.0},
        'test': {}
    },
    'capsule': {
        'train': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'do_aug': True, 'p_rotate': 0.0, 'p_horizonal_flip': 0.0, 'p_vertical_flip': 0.0},
        'test': {'bg_mask': 'W'}
    },
    'carpet': {
        'train': {'im_resize': 512, 'patch_size': 128, 'z_dim': 100, 'do_aug': True, 'rotate_angle_vari': 10},
        'test': {}
    },
    'grid': {
        'train': {'im_resize': 256, 'patch_size': 128, 'z_dim': 100, 'grayscale': True, 'do_aug': True},
        'test': {'grayscale': True}
    },
    'hazelnut': {
        'train': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'do_aug': True, 'p_rotate_crop': 0.0},
        'test': {'bg_mask': 'B'}
    },
    'leather': {
        'train': {'im_resize': 256, 'patch_size': 128, 'z_dim': 100, 'do_aug': True},
        'test': {}
    },
    'metal_nut': {
        'train': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'do_aug': True, 'p_rotate_crop': 0.0, 'p_horizonal_flip': 0.0, 'p_vertical_flip': 0.0},
        'test': {'bg_mask': 'B'}
    },
    'pill': {
        'train': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'do_aug': True, 'p_rotate': 0.0, 'p_horizonal_flip': 0.0, 'p_vertical_flip': 0.0},
        'test': {'bg_mask': 'B'}
    },
    'screw': {
        'train': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'grayscale': True, 'do_aug': True, 'p_rotate': 0.0},
        'test': {'grayscale': True, 'bg_mask': 'W'}
    },
    'tile': {
        'train': {'im_resize': 256, 'patch_size': 128, 'z_dim': 100, 'do_aug': True},
        'test': {}
    },
    'toothbrush': {
        'train': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'do_aug': True, 'p_rotate': 0.0, 'p_vertical_flip': 0.0},
        'test': {}
    },
    'transistor': {
        'train': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'do_aug': True, 'p_rotate': 0.0, 'p_vertical_flip': 0.0},
        'test': {}
    },
    'wood': {
        'train': {'im_resize': 256, 'patch_size': 128, 'z_dim': 100, 'do_aug': True, 'rotate_angle_vari': 15},
        'test': {}
    },
    'zipper': {
        'train': {'im_resize': 266, 'patch_size': 256, 'z_dim': 500, 'grayscale': True, 'do_aug': True, 'p_rotate': 0.0},
        'test': {'grayscale': True}
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train autoencoder for multiple MVTec AD classes')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for each class')
    return parser.parse_args()

def train_class(class_name, config, epochs):
    """Train a single class"""
    print(f"\nStarting training for class: {class_name}")
    print(f"Config: {config}")
    print(f"Epochs: {epochs}")
    
    # Build command arguments for train.py
    cmd = [sys.executable, 'train.py', '--name', class_name, '--epochs', str(epochs), '--loss', 'ssim_loss']

    # Add class-specific train configurations
    train_config = config.get('train', {})
    for key, value in train_config.items():
        if key == 'grayscale' and value:
            cmd.append('--grayscale')
        elif key == 'do_aug' and value:
            cmd.append('--do_aug')
        elif key not in ['grayscale', 'do_aug'] and value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run training
    start_time = time.time()
    
    try:
        # Run train.py as subprocess
        env = os.environ.copy()
        env['DISABLE_TQDM'] = '1'

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        
        # Print output in real-time
        last_line = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                # Only print if it's not a progress bar update
                if not (line.startswith('[') and ']' in line and '%|' in line):
                    print(f"[{class_name}] {line}")
                elif line != last_line:  # Print only if progress bar changed significantly
                    # Extract percentage for cleaner output
                    if '%|' in line:
                        try:
                            percent_part = line.split('%|')[0].split()[-1]
                            epoch_part = line.split(']')[0] + ']'
                            loss_part = line.split('loss=')[-1].split(']')[0] if 'loss=' in line else ''
                            clean_line = f"{epoch_part} {percent_part}% - loss={loss_part}" if loss_part else f"{epoch_part} {percent_part}%"
                            print(f"\r[{class_name}] {clean_line}", end='', flush=True)
                        except:
                            pass
        last_line = line
        
        return_code = process.poll()
        end_time = time.time()
        duration = end_time - start_time
        
        if return_code == 0:
            print(f"SUCCESS: {class_name} completed in {duration/60:.1f} minutes")
            return True, duration
        else:
            print(f"FAILED: {class_name} failed with return code {return_code}")
            return False, duration
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"ERROR: {class_name} failed with exception: {str(e)}")
        return False, duration

def main():
    args = parse_args()
    
    print("=" * 60)
    print("MVTec AD Multi-Class Training")
    print("=" * 60)
    print(f"Classes to train: {MVTEC_CLASSES}")
    print(f"Epochs per class: {args.epochs}")
    print(f"Total classes: {len(MVTEC_CLASSES)}")
    
    results = []
    total_start_time = time.time()
    
    try:
        for i, class_name in enumerate(MVTEC_CLASSES):
            print(f"\n{'='*60}")
            print(f"Training {i+1}/{len(MVTEC_CLASSES)}: {class_name}")
            print(f"{'='*60}")
            
            config = CLASS_CONFIGS.get(class_name, {})
            success, duration = train_class(class_name, config, args.epochs)
            
            results.append({
                'class': class_name,
                'success': success,
                'duration': duration
            })
            
            # Short break between classes (skip if failed)
            if i < len(MVTEC_CLASSES) - 1 and success:
                print(f"Waiting 30 seconds before next class...")
                time.sleep(30)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Final summary
    total_time = time.time() - total_start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    
    if successful:
        print(f"\nSuccessful classes:")
        for result in successful:
            print(f"  - {result['class']}: {result['duration']/60:.1f} min")
    
    if failed:
        print(f"\nFailed classes:")
        for result in failed:
            print(f"  - {result['class']}: {result['duration']/60:.1f} min")
    
    if results:
        print(f"\nAverage time per class: {sum(r['duration'] for r in results)/len(results)/60:.1f} minutes")
    
    print("Training completed!")

if __name__ == '__main__':
    main()