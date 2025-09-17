import torch
import numpy as np
from spikingjelly.activation_based import functional
from options import Options
import torch.nn as nn
import os

def debug_snn_behavior(model_snn, cfg, device):
    """Debug why some datasets give high AUC at T=1"""
    
    print("=== DEBUGGING SNN BEHAVIOR ===")
    
    # Create test input
    if cfg.grayscale:
        test_input = torch.randn(1, 1, cfg.patch_size, cfg.patch_size).to(device)
    else:
        test_input = torch.randn(1, 3, cfg.patch_size, cfg.patch_size).to(device)
    
    print(f"Test input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    
    # Test temporal consistency
    functional.reset_net(model_snn)
    outputs_over_time = []
    
    with torch.no_grad():
        for t in range(10):
            output = model_snn(test_input)
            outputs_over_time.append(output.clone())
            
            print(f"T={t+1}: Output range=[{output.min():.4f}, {output.max():.4f}], "
                  f"Mean={output.mean():.4f}, Std={output.std():.4f}")
    
    # Check if outputs change over time
    output_changes = []
    for i in range(1, len(outputs_over_time)):
        change = torch.abs(outputs_over_time[i] - outputs_over_time[i-1]).mean().item()
        output_changes.append(change)
    
    avg_change = np.mean(output_changes) if output_changes else 0
    print(f"\nAverage output change between timesteps: {avg_change:.8f}")
    
    # Check if SNN is actually spiking
    if avg_change < 1e-6:
        print("❌ CRITICAL: SNN outputs don't change over time!")
        print("   This means conversion failed or model isn't spiking")
        return False
    
    # Check spike patterns
    final_output = outputs_over_time[-1]
    spike_count = torch.sum(final_output > 0.5).item()
    total_neurons = final_output.numel()
    spike_rate = spike_count / total_neurons
    
    print(f"\nSpike statistics:")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Spiking neurons: {spike_count}")
    print(f"  Spike rate: {spike_rate:.4f}")
    
    if spike_rate > 0.95:
        print("❌ CRITICAL: Nearly all neurons spiking (saturated)")
        return False
    elif spike_rate < 0.01:
        print("❌ CRITICAL: Almost no neurons spiking")
        return False
    
    print("✅ SNN appears to be working correctly")
    return True

def check_voltage_scalers(model_snn):
    """Check VoltageScaler values"""
    print("\n=== VOLTAGE SCALER ANALYSIS ===")
    
    scaler_values = []
    for name, module in model_snn.named_modules():
        if 'VoltageScaler' in str(type(module)):
            if hasattr(module, 'scale'):
                scale_val = module.scale.item()
                scaler_values.append(scale_val)
                print(f"{name}: scale = {scale_val:.6f}")
    
    if scaler_values:
        print(f"\nScaler statistics:")
        print(f"  Min scale: {min(scaler_values):.6f}")
        print(f"  Max scale: {max(scaler_values):.6f}")
        print(f"  Mean scale: {np.mean(scaler_values):.6f}")
        
        # Check for problematic scalers
        if max(scaler_values) > 1000:
            print("❌ CRITICAL: Very large scale values detected!")
            return False
        elif min(scaler_values) < 0.001:
            print("❌ CRITICAL: Very small scale values detected!")
            return False
    
    return True

def compare_ann_vs_snn_single_timestep(model_ann, model_snn, cfg, device):
    """Compare ANN vs SNN output at T=1"""
    print("\n=== ANN vs SNN COMPARISON ===")
    
    # Create test input
    if cfg.grayscale:
        test_input = torch.randn(1, 1, cfg.patch_size, cfg.patch_size).to(device)
    else:
        test_input = torch.randn(1, 3, cfg.patch_size, cfg.patch_size).to(device)
    
    # Test ANN (without sigmoid)
    model_ann_no_sigmoid = model_ann
    decoder_layers = list(model_ann.decoder.children())
    if isinstance(decoder_layers[-1], nn.Sigmoid):
        model_ann_no_sigmoid.decoder = nn.Sequential(*decoder_layers[:-1])
    
    with torch.no_grad():
        ann_output = model_ann_no_sigmoid(test_input)
        print(f"ANN output: range=[{ann_output.min():.4f}, {ann_output.max():.4f}], "
              f"mean={ann_output.mean():.4f}")
    
    # Test SNN at T=1
    functional.reset_net(model_snn)
    with torch.no_grad():
        snn_output = model_snn(test_input)
        print(f"SNN output (T=1): range=[{snn_output.min():.4f}, {snn_output.max():.4f}], "
              f"mean={snn_output.mean():.4f}")
    
    # Check similarity
    if ann_output.shape == snn_output.shape:
        diff = torch.abs(ann_output - snn_output).mean()
        print(f"Mean difference: {diff:.6f}")
        
        if diff < 0.01:
            print("❌ CRITICAL: ANN and SNN outputs nearly identical!")
            print("   This suggests SNN is not properly converted")
            return False
    
    return True

def analyze_dataset_specific_issue(cfg):
    """Analyze why certain datasets have issues"""
    print(f"\n=== DATASET-SPECIFIC ANALYSIS FOR {cfg.name.upper()} ===")
    
    # Load a few training samples
    from glob import glob
    good_files = glob(cfg.train_data_dir + '/*png')[:5]
    
    pixel_stats = []
    for img_path in good_files:
        from utils import read_img
        import cv2
        
        img = read_img(img_path, cfg.grayscale)
        img = cv2.resize(img, (cfg.patch_size, cfg.patch_size))
        img_norm = img / 255.0
        
        pixel_stats.extend(img_norm.flatten())
    
    pixel_stats = np.array(pixel_stats)
    
    print(f"Dataset statistics:")
    print(f"  Min pixel: {pixel_stats.min():.4f}")
    print(f"  Max pixel: {pixel_stats.max():.4f}")  
    print(f"  Mean pixel: {pixel_stats.mean():.4f}")
    print(f"  Std pixel: {pixel_stats.std():.4f}")
    print(f"  Unique values: {len(np.unique(pixel_stats))}")
    
    # Check for dataset-specific patterns
    if cfg.name in ['wood', 'leather']:
        print(f"\n{cfg.name.upper()} SPECIFIC CHECKS:")
        print("- These are texture datasets")
        print("- May have different pixel distributions")
        print("- Conversion calibration might be inappropriate")
        
        # Check if very uniform texture
        if pixel_stats.std() < 0.1:
            print("❌ CRITICAL: Very uniform texture detected!")
            print("   This can cause conversion calibration issues")
            return False
    
    return True

def main_debug():
    """Main debug function"""
    cfg = Options().parse()
    
    print("=" * 60)
    print(f"DEBUGGING SNN CONVERSION ISSUES FOR {cfg.name.upper()}")
    print("=" * 60)
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ANN
    from network import AutoEncoder
    model_ann = AutoEncoder(cfg).to(device)
    checkpoint_path = os.path.join(cfg.chechpoint_dir, 'model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_ann.load_state_dict(checkpoint['model_state_dict'])
    model_ann.eval()
    
    # Convert to SNN
    from conversion_test import convert_to_snn
    model_snn = convert_to_snn(model_ann, cfg, device)
    
    # Run all debug checks
    checks_passed = 0
    total_checks = 5
    
    print("\n1. Testing SNN temporal behavior...")
    if debug_snn_behavior(model_snn, cfg, device):
        checks_passed += 1
    
    print("\n2. Checking VoltageScaler values...")
    if check_voltage_scalers(model_snn):
        checks_passed += 1
    
    print("\n3. Comparing ANN vs SNN...")
    if compare_ann_vs_snn_single_timestep(model_ann, model_snn, cfg, device):
        checks_passed += 1
    
    print("\n4. Analyzing dataset characteristics...")
    if analyze_dataset_specific_issue(cfg):
        checks_passed += 1
    
    print("\n5. Final assessment...")
    if checks_passed >= 4:
        print("✅ SNN conversion appears healthy")
        checks_passed += 1
    else:
        print("❌ SNN conversion has issues")
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    print(f"Checks passed: {checks_passed}/{total_checks}")
    
    if checks_passed < 3:
        print("❌ CRITICAL: Multiple conversion issues detected")
        print("   Recommendation: Re-train with different settings")
    elif checks_passed < 4:
        print("⚠️  WARNING: Some issues detected")
        print("   Recommendation: Investigate specific issues")
    else:
        print("✅ HEALTHY: Conversion appears successful")
        print("   The high T=1 AUC might be due to other factors")
    
    return checks_passed >= 4

if __name__ == '__main__':
    main_debug()