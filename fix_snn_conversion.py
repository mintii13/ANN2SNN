import torch
import torch.nn as nn
from spikingjelly.activation_based import ann2snn, neuron, functional
from network import AutoEncoder
from options import Options

def fix_snn_conversion():
    """Fix SNN conversion để có dynamic behavior"""
    cfg = Options().parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_ann = AutoEncoder(cfg).to(device)
    checkpoint_path = f"./results/{cfg.name}/chechpoints/{cfg.loss}/model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_ann.load_state_dict(checkpoint['model_state_dict'])
    model_ann.eval()
    
    # Remove Sigmoid
    original_decoder = list(model_ann.decoder.children())
    if isinstance(original_decoder[-1], nn.Sigmoid):
        model_ann.decoder = nn.Sequential(*original_decoder[:-1])
    
    print("=== TESTING DIFFERENT CONVERSION MODES ===")
    
    # Test different modes
    modes_to_test = ['max', '99.9%', '99%', '95%']
    
    for mode in modes_to_test:
        print(f"\n--- Testing mode: {mode} ---")
        
        try:
            # Create calibration data
            sample_data = torch.randn(8, cfg.input_channel, cfg.patch_size, cfg.patch_size).to(device)
            from torch.utils.data import TensorDataset, DataLoader
            dataset = TensorDataset(sample_data, torch.zeros(8))
            dataloader = DataLoader(dataset, batch_size=4)
            
            # Convert
            converter = ann2snn.Converter(
                dataloader=dataloader,
                device=device,
                mode=mode,
                momentum=0.1
            )
            
            model_snn = converter(model_ann)
            
            # Test behavior
            test_input = torch.randn(1, cfg.input_channel, cfg.patch_size, cfg.patch_size).to(device)
            
            # Reset and run multiple timesteps
            functional.reset_net(model_snn)
            outputs = []
            
            with torch.no_grad():
                for t in range(10):
                    output = model_snn(test_input)
                    outputs.append(output.clone())
            
            # Calculate variance
            output_stack = torch.stack(outputs)
            variance = output_stack.var(dim=0).mean().item()
            
            print(f"Output variance: {variance:.8f}")
            
            if variance > 1e-6:
                print(f"✅ SUCCESS with mode {mode}! Dynamic behavior detected.")
                return model_snn, mode
            else:
                print(f"❌ Mode {mode} still produces static output")
                
        except Exception as e:
            print(f"❌ Mode {mode} failed: {e}")
    
    print("\n=== TRYING MANUAL NEURON REPLACEMENT ===")
    
    # Try manual replacement with real LIF neurons
    try:
        model_snn = manual_snn_conversion(model_ann, device)
        
        # Test manual conversion
        test_input = torch.randn(1, cfg.input_channel, cfg.patch_size, cfg.patch_size).to(device)
        functional.reset_net(model_snn)
        outputs = []
        
        with torch.no_grad():
            for t in range(10):
                output = model_snn(test_input)
                outputs.append(output.clone())
        
        variance = torch.stack(outputs).var(dim=0).mean().item()
        print(f"Manual conversion variance: {variance:.8f}")
        
        if variance > 1e-6:
            print("✅ Manual conversion successful!")
            return model_snn, "manual"
        
    except Exception as e:
        print(f"❌ Manual conversion failed: {e}")
    
    return None, None

def manual_snn_conversion(model_ann, device):
    """Manually replace ReLU with LIF neurons"""
    print("Attempting manual LIF neuron replacement...")
    
    def replace_relu_with_lif(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                # Replace ReLU with LIF
                lif_neuron = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='s')
                setattr(module, name, lif_neuron)
                print(f"Replaced {name} with LIFNode")
            else:
                replace_relu_with_lif(child)
    
    # Create a copy and replace
    model_snn = type(model_ann)(Options().parse()).to(device)
    model_snn.load_state_dict(model_ann.state_dict())
    
    # Replace all ReLU with LIF
    replace_relu_with_lif(model_snn)
    
    return model_snn

def test_improved_rate_coding():
    """Test với improved rate coding strategies"""
    print("\n=== TESTING IMPROVED RATE CODING ===")
    
    model_snn, mode = fix_snn_conversion()
    
    if model_snn is None:
        print("❌ No working SNN conversion found")
        return False
    
    print(f"Using SNN converted with mode: {mode}")
    
    cfg = Options().parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_input = torch.randn(1, cfg.input_channel, cfg.patch_size, cfg.patch_size).to(device)
    
    # Test different rate coding strategies
    strategies = {
        'simple_average': lambda outputs: torch.stack(outputs).mean(dim=0),
        'weighted_average': lambda outputs: weighted_average(outputs, device),
        'stable_period': lambda outputs: torch.stack(outputs[len(outputs)//3:]).mean(dim=0),
        'exponential_moving': lambda outputs: exponential_moving_average(outputs),
        'median': lambda outputs: torch.stack(outputs).median(dim=0)[0]
    }
    
    for strategy_name, strategy_func in strategies.items():
        print(f"\n--- Testing {strategy_name} ---")
        
        functional.reset_net(model_snn)
        outputs = []
        
        with torch.no_grad():
            for t in range(20):
                output = model_snn(test_input)
                outputs.append(output.clone())
        
        # Apply strategy
        final_output = strategy_func(outputs)
        
        # Check convergence
        variance = torch.stack(outputs).var(dim=0).mean().item()
        output_range = f"[{final_output.min():.4f}, {final_output.max():.4f}]"
        
        print(f"  Variance: {variance:.8f}")
        print(f"  Output range: {output_range}")
        
        if variance > 1e-6:
            print(f"  ✅ {strategy_name} shows dynamic behavior")
        else:
            print(f"  ❌ {strategy_name} still static")
    
    return True

def weighted_average(outputs, device):
    """Weighted average with higher weights for later timesteps"""
    weights = torch.linspace(0.1, 1.0, len(outputs)).to(device)
    weighted_outputs = torch.stack([w * out for w, out in zip(weights, outputs)])
    return weighted_outputs.sum(dim=0) / weights.sum()

def exponential_moving_average(outputs, alpha=0.9):
    """Exponential moving average"""
    result = outputs[0]
    for i in range(1, len(outputs)):
        result = alpha * result + (1 - alpha) * outputs[i]
    return result

if __name__ == "__main__":
    print("🔧 FIXING SNN CONVERSION AND RATE CODING")
    print("="*60)
    
    success = test_improved_rate_coding()
    
    if not success:
        print("\n🎯 FINAL RECOMMENDATIONS:")
        print("1. Model architecture may be too complex for ann2snn")
        print("2. Try simpler AutoEncoder architecture")
        print("3. Implement SNN from scratch")
        print("4. Use different SNN conversion library")
    else:
        print("\n✅ Found working SNN conversion!")
        print("Update conversion_test.py with working configuration")