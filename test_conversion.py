import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from glob import glob
import cv2
from spikingjelly.activation_based import ann2snn, neuron, functional
from network import AutoEncoder
from options import Options
from utils import read_img


class ReLUAutoEncoder(nn.Module):
    """AutoEncoder with ReLU instead of LeakyReLU for testing conversion"""
    def __init__(self, cfg):
        super(ReLUAutoEncoder, self).__init__()
        
        # Build encoder layers
        encoder_layers = []
        
        # First two conv layers
        encoder_layers.extend([
            nn.Conv2d(cfg.input_channel, cfg.flc, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),  # Changed from LeakyReLU
            nn.Conv2d(cfg.flc, cfg.flc, 4, stride=2, padding=1), 
            nn.ReLU(inplace=True)   # Changed from LeakyReLU
        ])
        
        # Additional layer for patch_size=256
        if cfg.patch_size == 256:
            encoder_layers.extend([
                nn.Conv2d(cfg.flc, cfg.flc, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)   # Changed from LeakyReLU
            ])
        
        # Continue encoder
        encoder_layers.extend([
            nn.Conv2d(cfg.flc, cfg.flc, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.Conv2d(cfg.flc, cfg.flc*2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.Conv2d(cfg.flc*2, cfg.flc*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.Conv2d(cfg.flc*2, cfg.flc*4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.Conv2d(cfg.flc*4, cfg.flc*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.Conv2d(cfg.flc*2, cfg.flc, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.Conv2d(cfg.flc, cfg.z_dim, 8, stride=1, padding=0)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers  
        decoder_layers = []
        
        decoder_layers.extend([
            nn.ConvTranspose2d(cfg.z_dim, cfg.flc, 8, stride=1, padding=0),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.Conv2d(cfg.flc, cfg.flc*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.Conv2d(cfg.flc*2, cfg.flc*4, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.ConvTranspose2d(cfg.flc*4, cfg.flc*2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.Conv2d(cfg.flc*2, cfg.flc*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.ConvTranspose2d(cfg.flc*2, cfg.flc, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.Conv2d(cfg.flc, cfg.flc, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),   # Changed from LeakyReLU
            nn.ConvTranspose2d(cfg.flc, cfg.flc, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)    # Changed from LeakyReLU
        ])
        
        # Additional layer for patch_size=256
        if cfg.patch_size == 256:
            decoder_layers.extend([
                nn.ConvTranspose2d(cfg.flc, cfg.flc, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)    # Changed from LeakyReLU
            ])
        
        # Final output layer (NO SIGMOID for conversion)
        decoder_layers.extend([
            nn.ConvTranspose2d(cfg.flc, cfg.input_channel, 4, stride=2, padding=1)
            # Removed Sigmoid for conversion testing
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TestDataset(Dataset):
    """Simple dataset for testing"""
    def __init__(self, files, grayscale, patch_size):
        self.files = files[:10]  # Only use 10 samples for quick test
        self.grayscale = grayscale  
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        img = read_img(self.files[idx], self.grayscale)
        img = cv2.resize(img, (self.patch_size, self.patch_size))
        img = img / 255.0
        
        if self.grayscale:
            img = torch.FloatTensor(img).unsqueeze(0)
        else:
            img = torch.FloatTensor(img).permute(2, 0, 1)
        return img, torch.tensor(0)


def verify_conversion(model_ann, model_snn):
    """Verify if conversion was successful"""
    print("=== CONVERSION VERIFICATION ===")
    
    # Check if models are different objects
    print(f"Same object? {model_ann is model_snn}")
    
    # Check model types
    print(f"ANN type: {type(model_ann)}")
    print(f"SNN type: {type(model_snn)}")
    
    # Check if SNN has spiking neurons
    snn_modules = list(model_snn.named_modules())
    spiking_found = 0
    relu_found = 0
    
    for name, module in snn_modules:
        module_str = str(type(module))
        if 'spikingjelly' in module_str or 'neuron' in module_str.lower():
            print(f"Found spiking module: {name} -> {type(module)}")
            spiking_found += 1
        elif 'relu' in module_str.lower():
            print(f"Found ReLU (not converted): {name} -> {type(module)}")
            relu_found += 1
    
    print(f"Spiking modules found: {spiking_found}")
    print(f"Unconverted ReLU modules: {relu_found}")
    
    if spiking_found == 0:
        print("❌ WARNING: No spiking neurons found in converted model!")
        return False
    else:
        print(f"✅ Conversion successful: {spiking_found} spiking modules found")
        return True


def test_conversion_difference(model_ann, model_snn, sample_input, device):
    """Test if ANN and SNN outputs are different"""
    print("=== TESTING OUTPUT DIFFERENCE ===")
    
    sample_input = sample_input.to(device)
    
    with torch.no_grad():
        # ANN output
        model_ann.eval()
        ann_output = model_ann(sample_input)
        print(f"ANN output range: [{ann_output.min():.4f}, {ann_output.max():.4f}]")
        
        # SNN output (single timestep)
        model_snn.eval()
        functional.reset_net(model_snn)
        snn_output = model_snn(sample_input)
        print(f"SNN output range: [{snn_output.min():.4f}, {snn_output.max():.4f}]")
        
        # Check if outputs are identical
        diff = torch.abs(ann_output - snn_output).max()
        print(f"Max difference: {diff:.6f}")
        
        if diff < 1e-6:
            print("❌ WARNING: Outputs are identical - conversion may have failed!")
            return False
        else:
            print(f"✅ Outputs are different - conversion likely successful")
            return True


def copy_weights_from_leaky_to_relu(leaky_model, relu_model):
    """Copy weights from LeakyReLU model to ReLU model"""
    print("Copying weights from LeakyReLU model to ReLU model...")
    
    leaky_state = leaky_model.state_dict()
    relu_state = relu_model.state_dict()
    
    # Copy all conv/linear layer weights (skip activation layers)
    copied_layers = 0
    for key in relu_state.keys():
        if key in leaky_state:
            relu_state[key] = leaky_state[key].clone()
            copied_layers += 1
    
    relu_model.load_state_dict(relu_state)
    print(f"✅ Copied {copied_layers} layer weights")
    
    return relu_model


def test_ann2snn_conversion():
    """Test ANN to SNN conversion with ReLU model"""
    print("=" * 60)
    print("TESTING ANN2SNN CONVERSION WITH RELU")
    print("=" * 60)
    
    # Load config
    cfg = Options().parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Load original LeakyReLU model
    print("\n1. Loading original LeakyReLU model...")
    original_model = AutoEncoder(cfg).to(device)
    
    checkpoint_path = f"./results/{cfg.name}/chechpoints/{cfg.loss}/model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    original_model.load_state_dict(checkpoint['model_state_dict'])
    original_model.eval()
    print("✅ Original model loaded")
    
    # Step 2: Create ReLU model and copy weights
    print("\n2. Creating ReLU model and copying weights...")
    relu_model = ReLUAutoEncoder(cfg).to(device)
    relu_model = copy_weights_from_leaky_to_relu(original_model, relu_model)
    relu_model.eval()
    
    # Step 3: Test ANN outputs are similar
    print("\n3. Testing if ReLU model produces similar outputs...")
    sample_input = torch.randn(1, cfg.input_channel, cfg.patch_size, cfg.patch_size).to(device)
    
    with torch.no_grad():
        original_output = original_model(sample_input)
        relu_output = relu_model(sample_input)
        output_diff = torch.abs(original_output - relu_output).mean()
        print(f"Average output difference: {output_diff:.6f}")
        
        if output_diff > 0.1:
            print("⚠️  WARNING: ReLU model outputs significantly different from original")
        else:
            print("✅ ReLU model outputs similar to original")
    
    # Step 4: Create calibration dataloader
    print("\n4. Creating calibration dataloader...")
    good_files = glob(cfg.train_data_dir + '/*png')[:20]  # Only 20 files for quick test
    dataset = TestDataset(good_files, cfg.grayscale, cfg.patch_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"✅ Created dataloader with {len(dataset)} samples")
    
    # Step 5: Attempt conversion
    print("\n5. Attempting ANN2SNN conversion...")
    try:
        converter = ann2snn.Converter(
            dataloader=dataloader, 
            device=device, 
            mode='max',
            momentum=0.1
        )
        
        snn_model = converter(relu_model)
        print("✅ Conversion completed without errors")
        
        # Step 6: Verify conversion
        print("\n6. Verifying conversion...")
        conversion_success = verify_conversion(relu_model, snn_model)
        
        if conversion_success:
            # Step 7: Test output differences
            print("\n7. Testing output differences...")
            outputs_different = test_conversion_difference(relu_model, snn_model, sample_input, device)
            
            if outputs_different:
                print("\n" + "=" * 60)
                print("🎉 CONVERSION TEST SUCCESSFUL!")
                print("✅ ReLU model can be converted to SNN successfully")
                print("✅ Conversion produces different outputs (as expected)")
                print("✅ Spiking neurons detected in converted model")
                print("=" * 60)
                return True
            else:
                print("\n❌ CONVERSION TEST FAILED: Outputs identical")
                return False
        else:
            print("\n❌ CONVERSION TEST FAILED: No spiking neurons found")
            return False
            
    except Exception as e:
        print(f"\n❌ CONVERSION FAILED WITH ERROR: {e}")
        return False


if __name__ == "__main__":
    success = test_ann2snn_conversion()
    
    if success:
        print("\n🚀 RECOMMENDATION: You can proceed with training ReLU model for SNN conversion")
        print("   - Replace LeakyReLU with ReLU in network.py")
        print("   - Retrain the model")
        print("   - Use conversion_test.py with new model")
    else:
        print("\n🛑 RECOMMENDATION: ANN2SNN conversion still fails with ReLU")
        print("   - Consider manual SNN implementation")
        print("   - Or try different conversion approaches")
        print("   - Architecture may be too complex for automatic conversion")