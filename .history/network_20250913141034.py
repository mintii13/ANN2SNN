import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()
        
        # Build encoder layers
        encoder_layers = []
        
        # First two conv layers
        encoder_layers.extend([
            nn.Conv2d(cfg.input_channel, cfg.flc, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc, cfg.flc, 4, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Additional layer for patch_size=256
        if cfg.patch_size == 256:
            encoder_layers.extend([
                nn.Conv2d(cfg.flc, cfg.flc, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Continue encoder
        encoder_layers.extend([
            nn.Conv2d(cfg.flc, cfg.flc, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc, cfg.flc*2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc*2, cfg.flc*2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc*2, cfg.flc*4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc*4, cfg.flc*2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc*2, cfg.flc, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc, cfg.z_dim, 8, stride=1, padding=0)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers  
        decoder_layers = []
        
        decoder_layers.extend([
            nn.ConvTranspose2d(cfg.z_dim, cfg.flc, 8, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc, cfg.flc*2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc*2, cfg.flc*4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(cfg.flc*4, cfg.flc*2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc*2, cfg.flc*2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(cfg.flc*2, cfg.flc, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cfg.flc, cfg.flc, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(cfg.flc, cfg.flc, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Additional layer for patch_size=256
        if cfg.patch_size == 256:
            decoder_layers.extend([
                nn.ConvTranspose2d(cfg.flc, cfg.flc, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Final output layer
        decoder_layers.extend([
            nn.ConvTranspose2d(cfg.flc, cfg.input_channel, 4, stride=2, padding=1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded