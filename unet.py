"""
UNet Architecture Implementation from Scratch

Paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
Ronneberger et al., 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block: Conv -> ReLU -> Conv -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: ConvTranspose2d or Upsample -> DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # When we upsample in_channels and concat with skip connection of out_channels*2,
            # we get in_channels + out_channels*2 total channels
            # For standard UNet: in_channels=1024, skip=512 → 1536 → conv to 512
            # So conv should expect in_channels + (in_channels // 2)
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # ConvTranspose2d outputs in_channels // 2, concat with skip (in_channels // 2)
            # So total input to conv is in_channels
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Calculate padding if needed
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate skip connection (upsampled first, then skip)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet segmentation network.
    
    Architecture:
    - Encoder: 4 down-sampling blocks with skip connections
    - Decoder: 4 up-sampling blocks with concatenation
    - Output: Single channel segmentation mask
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, bilinear: bool = True):
        """
        Initialize UNet
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary segmentation)
            bilinear: Use bilinear interpolation for upsampling
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder (contracting path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder (expanding path)
        # For up layers: first arg is input channels, second is output channels
        # After concat with skip connection, actual input to conv will be double the first arg
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Output tensor (B, 1, H, W) or (B, out_channels, H, W)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        x = self.outc(x)
        return x


if __name__ == '__main__':
    # Test the model
    model = UNet(in_channels=3, out_channels=1)
    print("UNet Model:")
    print(model)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
