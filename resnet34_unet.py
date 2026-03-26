"""
ResNet34-UNet Architecture Implementation from Scratch

Combines ResNet-34 as encoder backbone with UNet decoder for semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    ResNet basic residual block: Conv -> BN -> ReLU -> Conv -> BN
    with optional skip connection
    """
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet34Encoder(nn.Module):
    """
    ResNet-34 encoder for feature extraction
    """
    
    def __init__(self, in_channels: int = 3):
        super(ResNet34Encoder, self).__init__()
        self.in_channels = in_channels
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 512, 3, stride=2)
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass returning intermediate features for skip connections
        
        Returns:
            Tuple of (features at different scales, final feature map)
        """
        # Initial convolution: 1/2 resolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 1/4 resolution
        
        # Layer 1: 1/4 resolution
        c1 = self.layer1(x)
        
        # Layer 2: 1/8 resolution
        c2 = self.layer2(c1)
        
        # Layer 3: 1/16 resolution
        c3 = self.layer3(c2)
        
        # Layer 4: 1/32 resolution
        c4 = self.layer4(c3)
        
        return c1, c2, c3, c4


class UNetDecoder(nn.Module):
    """
    UNet-style decoder with skip connections
    """
    
    def __init__(self, out_channels: int = 1):
        super(UNetDecoder, self).__init__()
        
        # Decoder blocks
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Additional final upsample to restore full resolution (1/2 -> 1/1)
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Final output layer
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, c4, c3, c2, c1):
        """
        Forward pass with skip connections
        
        Args:
            c4: Feature map from encoder layer 4 (1/32)
            c3: Feature map from encoder layer 3 (1/16)
            c2: Feature map from encoder layer 2 (1/8)
            c1: Feature map from encoder layer 1 (1/4)
        """
        # Decode layer 4 -> 3
        x = self.up4(c4)
        x = torch.cat([x, c3], dim=1)
        x = self.dec4(x)
        
        # Decode layer 3 -> 2
        x = self.up3(x)
        x = torch.cat([x, c2], dim=1)
        x = self.dec3(x)
        
        # Decode layer 2 -> 1
        x = self.up2(x)
        x = torch.cat([x, c1], dim=1)
        x = self.dec2(x)
        
        # Decode layer 1 -> 1/2
        x = self.up1(x)
        x = self.dec1(x)
        
        # Upsample to full resolution
        x = self.up0(x)
        
        # Final output
        x = self.final(x)
        return x


class ResNet34UNet(nn.Module):
    """
    ResNet34-UNet: Combines ResNet34 encoder with UNet decoder
    
    Architecture:
    - Encoder: ResNet-34 with 4 residual blocks
    - Decoder: 4 up-sampling blocks with skip connections from encoder
    - Output: Binary segmentation mask
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        """
        Initialize ResNet34-UNet
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary segmentation)
        """
        super(ResNet34UNet, self).__init__()
        self.encoder = ResNet34Encoder(in_channels)
        self.decoder = UNetDecoder(out_channels)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Output tensor (B, 1, H, W) or (B, out_channels, H, W)
        """
        # Encode
        c1, c2, c3, c4 = self.encoder(x)
        
        # Decode with skip connections
        x = self.decoder(c4, c3, c2, c1)
        
        return x


if __name__ == '__main__':
    # Test the model
    model = ResNet34UNet(in_channels=3, out_channels=1)
    print("ResNet34-UNet Model:")
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
