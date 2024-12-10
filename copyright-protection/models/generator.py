import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, use_instance_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.instance_norm = nn.InstanceNorm2d(out_channels) if use_instance_norm else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm:
            x = self.instance_norm(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.instance_norm1 = nn.InstanceNorm2d(channels)
        self.instance_norm2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.instance_norm1(self.conv1(x)))
        x = self.instance_norm2(self.conv2(x))
        return x + residual

class Generator(nn.Module):
    def __init__(self, input_channels=3, dropout_rate=0.2):
        super().__init__()
        self.d64 = nn.Sequential(
            ConvBlock(input_channels*2, 64),
            nn.Dropout2d(p=dropout_rate)  # Sử dụng Dropout2d cho conv layers
        )
        self.d128 = nn.Sequential(
            ConvBlock(64, 128),
            nn.Dropout2d(p=dropout_rate)
        )
        self.d256 = nn.Sequential(
            ConvBlock(128, 256),
            nn.Dropout2d(p=dropout_rate)
        )
        
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(4)])
        
        self.u128 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )
        self.u64 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )
        
        self.u32 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

        self.final = nn.Sequential(
            nn.Conv2d(32, input_channels, 3, 1, 1),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, m):
        # Debug prints
        combined = torch.cat([x, m], dim=1)
        # combined = x + m
        # print(f"Input size: {combined.shape}")
        
        e1 = self.d64(combined)
        # print(f"After d64: {e1.shape}")
        
        e2 = self.d128(e1)
        # print(f"After d128: {e2.shape}")
        
        e3 = self.d256(e2)
        # print(f"After d256: {e3.shape}")
        
        r = self.residual_blocks(e3)
        # print(f"After residual: {r.shape}")
        
        d1 = self.u128(r)
        # print(f"After u128: {d1.shape}")
        
        d2 = self.u64(d1)
        # print(f"After u64: {d2.shape}")

        d3 = self.u32(d2)
        # print(f"After u32: {d3.shape}")
        
        out = self.final(d3)
        # print(f"Output size: {out.shape}")
        
        # Ensure output matches input spatial dimensions
        assert out.shape[2:] == x.shape[2:], f"Output shape {out.shape} doesn't match input shape {x.shape}"
        
        return out + x