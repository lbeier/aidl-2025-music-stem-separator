import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution Block:
    - Applies two consecutive convolution layers
    - Each followed by a ReLU activation
    - Optionally applies Dropout after the second ReLU
    - Used as the basic building block inside U-Net
    """

    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 1st Conv layer
            nn.ReLU(inplace=True),  # 1st ReLU activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 2nd Conv layer
            nn.ReLU(inplace=True)  # 2nd ReLU activation
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(dropout_p))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the double convolution block.

        Args:
            x (torch.Tensor): Input feature map

        Returns:
            torch.Tensor: Output feature map after two convolutions
        """
        return self.conv(x)


class UNetSmall(nn.Module):
    """
    Lightweight U-Net Model:
    - Encoder-decoder architecture with skip connections
    - Fewer feature maps to keep model small and fast
    """

    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128], dropout_p=0.3):
        """
        Initialize the U-Net.

        Args:
            in_channels (int): Number of input channels (e.g., 1 for mel-spectrograms)
            out_channels (int): Number of output channels (e.g., 1 if predicting a mask or spectrogram)
            features (list): Number of feature channels at each downsampling step
            dropout_p (float): Dropout probability to use in DoubleConv blocks.
        """
        super().__init__()

        self.downs = nn.ModuleList()  # Downsampling path (encoder)
        self.ups = nn.ModuleList()  # Upsampling path (decoder)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling after each down block

        # --- Build Encoder ---
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_p=dropout_p))
            in_channels = feature  # Update in_channels for next layer

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_p=dropout_p)

        # --- Build Decoder ---
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )  # Upsampling (transposed convolution)
            self.ups.append(
                DoubleConv(feature * 2, feature, dropout_p=dropout_p)
            )  # Double conv after skip connection

        # --- Final output layer ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)  # 1x1 convolution to output

    def forward(self, x):
        """
        Forward pass through U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        skip = []  # Store outputs for skip connections

        # --- Encoder: Downsampling path ---
        for down in self.downs:
            x = down(x)  # Double convolution
            skip.append(x)  # Save for skip connection
            x = self.pool(x)  # Downsample

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # Reverse skip connections for use in decoder
        skip = skip[::-1]

        # --- Decoder: Upsampling path ---
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Transposed convolution (upsample)

            # Handle size mismatch due to rounding during pooling
            if x.shape != skip[idx // 2].shape:
                x = F.interpolate(x, size=skip[idx // 2].shape[2:])

            # Concatenate skip connection from encoder
            x = torch.cat((skip[idx // 2], x), dim=1)  # Channel-wise concatenation

            # Double convolution after concatenation
            x = self.ups[idx + 1](x)

        # --- Final Convolution to map to output channels ---
        return self.final_conv(x)