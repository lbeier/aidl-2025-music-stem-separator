import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two sequential convolutional layers each followed by a ReLU activation.
    (Conv -> ReLU -> Conv -> ReLU)

    Used as a basic building block for U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 1st conv layer
            nn.ReLU(inplace=True),                                            # ReLU activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 2nd conv layer
            nn.ReLU(inplace=True),                                            # ReLU activation
        )

    def forward(self, x):
        """
        Apply two convolutional layers to input x.
        """
        return self.conv(x)

class UNetSmall(nn.Module):
    """
    Lightweight U-Net architecture.
    Encoder-decoder structure with skip connections.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale/mel/stft)
        out_channels (int): Number of output channels (e.g., 1 for predicting mask/spectrogram)
        features (list): List defining the number of channels at each layer.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        super().__init__()

        self.downs = nn.ModuleList()  # List to store downsampling (encoder) blocks
        self.ups = nn.ModuleList()    # List to store upsampling (decoder) blocks
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling between down blocks

        # --- Build Encoder (Downsampling path) ---
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))  # Double conv block
            in_channels = feature  # Update input channels for next layer

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)  # Deepest layer

        # --- Build Decoder (Upsampling path) ---
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )  # Upsample
            self.ups.append(
                DoubleConv(feature * 2, feature)
            )  # Double conv after skip connection

        # --- Final Convolution ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)  # Map to output channels

    def forward(self, x):
        """
        Forward pass through U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width)
        Returns:
            torch.Tensor: Output tensor
        """
        skip = []  # Store skip connections for later concatenation

        # --- Encoder Downsampling ---
        for down in self.downs:
            x = down(x)
            skip.append(x)  # Save output before pooling
            x = self.pool(x)  # Downsample

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # Reverse skip list to match decoder order
        skip = skip[::-1]

        # --- Decoder Upsampling ---
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample (ConvTranspose2d)

            # Fix any size mismatch due to rounding in pooling
            if x.shape != skip[idx // 2].shape:
                x = F.interpolate(x, size=skip[idx // 2].shape[2:])

            # Concatenate skip connection
            x = torch.cat((skip[idx // 2], x), dim=1)  # Channel-wise concat

            # Apply Double Conv
            x = self.ups[idx + 1](x)

        # --- Final output ---
        return self.final_conv(x)