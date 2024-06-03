import torch
import torch.nn as nn

class UNet_S(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation.

    The U-Net architecture consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) that enables precise localization. This implementation includes:

    - Encoder: Four convolutional blocks with downsampling via max pooling.
    - Decoder: Four convolutional blocks with upsampling via transposed convolutions.
    - Skip Connections: Each layer in the encoder is connected to the corresponding layer in the decoder.

    Attributes:
        enc1 (nn.Sequential): First encoder block.
        enc2 (nn.Sequential): Second encoder block.
        enc3 (nn.Sequential): Third encoder block.
        enc4 (nn.Sequential): Fourth encoder block.
        pool (nn.MaxPool2d): Max pooling layer to downsample the feature maps.
        upconv4 (nn.ConvTranspose2d): Upconvolution layer to upsample the feature maps.
        upconv3 (nn.ConvTranspose2d): Upconvolution layer to upsample the feature maps.
        upconv2 (nn.ConvTranspose2d): Upconvolution layer to upsample the feature maps.
        dec4 (nn.Sequential): First decoder block.
        dec3 (nn.Sequential): Second decoder block.
        dec2 (nn.Sequential): Third decoder block.
        dec1 (nn.Sequential): Fourth decoder block with output layer.
    """

    def __init__(self):
        super(UNet_S, self).__init__()
        self.enc1 = self.conv_block(1, 64)  # Input channels = 1 for grayscale images
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.upconv4 = self.upconv(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.upconv2 = self.upconv(128, 64)
        self.dec4 = self.conv_block(512, 256)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = self.conv_block(64, 1, final_layer=True)  # Output channels = 1 for grayscale images

    def conv_block(self, in_channels, out_channels, final_layer=False):
        """
        Create a convolutional block with two convolutional layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            final_layer (bool): If True, applies Tanh activation. Otherwise, applies ReLU activation.

        Returns:
            nn.Sequential: Convolutional block.
        """
        if final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def upconv(self, in_channels, out_channels):
        """
        Create an upconvolution (transposed convolution) layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.ConvTranspose2d: Upconvolution layer.
        """
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        dec4 = self.dec4(torch.cat((self.upconv4(enc4), enc3), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc2), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc1), dim=1))
        dec1 = self.dec1(dec2)
        return dec1

# Example usage
if __name__ == "__main__":
    model = UNet_S()
    x = torch.randn((1, 1, 256, 256))  # Example input tensor
    y = model(x)
    print(y.shape)  # Should output torch.Size([1, 1, 256, 256])
