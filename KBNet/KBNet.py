import torch
import torch.nn as nn
import torch.nn.functional as F

class KBAModule(nn.Module):
    def __init__(self, in_channels, kernel_basis_count=32, kernel_size=3):
        super(KBAModule, self).__init__()
        self.kernel_basis_count = kernel_basis_count
        self.kernel_size = kernel_size

        # Learnable kernel bases
        self.kernel_bases = nn.Parameter(torch.randn(kernel_basis_count, in_channels, kernel_size, kernel_size))

        # Lightweight convolution branch to predict fusion coefficients
        self.fusion_coeff_conv = nn.Sequential(
            nn.Conv2d(in_channels, kernel_basis_count, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_basis_count, kernel_basis_count, kernel_size=3, padding=1)
        )

        # 1x1 convolution to transform the input feature map
        self.feature_transform = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Predict fusion coefficients
        fusion_coeffs = self.fusion_coeff_conv(x)

        # Transform input features
        x_transformed = self.feature_transform(x)

        # Initialize the output feature map
        output = torch.zeros_like(x_transformed)

        # Fuse the kernel bases linearly to obtain adaptive and diverse aggregation weights
        for i in range(self.kernel_basis_count):
            basis = self.kernel_bases[i]
            coeff = fusion_coeffs[:, i, :, :].unsqueeze(1)
            output += F.conv2d(x_transformed, basis.unsqueeze(0), padding=self.kernel_size // 2, groups=1) * coeff

        return output

class MFFBlock(nn.Module):
    def __init__(self, in_channels):
        super(MFFBlock, self).__init__()

        # Channel attention branch
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Depthwise convolution branch
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

        # KBA module branch
        self.kba_module = KBAModule(in_channels)

        # Layer normalization over channel dimension
        self.layer_norm = nn.LayerNorm(in_channels)

        # 1x1 convolution
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Layer normalization over channel dimension
        x_norm = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Channel attention
        ca = self.channel_attention(x_norm) * x_norm

        # Depthwise convolution
        dw = self.depthwise_conv(x_norm)

        # KBA module
        kba = self.kba_module(x_norm)

        # Point-wise multiplication to fuse features
        fused = ca * dw * kba

        # 1x1 convolution to produce final output
        output = self.pointwise_conv(fused)

        return output

class UNetKBNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNetKBNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.size() != skip_connection.size():
                x = F.interpolate(x, size=skip_connection.size()[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)

        return self.final_conv(x)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MFFBlock(out_channels)
        )

# Example usage
if __name__ == "__main__":
    model = UNetKBNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)  # Example input
    output = model(x)
    print(output.shape)
