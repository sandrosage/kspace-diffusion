from  complexPyTorch.complexLayers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexMaxPool2d, ComplexConvTranspose2d, ComplexDropout2d
from complexPyTorch.complexFunctions import complex_avg_pool2d
from torch import nn
import torch.nn.functional as F
import torch

class ComplexDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            self.c1 = ComplexConv2d(in_channels, mid_channels, kernel_size=3, padding=1,  bias=False)
            self.bn1 = ComplexBatchNorm2d(mid_channels)
            self.r1 = ComplexReLU()
            self.c2 = ComplexConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = ComplexBatchNorm2d(out_channels)
            self.r2 = ComplexReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = self.r1(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.r2(x)
        return x
class ComplexOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class ComplexUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ComplexDoubleConv(n_channels, 64)
        self.down1 = ComplexDown(64, 128)
        self.down2 = ComplexDown(128, 256)
        self.down3 = ComplexDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = ComplexDown(512, 1024 // factor)
        self.up1 = ComplexUp(1024, 512 // factor)
        self.up2 = ComplexUp(512, 256 // factor)
        self.up3 = ComplexUp(256, 128 // factor)
        self.up4 = ComplexUp(128, 64)
        self.outc = ComplexOutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class ComplexConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            ComplexConv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(out_chans),
            ComplexReLU(),
            # ComplexDropout2d(drop_prob),
            ComplexConv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(out_chans),
            ComplexReLU(),
            # ComplexDropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)
    
class ComplexDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            ComplexMaxPool2d(2),
            ComplexDoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class ComplexUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = ComplexConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = ComplexDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ComplexTransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            ComplexConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            ComplexBatchNorm2d(out_chans),
            ComplexReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
    
class Complex_FastMRI(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ComplexConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ComplexConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ComplexConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(ComplexTransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ComplexConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(ComplexTransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ComplexConvBlock(ch * 2, ch, drop_prob),
                ComplexConv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = complex_avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output