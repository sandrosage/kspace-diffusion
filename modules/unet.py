from torch import nn
import torch
from typing import Tuple
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

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
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)



class TransposeConvBlock(nn.Module):
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
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
    
class Unet_FastMRI(nn.Module):
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
        with_residuals: bool = True,
        latent_dim: int = 128
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
        self.with_residuals = with_residuals
        self.latent_dim = latent_dim
        self.res_weight = 16
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv_in = ConvBlock(ch, self.latent_dim, drop_prob)
        # self.proj = nn.Linear(self.latent_dim*20*20, self.latent_dim*20*20)
        self.conv_out = ConvBlock(self.latent_dim, ch*2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            if self.with_residuals:
                self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            else:
                self.up_conv.append(ConvBlock(ch, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        if self.with_residuals: 
            self.up_conv.append(
                nn.Sequential(
                    ConvBlock(ch * 2, ch, drop_prob),
                    nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
                    # optional Tanh
                    # nn.Tanh()
                )
            )
        else:
            self.up_conv.append(
                nn.Sequential(
                    ConvBlock(ch, ch, drop_prob),
                    nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
                    # optional Tanh
                    # nn.Tanh()
                )
            )
    
    def _downsample(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        stack = []
        output = x

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv_in(output)
        # output = output.view(output.size(0), -1)
        # output = self.proj(output)
        return output, stack
        
    def _upsample(self, x: torch.Tensor, stack: list) -> torch.Tensor:
        # res_weight = self.res_weight
        # x = x.view(1, self.latent_dim, 20, 20)
        x = self.conv_out(x)
        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            # downsample_layer = downsample_layer*(1/res_weight)
            x = transpose_conv(x)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if x.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if x.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                x = F.pad(x, padding, "reflect")

            if self.with_residuals:
                x = torch.cat([x, downsample_layer], dim=1)
    
            x = conv(x)
            # res_weight /= 2
        
        return x

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        z, stack = self._downsample(image)
        output = self._upsample(z, stack)
        return output
    

class Unet_NoSkip(nn.Module):
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
        latent_dim: int = 128
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
        self.latent_dim = latent_dim
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv_in = ConvBlock(ch, self.latent_dim, drop_prob)
        # self.proj = nn.Linear(self.latent_dim*20*20, self.latent_dim*20*20)
        self.conv_out = ConvBlock(self.latent_dim, ch*2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
                # optional Tanh
                # nn.Tanh()
            )
        )
    
    def _downsample(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        stack = []
        output = x

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv_in(output)
        # output = output.view(output.size(0), -1)
        # output = self.proj(output)
        return output, stack
        
    def _upsample(self, x: torch.Tensor, stack: list) -> torch.Tensor:
        # res_weight = self.res_weight
        # x = x.view(1, self.latent_dim, 20, 20)
        x = self.conv_out(x)
        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            # downsample_layer = downsample_layer*(1/res_weight)
            x = transpose_conv(x)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if x.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if x.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                x = F.pad(x, padding, "reflect")

            # if self.with_residuals:
            #     x = torch.cat([x, downsample_layer], dim=1)
    
            x = conv(x)
            # res_weight /= 2
        
        return x

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        z, stack = self._downsample(image)
        output = self._upsample(z, stack)
        return output
