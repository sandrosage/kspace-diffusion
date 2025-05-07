import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from fastmri.data.transforms import to_tensor
import numpy as np
import random
from typing import Tuple
import matplotlib.pyplot as plt
from typing import Tuple
def norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

def unnorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

def visualize_spheres(images: np.ndarray, n: int = 12):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, img in zip(axes.flat, images):

        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def generate_image_with_sphere_numpy(image_size: int = 128, n_images: int = 500, radius_range: Tuple[int,int] = (3, 10), n_spheres: int = 3):
    images = []
    for _ in range(n_images):
        # Create empty image
        image = np.zeros((image_size, image_size), dtype=np.uint8)
        for _ in range(n_spheres):
        # Random radius and center
            radius = random.randint(*radius_range)
            center_x = random.randint(radius, image_size - radius)
            center_y = random.randint(radius, image_size - radius)

            # Create coordinate grid
            Y, X = np.ogrid[:image_size, :image_size]
            
            # Create mask for pixels within the circle
            dist_from_center = (X - center_x)**2 + (Y - center_y)**2
            mask = dist_from_center <= radius**2

            # Fill the circle area with white (255)
            image[mask] = 255

            # Return shape (128, 128, 1)
        images.append(np.expand_dims(image, axis=-1))
    return np.array(images)

def to_freq_domain(x: np.ndarray) -> torch.Tensor:
    x = x.squeeze()
    x = np.fft.fft2(x)
    x = np.fft.fftshift(x)
    return to_tensor(x)

def to_image_domain(x: torch.Tensor):
    x = torch.view_as_complex(x)
    x = torch.fft.ifft2(x)
    x = torch.abs(x)
    return x

class SphereDataset(Dataset):
    def __init__(self, data: np.ndarray):
        super().__init__()
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.data.shape[0]


class Unet(nn.Module):
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
        chans: int = 512,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        kernel_size: int = 3,
        avg_pool: bool = True
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
        self.kernel_size = kernel_size
        self.avg_pool = avg_pool

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, self.kernel_size)])
        if avg_pool: 
            self.down_sample_pools = nn.ModuleList([nn.AvgPool2d(kernel_size=2, stride=2, padding=0)])
        else:
            self.down_sample_pools = nn.ModuleList([nn.Conv2d(in_channels=chans, out_channels=chans, kernel_size=2, stride=2, padding=0)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob, self.kernel_size))
            if avg_pool:
                self.down_sample_pools.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                self.down_sample_pools.append(nn.Conv2d(in_channels=ch*2, out_channels=ch*2, kernel_size=2, stride=2, padding=0))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob, self.kernel_size))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob, self.kernel_size),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        # print(self.up_conv)
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
        for layer, pool in zip(self.down_sample_layers, self.down_sample_pools):
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            output = pool(output)

        output = self.conv(output)
        print(output.shape)

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


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, kernel_size: int = 3, stride: int = 1):
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - stride) // 2

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Dropout2d(drop_prob),
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

    def __init__(self, in_chans: int, out_chans: int, kernel_size: int = 2, stride: int = 2):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.stride = stride

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=self.kernel_size, stride=self.stride, bias=False
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

from torch import optim

class Trainer():
    def __init__(self, model, train_dl, test_dl, k):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(self.device)
        self.model = model.to(self.device)
        self.train_dataloader = train_dl
        self.test_dataloader = test_dl
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.epochs = 2
        self.k = k

    def fit(self):
        input_images = []
        reconstructed_images = []
        for epoch in range(self.epochs):
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                input = to_freq_domain(batch).permute(-1, 0, 1).contiguous().unsqueeze(0).to(torch.float32)
                input = input.to(self.device)
                input, mean, std = norm(input)
                output = self.model(input)
                loss = self.criterion(input, output)
                loss.backward()
                self.optimizer.step()
                output = unnorm(output,mean,std)
                batch_reconstructed = to_image_domain(output.squeeze(0).permute(1,2,0).contiguous())
                if epoch == self.epochs-1:
                    if batch_idx %  100 == 0:
                        # print(f'Epoch {epoch+1}/{self.epochs}, Batch {batch_idx}, Loss: {loss.item()}')
                        input_images.append(batch.squeeze().cpu())
                        reconstructed_images.append(batch_reconstructed.cpu())
        return input_images, reconstructed_images

def plot_images(x, name):
    x = x[:4] + x[16:]
    fig, axes = plt.subplots(1, 8, figsize=(18, 18))
    for ax, img in zip(axes.flat, x):

        ax.imshow(img.detach(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.title(name[:-4])
    plt.show()
    plt.savefig(name, dpi=1000)

if __name__ == "__main__":
    # Example usage
    images = generate_image_with_sphere_numpy(n_images=1000)
    ds = SphereDataset(images)
    train_dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    kernel_sizes = [3,5,7,9]
    avgPool = [True, False]
    for k in kernel_sizes:
        for flag in avgPool:
            model = Unet(2, 2, 32, 2, kernel_size=k, avg_pool=flag)
            trainer = Trainer(model, train_dataloader, train_dataloader)
            input_images, reconstructed_images = trainer.fit()
            name = str(k) +  "_" + str(flag) + ".jpg"
            plot_images(reconstructed_images, name)
            plot_images(input_images, "input.jpg")