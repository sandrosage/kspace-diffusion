from modules.transforms import KspaceUNetDataTransform320, ImageUNetDataTransform320
from fastmri.pl_modules import FastMriDataModule
from pathlib import Path
from fastmri.data.subsample import create_mask_for_mask_type
import torch
from torch import optim, nn
from modules.transforms import norm, unnorm, kspace_to_mri
import matplotlib.pyplot as plt
import fastmri
from modules.unet import Unet_FastMRI

# --- Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

# --- Encoder ---
class KspaceEncoder(nn.Module):
    def __init__(self, input_channels=2, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(128 * 80 * 80, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# --- Decoder ---
class KspaceDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 80 * 80),
            nn.Unflatten(1, (128, 80, 80)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )

    def forward(self, z):
        return self.decoder(z)


if __name__ == "__main__":
    print("Starting...")
    mask_type = "random"
    center_fractions = [0.04]
    accelerations = [4]
    mask = create_mask_for_mask_type(
        mask_type, center_fractions, accelerations
    )

    train_transform = val_transform = test_transform = KspaceUNetDataTransform320(mask_func=mask)
    # train_transform = val_transform = test_transform = UnetDataTransform(which_challenge="singlecoil")
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=Path("/home/saturn/iwai/iwai113h/IdeaLab/knee_dataset"),
        challenge="singlecoil",
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=False,
        test_split="test",
        sample_rate=None,
        batch_size=1,
        num_workers=4,
        distributed_sampler=False,
        use_dataset_cache_file=True
    )

    train_dataloader = data_module.train_dataloader()

    model = Unet_FastMRI(2,2,32, 4, 0, False)
    class Trainer():
        def __init__(self,  train_dl, test_dl, model=None):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # print(self.device)
            # self.encoder = KspaceEncoder(latent_dim=512).to(self.device)
            # self.decoder = KspaceDecoder(latent_dim=512).to(self.device)
            # self.encoder.train()
            # self.decoder.train()
            self.model = model.to(self.device)
            self.train_dataloader = train_dl
            self.test_dataloader = test_dl
            self.criterion = nn.L1Loss()
            # self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4)
            self.optimizer = optim.Adam(list(self.model.parameters()), lr=1e-4)
            self.epochs = 20
        
        def fit(self):
            input_images = []
            reconstructed_images = []
            for epoch in range(self.epochs):
                for batch_idx, batch in enumerate(self.train_dataloader):
                    # mask = batch.mask
                    # mask = mask.to(self.device)
                    # input =  batch.masked_kspace
                    input = reference = batch.full_kspace
                    # input = fastmri.ifft2c(input)
                    input = input.permute(0,3,1,2).contiguous()
                    self.optimizer.zero_grad()
                    input = input.to(self.device)
                    input, mean, std = norm(input)
                    # z = self.encoder(input)
                    # # print(z.shape)
                    # output = self.decoder(z)
                    output = self.model(input)
                    loss = self.criterion(input, output)
                    loss.backward()
                    self.optimizer.step()
                    input = unnorm(input, mean, std)
                    output = unnorm(output, mean, std)
                    input = input.permute(0,2,3,1).contiguous()
                    output = output.permute(0,2,3,1).contiguous()
                    # zero = torch.zeros(1, 1, 1, 1).to(self)
                    # soft_dc = torch.where(mask, output - input, zero)
                    # output = output - soft_dc
                    # input_img = fastmri.complex_abs(input[0,...])
                    # output_img = fastmri.complex_abs(output[0,...])
                    input_img = kspace_to_mri(input[0,...], (320,320))
                    output_img = kspace_to_mri(output[0,...], (320,320))
                    # reference_img = kspace_to_mri(reference, (320,320))

                    if batch_idx %  1000 == 0:
                        print(f'Epoch {epoch+1}/{self.epochs}, Batch {batch_idx}, Loss: {loss.item()}')
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        axes[0].imshow(input_img.cpu().detach().squeeze(0),cmap="gray")
                        axes[0].set_title("Undersampled")
                        axes[1].imshow(output_img.cpu().detach().squeeze(0),cmap="gray")
                        axes[1].set_title("Undersampled Recon")
                        plt.savefig(f"kspace_1/{epoch}_{batch_idx}.png")
                        plt.close()
                        # plt.imshow(reference_img.cpu().detach().squeeze(0),cmap="gray")
                        # plt.title("Full")
                        # plt.show()
                        # input_images.append(batch.squeeze().cpu())
                        # reconstructed_images.append(batch_reconstructed.cpu())
            return input_images, reconstructed_images
        
    # batch = next(iter(train_dataloader))
    trainer = Trainer(train_dataloader, train_dataloader, model=model)
    trainer.fit()
    