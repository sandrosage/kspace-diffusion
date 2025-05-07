import torch
import diffusers
from datasets import load_dataset
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb
import random
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
import numpy as np

def min_max_scale_array(arr):
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val) / (max_val - min_val)

class DiffusionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = diffusers.models.UNet2DModel(
            block_out_channels=(64, 128, 256),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
        )
        self.scheduler = diffusers.schedulers.DDPMScheduler()
        self.train_log_indices = None
        self.num_log_images = 128

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return {
            "loss": loss,
            "residual": residual,
            "noisy_images": noisy_images
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.train_log_indices is None:
            self.train_log_indices = list(
                np.random.permutation(len(self.trainer.train_dataloader))[
                    : self.num_log_images - 1
                ]
            )
        self.train_log_indices.insert(0, 1)
        if batch_idx in self.train_log_indices:
            images = batch["images"]
            residual = outputs["residual"]
            noisy_images = outputs["noisy_images"]
            denoised_images = noisy_images - residual
            self.log_image(batch_idx, residual, noisy_images, denoised_images, images)
    
    def log_image(self, batch_idx, residual: torch.Tensor, noisy_images: torch.Tensor, denoised_images: torch.Tensor, images: torch.Tensor):

        idx = random.randint(0, len(residual)-1)
        fig, ax = plt.subplots(1,4,figsize=(18,5))
        fig.subplots_adjust(wspace=0.0)
        ax[0].imshow(min_max_scale_array(residual[idx,...].cpu().detach().permute(1,2,0).numpy()))
        ax[0].set_title("Noise")

        ax[1].imshow(min_max_scale_array(noisy_images[idx,...].cpu().detach().permute(1,2,0).numpy()))
        ax[1].set_title("Noisy image")

        ax[2].imshow(min_max_scale_array(denoised_images[idx,...].cpu().detach().permute(1,2,0).numpy()))
        ax[2].set_title("Denoised image")

        ax[3].imshow(min_max_scale_array(images[idx,...].cpu().detach().permute(1,2,0).numpy()))
        ax[3].set_title("Original image")
        
        
        # remove all the ticks (both axes), and tick labels
        for axes in ax:
            axes.set_xticks([])
            axes.set_yticks([])
        # remove the frame of the chart
        for axes in ax:
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)
        # remove the white space around the chart
        plt.tight_layout()
        self.logger.experiment.log({'images/{}_Grid.png'.format(batch_idx) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()

class DiffusionData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.augment = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def prepare_data(self):
        load_dataset("cifar10")

    def train_dataloader(self):
        dataset = load_dataset("cifar10")
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return torch.utils.data.DataLoader(dataset["train"], batch_size=128, num_workers=4)


if __name__ == "__main__":
    config = {
        "test": 32,
    }
    run_name = "Diffusion_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    wandb.login(key="c210746318a0cf3a3fb1d542db1864e0a789e94c")
    wandb_logger = WandbLogger(project="Kspace-Unet", name=run_name, log_model=True, config=config)
    model = DiffusionModel()
    data = DiffusionData()
    trainer = pl.Trainer(max_epochs=150, logger=wandb_logger)
    trainer.fit(model, data)

