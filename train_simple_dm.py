import torch
import diffusers
from datasets import load_dataset
from torchvision import transforms
import pytorch_lightning as L


class DiffusionModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = diffusers.models.UNet2DModel(
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            sample_size=32,
        )
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            variance_type="fixed_large", clip_sample=False, timestep_spacing="trailing",
        )

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        print(images.shape)
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    

class DiffusionData(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.augment = transforms.Compose([
            transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
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
        return torch.utils.data.DataLoader(dataset["train"], batch_size=128, shuffle=True, num_workers=4)


if __name__ == "__main__":
    model = DiffusionModel()
    data = DiffusionData()
    trainer = L.Trainer(max_epochs=150)
    trainer.fit(model, data)