import pytorch_lightning as pl
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import optim, nn
from modules.transforms import norm, unnorm, KspaceUNetSample
import torch
from pl_modules.diffusers_vae_module import Diffusers_VAE
import torch.nn.functional as F
from typing import Tuple, Literal

class AdaptivePoolTransform(nn.Module):
    def __init__(self, output_size: Tuple[int, int], pool_type: Literal["avg", "max"] = "avg"):
        super().__init__()
        assert pool_type in ("max", "avg"), "pooling type must either be 'max' or 'avg'"
        self.output_size = output_size
        self.pool_type = pool_type
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "avg":
            return F.adaptive_avg_pool2d(input, self.output_size)
        else:
            return F.adaptive_max_pool2d(input, self.output_size)
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_size={self.output_size}, pool_type={self.pool_type})"
    
class LDM(pl.LightningModule):
    def __init__(self, first_stage_model: Diffusers_VAE):
        super().__init__()

        self.first_stage = first_stage_model
        self.first_stage.eval()
        self.transform = AdaptivePoolTransform((640, 386))

        self.model = UNet2DModel(
            in_channels=self.first_stage.latent_dim,
            out_channels=self.first_stage.latent_dim,
            block_out_channels=(64, 128, 256),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
        )

        self.scheduler = DDPMScheduler()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    
    def diffusion(self, z: torch.Tensor) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(z)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (z.size(0),), device=self.device)
        return self.scheduler.add_noise(z, noise, steps), steps, noise

    def training_step(self, batch: KspaceUNetSample, batch_idx):
        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input = self.transform(input)
        input, mean, std = norm(input)
        
        with torch.no_grad():
            z = self.first_stage.encode(input)
    
        z_diffused, steps, noise = self.diffusion(z)
        residual = self.model(z_diffused, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)

        self.log("mse_noise_loss", loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return {
            "loss": loss,
        }