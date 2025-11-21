import pytorch_lightning as pl
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from contextlib import contextmanager
from torch import optim
from modules.transforms import norm, unnorm, LDMSample, kspace_to_mri 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import wandb
from fastmri.evaluate import nmse
from modules.losses import ssim, psnr
from pytorch_lightning.utilities.rank_zero import rank_zero_only

def to_range(step: torch.Tensor, device):
    return torch.arange(int(step.item()), -1, -1, device=device)

    
class LDM(pl.LightningModule):
    def __init__(self, first_stage, in_channels: int, out_channels: int, rescale_latents: bool = False, normalize_latents: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_stage = first_stage
        for p in self.first_stage.parameters():
            p.requires_grad = False
        self.first_stage = self.first_stage.eval()
        self.val_log_indices = None
        self.num_log_images = 32
        self.scale_factor = 1.0
        self.rescale_latents = rescale_latents
        self.normalize_latents = normalize_latents
        self.model = UNet2DModel(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            block_out_channels=( 128, 256, 512, 512),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

        self.scheduler = DDPMScheduler(variance_type="fixed_small", clip_sample=False, timestep_spacing="trailing", beta_schedule="squaredcos_cap_v2", num_train_timesteps=1000)
        # self.scheduler.set_timesteps(1000)
        self.save_hyperparameters(ignore=['first_stage'])
        if self.rescale_latents:
            print("Rescaling latents")
        if self.normalize_latents:
            print("Normalizing latents")

    def configure_optimizers(self):
        optimizer = optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    
    def diffusion(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(z).contiguous()
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (z.size(0),), device=self.device, dtype=torch.long)
        z_diffused = self.scheduler.add_noise(z, noise, steps).contiguous()
        return z_diffused, steps, noise

    def training_step(self, batch: LDMSample, batch_idx):
        input = batch.full_latent_tensor.contiguous()
        z = self.scale_factor * input 
        if self.normalize_latents:
            z, mean, std = norm(z)
        z_diffused, steps, noise = self.diffusion(z)
        residual = self.model(z_diffused, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise, reduction="mean")

        self.log("train/mse_noise_loss", loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_latent_tensor.shape[0])
        return {
            "loss": loss,
            "diffused_samples": z_diffused,
        }
    
    def validation_step(self, batch: LDMSample, batch_idx):
        input = batch.full_latent_tensor.contiguous()
        z = self.scale_factor * input
        if self.normalize_latents:
            z, mean, std = norm(z)
        else:
            mean = None
            std = None
        z_diffused, steps, noise = self.diffusion(z)
        residual = self.model(z_diffused, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        
        self.log("val/mse_noise_loss", loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_latent_tensor.shape[0])

        return {
            "loss": loss,
            "diffused_samples": z_diffused,
            "z_mean": mean,
            "z_std": std,
            "t": steps
        }
    
    def on_validation_batch_end(self, outputs, batch: LDMSample, batch_idx):

        for k in ("diffused_samples", "z_mean", "z_std", "t"):
            if k not in outputs.keys():
                raise RuntimeError(f"Expected key {k} in dict returned by validation_step")
            
        targets = batch.target
        reconstructions = batch.full_latent_tensor
        diffused_samples = outputs["diffused_samples"]
        z_mean, z_std = outputs["z_mean"], outputs["z_std"]

        if self.val_log_indices is None:
            self.val_log_indices = list([1]) + list(
                np.random.permutation(len(self.trainer.val_dataloaders))[
                    : self.num_log_images - 1
                ]
            )

        if batch_idx in self.val_log_indices:
            idx = random.sample(range(reconstructions.size(0)), 1)[0]
            target = targets[idx]
            reconstruction = reconstructions[idx].unsqueeze(0)
            diffused_sample = diffused_samples[idx].unsqueeze(0)
            current_step = outputs["t"][idx]
            slice_num = batch.slice_num[idx]
            fname = batch.fname[idx]
            target = target / target.max()
            print(f"Current step: {current_step}")
            # print(f"Sampling T={self.scheduler.timesteps[0]+1} ...")
            for t in to_range(current_step, self.device):
                    with torch.inference_mode():
                        pred_noise = self.model(diffused_sample,t).sample
                        diffused_sample = self.scheduler.step(pred_noise, t, diffused_sample).prev_sample
            if self.normalize_latents:
                diffused_sample = unnorm(diffused_sample, z_mean[idx], z_std[idx])
            diffused_sample = self.scale_factor * diffused_sample

            target = target.squeeze(0).cpu().numpy()
            with torch.inference_mode():
                diffused_sample = self.first_stage.decode(diffused_sample)
                reconstruction = self.first_stage.decode(reconstruction)
            reconstruction = unnorm(reconstruction, batch.mean_full[idx], batch.std_full[idx])
            reconstruction = reconstruction.permute(0, 2, 3, 1).contiguous()
            reconstruction = kspace_to_mri(reconstruction)
            reconstruction = reconstruction.squeeze(0).cpu().numpy()
            diffused_sample = unnorm(diffused_sample, batch.mean_full[idx], batch.std_full[idx])
            diffused_sample = diffused_sample.permute(0, 2, 3, 1).contiguous()
            diffused_sample = kspace_to_mri(diffused_sample)
            diffused_sample = diffused_sample.squeeze(0).detach().cpu().numpy()
            self.log_image(fname, batch_idx, slice_num, target, reconstruction, diffused_sample, "val")
            # self.calculate_metrics(diffused_sample, target, None, "val")

    def log_image(self, fname, batch_idx, slice_num, target, reconstruction, diffused_sample, flag):
        fig, ax = plt.subplots(1,3,figsize=(18,5))
        fig.subplots_adjust(wspace=0.0)
    
        ax[0].imshow(target,'gray')
        ax[0].set_title("Target")

        ax[1].imshow(reconstruction,'gray')
        ax[1].set_title("Reconstruction")

        ax[2].imshow(diffused_sample, 'gray')
        ax[2].set_title("Model sampling")
        
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
        self.logger.experiment.log({'images/{}/{}_{}_{}_{}_Grid.png'.format(self.trainer.current_epoch, flag, fname[:-3], batch_idx, str(slice_num.cpu().numpy())) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()
    
    def calculate_metrics(self, reconstructions: torch.Tensor, targets: torch.Tensor, max_vals: torch.Tensor, flag:str):
        targets = np.expand_dims(targets, axis=0)
        reconstructions = np.expand_dims(reconstructions, axis=0)
        
        SSIM = torch.Tensor(ssim(targets, reconstructions, max_vals)).to(self.device)
        PSNR = torch.Tensor(psnr(targets, reconstructions, max_vals)).to(self.device)
        NMSE = torch.Tensor(nmse(targets, reconstructions)).to(self.device)
        self.log_dict({
            f"{flag}/ssim": SSIM,
            f"{flag}/psnr": PSNR,
            f"{flag}/nmse": NMSE,
        }, on_epoch=True, on_step=True, sync_dist=True, batch_size=reconstructions.shape[0])

    def on_train_batch_start(self, batch, batch_idx):
        if self.rescale_latents:
            z = batch.full_latent_tensor.contiguous()
            self.scale_factor = 1.0 / z.flatten().std()

class EmaLDM(pl.LightningModule):
    def __init__(self, first_stage, block_out_channels, in_channels: int, out_channels: int, rescale_latents: bool = False, normalize_latents: bool = False):
            super().__init__()

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.first_stage = first_stage
            for p in self.first_stage.parameters():
                p.requires_grad = False
            self.first_stage.eval()
            self.val_log_indices = None
            self.num_log_images = 32
            self.scale_factor = 1.0
            self.rescale_latents = rescale_latents
            self.normalize_latents = normalize_latents
            if block_out_channels is not None:
                block_out_channels = block_out_channels
            else:
                block_out_channels = ( 128, 256, 512, 512)
            self.model = UNet2DModel(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                block_out_channels=( 128, 256, 512, 512),
                down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
                up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            )

            self.ema = EMAModel(
                self.model.parameters(), 
                device=self.device
            )
            self.ema.to(self.device)

            self.scheduler = DDPMScheduler(variance_type="fixed_small", clip_sample=False, timestep_spacing="trailing", beta_schedule="squaredcos_cap_v2", num_train_timesteps=1000)
            # self.scheduler.set_timesteps(1000)
            self.save_hyperparameters(ignore=['first_stage'])
            if self.rescale_latents:
                print("Rescaling latents")
            if self.normalize_latents:
                print("Normalizing latents")
            
    def on_fit_start(self):
        self.ema.to(self.device)

    def configure_optimizers(self):
        optimizer = optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    
    def diffusion(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(z).contiguous()
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (z.size(0),), device=self.device, dtype=torch.long)
        z_diffused = self.scheduler.add_noise(z, noise, steps).contiguous()
        return z_diffused, steps, noise

    def training_step(self, batch: LDMSample, batch_idx):
        input = batch.full_latent_tensor.contiguous()
        z = self.scale_factor * input 
        if self.normalize_latents:
            z, mean, std = norm(z)
        z_diffused, steps, noise = self.diffusion(z)
        residual = self.model(z_diffused, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise, reduction="mean")

        self.log("train/mse_noise_loss", loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_latent_tensor.shape[0])
        return {
            "loss": loss,
            "diffused_samples": z_diffused,
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        optimizer.step(closure=optimizer_closure)
        self.ema.step(self.model.parameters())
        optimizer.zero_grad()
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
    
    @contextmanager
    def ema_weights(self):
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        try:
            yield
        finally:
            self.ema.restore(self.model.parameters())
    
    @torch.no_grad()
    def validation_step(self, batch: LDMSample, batch_idx):
        input = batch.full_latent_tensor.contiguous()
        z = self.scale_factor * input
        if self.normalize_latents:
            z, mean, std = norm(z)
        else:
            mean = None
            std = None
        z_diffused, steps, noise = self.diffusion(z)
        residual = self.model(z_diffused, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        
        # if batch_idx % 100 == 0:
        self.log("val/mse_noise_loss", loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=False, batch_size=batch.full_latent_tensor.shape[0])
        return {
            "loss": loss,
            "diffused_samples": z_diffused,
            "z_mean": mean,
            "z_std": std,
            "t": steps
        }
    
    @torch.no_grad()
    def on_validation_batch_end(self, outputs, batch: LDMSample, batch_idx):

        for k in ("diffused_samples", "z_mean", "z_std", "t"):
            if k not in outputs.keys():
                raise RuntimeError(f"Expected key {k} in dict returned by validation_step")
            
        targets = batch.target
        reconstructions = batch.full_latent_tensor
        diffused_samples = outputs["diffused_samples"]
        z_mean, z_std = outputs["z_mean"], outputs["z_std"]

        if self.val_log_indices is None:
            self.val_log_indices = list([1]) + list(
                np.random.permutation(len(self.trainer.val_dataloaders))[
                    : self.num_log_images - 1
                ]
            )

        if batch_idx in self.val_log_indices:
            idx = random.sample(range(reconstructions.size(0)), 1)[0]
            target = targets[idx]
            reconstruction = reconstructions[idx].unsqueeze(0)
            diffused_sample = diffused_samples[idx].unsqueeze(0)
            current_step = outputs["t"][idx]
            slice_num = batch.slice_num[idx]
            fname = batch.fname[idx]
            target = target / target.max()
            # print(f"Sampling T={self.scheduler.timesteps[0]+1} ...")
            with self.ema_weights():
                for t in to_range(current_step, self.device):
                        with torch.inference_mode():
                            pred_noise = self.model(diffused_sample,t).sample
                            diffused_sample = self.scheduler.step(pred_noise, t, diffused_sample).prev_sample
            if self.normalize_latents:
                diffused_sample = unnorm(diffused_sample, z_mean[idx], z_std[idx])
            diffused_sample = self.scale_factor * diffused_sample

            target = target.squeeze(0).cpu().numpy()
            with torch.inference_mode():
                diffused_sample = self.first_stage.decode(diffused_sample)
                reconstruction = self.first_stage.decode(reconstruction)
            reconstruction = unnorm(reconstruction, batch.mean_full[idx], batch.std_full[idx])
            reconstruction = reconstruction.permute(0, 2, 3, 1).contiguous()
            reconstruction = kspace_to_mri(reconstruction)
            reconstruction = reconstruction.squeeze(0).cpu().numpy()
            diffused_sample = unnorm(diffused_sample, batch.mean_full[idx], batch.std_full[idx])
            diffused_sample = diffused_sample.permute(0, 2, 3, 1).contiguous()
            diffused_sample = kspace_to_mri(diffused_sample)
            diffused_sample = diffused_sample.squeeze(0).detach().cpu().numpy()
            self.log_image(fname, current_step, slice_num, target, reconstruction, diffused_sample, "val")
            # self.calculate_metrics(diffused_sample, target, None, "train")

    @rank_zero_only
    def log_image(self, fname, batch_idx, slice_num, target, reconstruction, diffused_sample, flag):
        fig, ax = plt.subplots(1,3,figsize=(18,5))
        fig.subplots_adjust(wspace=0.0)
    
        ax[0].imshow(target,'gray')
        ax[0].set_title("Target")

        ax[1].imshow(reconstruction,'gray')
        ax[1].set_title("Reconstruction")

        ax[2].imshow(diffused_sample, 'gray')
        ax[2].set_title("Model sampling")
        
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
        self.logger.experiment.log({'images/{}/{}_{}_{}_{}_Grid.png'.format(self.trainer.current_epoch, flag, fname[:-3], batch_idx, str(slice_num.cpu().numpy())) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()
    
    @torch.no_grad()
    def calculate_metrics(self, reconstructions: torch.Tensor, targets: torch.Tensor, max_vals: torch.Tensor, flag:str):
        targets = np.expand_dims(targets, axis=0)
        reconstructions = np.expand_dims(reconstructions, axis=0)
        
        SSIM = torch.Tensor(ssim(targets, reconstructions, max_vals)).to(self.device)
        PSNR = torch.Tensor(psnr(targets, reconstructions, max_vals)).to(self.device)
        NMSE = torch.Tensor(nmse(targets, reconstructions)).to(self.device)
        self.log_dict({
            f"{flag}/ssim": SSIM,
            f"{flag}/psnr": PSNR,
            f"{flag}/nmse": NMSE,
        }, on_epoch=True, on_step=True, sync_dist=True, batch_size=reconstructions.shape[0])

    def on_train_batch_start(self, batch, batch_idx):
        if self.rescale_latents:
            z = batch.full_latent_tensor.contiguous()
            self.scale_factor = 1.0 / z.flatten().std()