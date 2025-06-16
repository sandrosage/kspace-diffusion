from pl_modules.mri_module import NewMRIModule
from modules.unet import Unet_FastMRI
from torch import nn
import torch
from typing import Tuple, Literal
from modules.transforms import KspaceUNetSample, kspace_to_mri
from fastmri.data.transforms import complex_center_crop

class MyUnetModule(NewMRIModule):
    def __init__(self, 
                 n_channels: int, 
                 loss_domain: Literal["kspace", "image", "combined", "ssim"] = "kspace", 
                 with_residual: bool = True,
                 num_log_images = 32,
                 latent_dim = 128, 
                 mode: Literal["interpolation", "reconstruction"] = "interpolation"
        ):
        super().__init__(num_log_images)

        assert loss_domain in ("kspace", "image", "combined", "ssim"), "The loss domain can either be 'kspace', 'image', 'combined' or 'ssim'"
        assert mode in ("interpolation", "reconstruction")
        self.model = Unet_FastMRI(2, 2, n_channels, 4, with_residuals=with_residual, latent_dim=latent_dim)
        self.criterion = nn.L1Loss()
        self.mode = mode
        if self.mode == "interpolation":
            self.with_dc = True
        else:
            self.with_dc = False
        # if soft_dc: 
        #     self.dc_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        # else:
        self.dc_weight = torch.ones(1)
    
        self.loss_domain = loss_domain
        self.lr = 0.0003
        self.lr_step_size = 40
        self.lr_gamma = 0.1
        self.weight_decay = 0.0
        self.save_hyperparameters()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def upsample(self, x: torch.Tensor, stack: list) -> torch.Tensor:
        return self.model._upsample(x, stack)
    
    def downsample(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        return self.model._downsample(x)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        input = masked_kspace.permute(0,3,1,2).contiguous()
        input, mean, std = self.norm(input)
        output = self.model(input)
        output = self.unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        if self.with_dc: 
            zero = torch.zeros(1, 1, 1, 1).to(output)
            soft_dc = torch.where(mask, output - masked_kspace, zero) * self.dc_weight.to(output)
            output = output - soft_dc
        return output

    def training_step(self, batch: KspaceUNetSample, batch_idx):
        full_kspace, reconstructed_kspace, full_mri_image, undersampled_mri_image, reconstructed_mri_image = self.shared_step(batch)
        loss = self.criterion(full_kspace, reconstructed_kspace)
        mri_loss = self.criterion(full_mri_image, reconstructed_mri_image)

        self.log_dict({
            "train/kspace_loss": loss, 
            "train/mri_loss": mri_loss,
            "train/dc_weight": self.dc_weight.detach(),
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=full_kspace.shape[0])

        if self.loss_domain == "image":
            loss = mri_loss
        if self.loss_domain == "combined":
            loss += mri_loss
        if self.loss_domain == "ssim":
            loss = self.SSIM(reconstructed_mri_image, full_mri_image, batch.max_value)
        return {
            "loss": loss, 
            "undersampled_mri_image": undersampled_mri_image,
            "reconstructed_mri_image": reconstructed_mri_image,
            "full_mri_image": full_mri_image
        }

    def validation_step(self, batch, batch_idx):
        full_kspace, reconstructed_kspace, full_mri_image, undersampled_mri_image, reconstructed_mri_image = self.shared_step(batch)
        loss = self.criterion(full_kspace, reconstructed_kspace)
        mri_loss = self.criterion(full_mri_image, reconstructed_mri_image)

        self.log_dict({
            "val/kspace_loss": loss, 
            "val/mri_loss": mri_loss,
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=False, batch_size=full_kspace.shape[0])

        if self.loss_domain == "image":
            loss = mri_loss
        if self.loss_domain == "combined":
            loss += mri_loss
        if self.loss_domain == "ssim":
            loss = self.SSIM(reconstructed_mri_image, full_mri_image, batch.max_value)
        return {
            "loss": loss, 
            "undersampled_mri_image": undersampled_mri_image,
            "reconstructed_mri_image": reconstructed_mri_image,
            "full_mri_image": full_mri_image
        }
    
    def test_step(self, batch, batch_idx):
        full_kspace, reconstructed_kspace, full_mri_image, undersampled_mri_image, reconstructed_mri_image = self.shared_step(batch)
        loss = self.criterion(full_kspace, reconstructed_kspace)
        mri_loss = self.criterion(full_mri_image, reconstructed_mri_image)

        self.log_dict({
            "test/kspace_loss": loss, 
            "test/mri_loss": mri_loss,
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=False, batch_size=full_kspace.shape[0])

        if self.loss_domain == "image":
            loss = mri_loss
        if self.loss_domain == "combined":
            loss += mri_loss
        if self.loss_domain == "ssim":
            loss = self.SSIM(reconstructed_mri_image, full_mri_image, batch.max_value)
        return {
            "loss": loss, 
            "undersampled_mri_image": undersampled_mri_image,
            "reconstructed_mri_image": reconstructed_mri_image,
            "full_mri_image": full_mri_image
        }
        

    def shared_step(self, batch: KspaceUNetSample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # full_kspace = complex_center_crop(batch.full_kspace, (320,320))
        full_kspace = batch.full_kspace
        if self.mode == "interpolation":
            masked_kspace = batch.masked_kspace
        else:
            masked_kspace = batch.full_kspace
        # masked_kspace = complex_center_crop(masked_kspace, (320,320))
        mask = batch.mask
        reconstructed_kspace = self(masked_kspace, mask)
        full_mri_image = kspace_to_mri(full_kspace, (320,320))
        undersampled_mri_image = kspace_to_mri(masked_kspace, (320,320))
        reconstructed_mri_image = kspace_to_mri(reconstructed_kspace, (320,320))
        return full_kspace, reconstructed_kspace, full_mri_image, undersampled_mri_image, reconstructed_mri_image
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]