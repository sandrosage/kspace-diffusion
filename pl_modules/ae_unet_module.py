
from pl_modules.mri_module import NewMRIModule
from modules.unet import Unet_FastMRI, Unet_NoSkip
from torch.nn import L1Loss
from torch import optim
from modules.transforms import norm, unnorm, kspace_to_mri, KspaceUNetSample
import torch
from fastmri.losses import SSIMLoss

class Kspace_AE_Unet(NewMRIModule):
    def __init__(self, num_log_images = 32, n_channels: int = 32, latent_dim: int = 128):
        super().__init__(num_log_images)
        self.model = Unet_NoSkip(
            in_chans=2,
            out_chans=2, 
            chans=n_channels, 
            num_pool_layers=4, 
            latent_dim=latent_dim,
        )
        self.criterion = L1Loss()

        self.save_hyperparameters()

    def configure_optimizers(self):
        return optim.Adam(list(self.model.parameters()), lr=1e-4)
    
    def shared_step(self, batch: KspaceUNetSample):
        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)
        output = self(input)
        loss = self.criterion(input, output)
        output = unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        return loss, output

    
    def training_step(self, batch: KspaceUNetSample, batch_idx):
        loss, output = self.shared_step(batch)

        input_img = kspace_to_mri(batch.full_kspace)
        output_img = kspace_to_mri(output)

        img_loss = self.criterion(input_img, output_img)

        self.log_dict({
            "train/kspace_l1": loss, 
            "train/img_l1": img_loss
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_kspace.shape[0])

        return {
            "loss": loss,
            "undersampled_mri_image": input_img,
            "reconstructed_mri_image": output_img, 
            "full_mri_image": batch.target
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, self.stack = self.model._downsample(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model._upsample(z, self.stack)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def validation_step(self, batch, batch_idx):
        loss, output = self.shared_step(batch)
    
        input_img = kspace_to_mri(batch.full_kspace)
        output_img = kspace_to_mri(output)

        img_loss = self.criterion(input_img, output_img)

        self.log_dict({
            "val/kspace_l1": loss, 
            "val/img_l1": img_loss
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_kspace.shape[0])
        
        return {
            "loss": loss,
            "undersampled_mri_image": input_img,
            "reconstructed_mri_image": output_img, 
            "full_mri_image": batch.target
        }
        
    
class Kspace_AE_Unet_SSIM(NewMRIModule):
    def __init__(self, num_log_images = 32, n_channels: int = 32, latent_dim: int = 128):
        super().__init__(num_log_images)
        self.model = Unet_NoSkip(
            in_chans=2,
            out_chans=2, 
            chans=n_channels, 
            num_pool_layers=4, 
            latent_dim=latent_dim,
        )
        self.criterion = L1Loss()
        self.SSIM = SSIMLoss()

        self.save_hyperparameters()

    def configure_optimizers(self):
        return optim.Adam(list(self.model.parameters()), lr=1e-4)
    
    def shared_step(self, batch: KspaceUNetSample):
        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)
        output = self(input)
        output = unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        return output

    
    def training_step(self, batch: KspaceUNetSample, batch_idx):
        output = self.shared_step(batch)

        input_img = kspace_to_mri(batch.full_kspace)
        output_img = kspace_to_mri(output)

        img_loss = self.criterion(input_img, output_img)

        loss = self.SSIM(output_img, batch.target, data_range=batch.max_value)

        self.log_dict({
            "train/ssim": loss, 
            "train/img_l1": img_loss
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_kspace.shape[0])

        return {
            "loss": loss,
            "undersampled_mri_image": input_img,
            "reconstructed_mri_image": output_img, 
            "full_mri_image": batch.target
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, self.stack = self.model._downsample(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model._upsample(z, self.stack)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
    
        input_img = kspace_to_mri(batch.full_kspace)
        output_img = kspace_to_mri(output)

        img_loss = self.criterion(input_img, output_img)

        loss = self.SSIM(output_img, batch.target, data_range=batch.max_value)

        self.log_dict({
            "val/ssim": loss, 
            "val/img_l1": img_loss
        }, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch.full_kspace.shape[0])
        
        return {
            "loss": loss,
            "undersampled_mri_image": input_img,
            "reconstructed_mri_image": output_img, 
            "full_mri_image": batch.target
        }
        