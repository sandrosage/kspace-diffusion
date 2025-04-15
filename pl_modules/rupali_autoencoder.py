from modules.rupali import VAE
from pl_modules.mri_module import MRIModule
from modules.transforms import complex_center_crop_c_h_w, kspace_to_mri
from torch import nn, optim
import torch

class RupaliAutoencoderModule(MRIModule):
    def __init__(self, in_channels: int = 2, latent_dim: int = 128):
        super().__init__()
        self.in_channels = 2 // 2
        self.latent_dim = latent_dim
        self.real_vae = VAE(self.in_channels, self.latent_dim)
        self.imag_vae = VAE(self.in_channels, self.latent_dim)
        self.criterion = nn.L1Loss()
    
    def forward(self, real, imag):
        real_output = self.real_vae(real)[0]
        imag_output = self.imag_vae(imag)[0]
        return real_output, imag_output

    def configure_optimizers(self):
        return optim.Adam(
            params= list(self.real_vae.parameters())+ list(self.imag_vae.parameters()), 
            lr=1e-3
            )
    
    def training_step(self, batch, batch_idx):
        kspace = batch.masked_kspace.permute(0,3,1,2).contiguous()
        kspace = complex_center_crop_c_h_w(kspace, (320,320))
        real, imag = kspace[:,0:1,...], kspace[:,1:2,...]
        real_output, imag_output = self(real, imag)
        kspace_output = torch.cat([real_output, imag_output], dim=1)
        real_loss = self.criterion(real, real_output)
        imag_loss = self.criterion(imag, imag_output)
        kspace_loss = self.criterion(kspace, kspace_output)
        mri_image = kspace_to_mri(kspace.permute(0,2,3,1).contiguous())
        mri_image_output = kspace_to_mri(kspace_output.permute(0,2,3,1).contiguous())
        mri_image_loss = self.criterion(mri_image, mri_image_output)

        total_loss = real_loss + imag_loss
        self.log_dict({
                "real_loss": real_loss,
                "imag_loss": imag_loss,
                "kspace_loss": kspace_loss,
                "mri_loss": mri_image_loss
             }, 
            on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size= kspace.shape[0]
        )
        return {
            "loss": total_loss,
            "input": kspace.permute(0,2,3,1).contiguous(), 
            "reconstruction": kspace_output.permute(0,2,3,1).contiguous(),
            "rec_img": mri_image_output,
            "target": mri_image
        }





