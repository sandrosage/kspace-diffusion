from pl_modules.mri_module import MRIModule
from torch import optim, nn
from modules.transforms import kspace_to_mri, KspaceUNetSample, norm, unnorm
from modules.unet import  Unet_FastMRI

class UNet(MRIModule):
    def __init__(self, 
                 in_channels: int = 2, 
                 out_channels: int = 2, 
                 latent_dim: int = 16,
                 n_channels: int = 128,
                 downsample_layers: int = 4,
                 with_residuals: bool = True,
                 num_log_images: int = 32,
                 norm: bool = True,
                 undersampling: bool = False):
        
        super().__init__(num_log_images=num_log_images)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.downsample_layers = downsample_layers
        self.n_channels = n_channels
        self.with_residuals = with_residuals
        self.norm = norm
        self.undersampling = undersampling
        print(f"Undersampling: {self.undersampling}")

        self.save_hyperparameters()

        # self.model = UNet2DModel(in_channels=2, out_channels=2, block_out_channels=(64, 128, 256, 512))
        self.model = Unet_FastMRI(
            in_chans=self.in_channels,
            out_chans=self.out_channels, 
            chans=self.n_channels, 
            num_pool_layers=self.downsample_layers, 
            latent_dim=self.latent_dim,
            with_residuals=self.with_residuals)
        
        self.criterion = nn.L1Loss()

    def forward(self, x):
        # return self.model.forward(x, timestep=torch.tensor(0)).sample
        output = self.model.forward(x)
        return output

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)
    
        
    def training_step(self, batch: KspaceUNetSample):
        input =  batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        if self.norm:
            input, mean, std = norm(input)
        output = self(input)
        loss = self.criterion(input, output)
        if self.norm:
            output = unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        output_img = kspace_to_mri(output)
        mri_loss = self.criterion(batch.target, output_img)
        self.log("train/mri_loss", mri_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=input.shape[0])
        self.log("train/kspace_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=input.shape[0])
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=input.shape[0])

        return {
            "loss": loss,
            "reconstructions": output_img

        }
    
    def validation_step(self, batch: KspaceUNetSample, batch_idx):
        input =  batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        if self.norm:
            input, mean, std = norm(input)
        output = self(input)
        loss = self.criterion(input, output)
        if self.norm:
            output = unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        output_img = kspace_to_mri(output)
        mri_loss = self.criterion(batch.target, output_img)
        self.log("val/mri_loss", mri_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=input.shape[0])
        self.log("val/kspace_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=input.shape[0])
        self.log("val/total_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=input.shape[0])

        return {
            "reconstructions": output_img
        }
    
    def test_step(self, batch: KspaceUNetSample, batch_idx):
        
        return_dict = {
            "reconstructions": None,
            "inputs": None,
            "outputs": None
        }
        if self.undersampling:
            input = batch.masked_kspace
        else:
            input =  batch.full_kspace
        return_dict["inputs"] = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        if self.norm:
            input, mean, std = norm(input)
        output = self(input)
        if self.norm:
            output = unnorm(output, mean, std)
        output = output.permute(0,2,3,1).contiguous()
        return_dict["outputs"] = output
        output_img = kspace_to_mri(output)
        return_dict["reconstructions"] = output_img
        
        return return_dict 
    
    def downsample(self, x):
        return self.model._downsample(x)
    
    def upsample(self, z, stack):
        return self.model._upsample(z, stack)