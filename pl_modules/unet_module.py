from pl_modules.mri_module import MRIModule
from torch import optim, nn
from modules.transforms import kspace_to_mri,complex_center_crop_c_h_w
import torch
import fastmri.data.transforms as fT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import functional as F
import time
from modules.unet import NormUnet, Unet_FastMRI 
from modules.complex_unet import Complex_FastMRI, ComplexUNet
from typing import Tuple
import fastmri.data.transforms as fT

class UNet(MRIModule):
    def __init__(self):
        super().__init__()
        # self.model = UNet2DModel(in_channels=2, out_channels=2, block_out_channels=(64, 128, 256, 512))
        self.model = Unet_FastMRI(2,2, 128, 4)
        self.criterion = nn.L1Loss()

    def forward(self, x):
        # return self.model.forward(x, timestep=torch.tensor(0)).sample
        output = self.model.forward(x)
        return output

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)
    
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
        
    def training_step(self, batch, batch_idx):
        input =  fT.complex_center_crop(batch.kspace, (320,320))
        input = input.permute(0,3,1,2).contiguous()
        # input, mean, std = self.norm(input)
        output = self(input)
        loss = self.criterion(input, output)
        # input = self.unnorm(input, mean, std)
        # output = self.unnorm(output, mean, std)
        input = input.permute(0,2,3,1).contiguous()
        output = output.permute(0,2,3,1).contiguous()
        rec_img = kspace_to_mri(output)
        target = kspace_to_mri(input)
        mri_loss = self.criterion(target, rec_img)
        self.log("mri_loss", mri_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch.kspace.shape[0])
        self.log("kspace_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch.kspace.shape[0])
        self.log("total_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch.kspace.shape[0])

        return {
            "loss": mri_loss,
            "input": input, 
            "reconstruction": output,
            "rec_img": rec_img,
            "target": target

        }
    
class ComplexUNetModule(MRIModule):
    def __init__(self):
        super().__init__()
        # self.model = UNet2DModel(in_channels=2, out_channels=2, block_out_channels=(64, 128, 256, 512))
        self.model = Complex_FastMRI(1, 1)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()

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

    def forward(self, x):
        # return self.model.forward(x, timestep=torch.tensor(0)).sample
        return self.model.forward(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)
    
    def training_step(self, batch, batch_idx):
        input = batch.kspace
        input = fT.complex_center_crop(input, (320,320))
        input, std, mean = self.norm(input)
        input_complex = torch.view_as_complex(input).unsqueeze(0)
        output_complex = self(input_complex)
        loss = self.criterion(input_complex, output_complex)
        output = torch.view_as_real(output_complex.squeeze(0))
        output = self.unnorm(output, mean, std)
        img = kspace_to_mri(input)
        rec_img = kspace_to_mri(output)
        mri_loss = self.criterion(img, rec_img)
        self.log("mri_loss", mri_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch.kspace.shape[0])
        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch.kspace.shape[0])
        return {
            "loss": loss,
            "input": input, 
            "reconstruction": output,
            "rec_img": rec_img,
            "target": img

        }


class ImageUNet(MRIModule):
    def __init__(self, num_log_images = 16):
        super().__init__(num_log_images)
        self.model = Unet_FastMRI(1, 1)
        self.criterion = nn.L1Loss()

    def forward(self, x: torch.Tensor):
        start = time.time()
        output = self.model(x.unsqueeze(1)).squeeze(1)
        end = time.time()
        return output, end-start
    def training_step(self, batch, batch_idx):
        x_hat, t = self(batch.image)
        loss = self.criterion(x_hat, batch.target)
        self.log("l1_loss", loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=1)
        self.log("time", round(t,5), on_epoch=True, on_step=True, sync_dist=True, batch_size=1)
        return {
            "loss": loss,
            "input": torch.zeros(1,320, 320, 2), 
            "reconstruction": torch.zeros(1,320, 320, 2),
            "rec_img": x_hat,
            "target": batch.target

        }
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)


    
class NormUnetModule(MRIModule):
    def __init__(self, num_log_images = 16):
        super().__init__(num_log_images)
        self.model = NormUnet(chans=2,num_pools=4)
        self.criterion = nn.L1Loss()

    def forward(self, x):
            # return self.model.forward(x, timestep=torch.tensor(0)).sample
            output = self.model.forward(x)
            return output

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)
        
    def training_step(self, batch, batch_idx):
        input = fT.complex_center_crop(batch.kspace, (320, 320))
        output = self(input)
        loss = self.criterion(input, output)
        rec_img = kspace_to_mri(output)
        target = kspace_to_mri(input)
        # target, rec_img = fT.center_crop_to_smallest(batch.target, rec_img)
        mri_loss = self.criterion(target, rec_img)
        self.log("mri_loss", mri_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False, batch_size=batch.kspace.shape[0])
        self.log("kspace_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False, batch_size=batch.kspace.shape[0])
        self.log("total_loss", loss + mri_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch.kspace.shape[0])

        return {
            "loss": loss,
            "input": input, 
            "reconstruction": output,
            "rec_img": rec_img,
            "target": target

        }