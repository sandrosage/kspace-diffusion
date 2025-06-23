from pl_modules.mri_module import NewMRIModule
from diffusers.models.autoencoders.vae import Decoder, Encoder
from torch.nn import L1Loss
import torch
from modules.transforms import KspaceUNetSample, norm, unnorm, kspace_to_mri
from torch import optim, nn
from typing import Tuple, Literal
import torch.nn.functional as F

class ZeroPaddingTransform(nn.Module):
    def __init__(self, target_size: Tuple[int, int]):
        super().__init__()
        self.target_size = target_size
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h, w = input.shape[-2:]  # Assuming (C, H, W) or (B, C, H, W)
        pad_h = max(0, self.target_size[0] - h)
        pad_w = max(0, self.target_size[1] - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # Padding format for torch.nn.functional.pad: (left, right, top, bottom)
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        return F.pad(input, padding, mode='constant', value=0)
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_size={self.target_size})"
    
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
    
def create_channels(n: int, chans: int = 32):
    blocks = (chans,)
    for i in range(n-1):
        blocks += (2*chans,)
        chans = 2*chans
    return blocks

class Diffusers_VAE(NewMRIModule):
    def __init__(self, 
                 in_channels: int = 2, 
                 out_channels: int = 2, 
                 latent_dim: int = 16,
                 down_layers: int = 4,  
                 num_log_images = 32):
        super().__init__(num_log_images)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.down_layers = down_layers
        self.transform = AdaptivePoolTransform((640,368))

        self.save_hyperparameters()
        
        down_block_out_channels = create_channels(self.down_layers)
        up_block_out_channels = create_channels(self.down_layers)[::-1]
        up_block_out_channels = down_block_out_channels[::-1]
        down_blocks = self.down_layers*("DownEncoderBlock2D", )
        up_blocks = self.down_layers* ("UpDecoderBlock2D", )

        self.encoder = Encoder(
            in_channels=self.in_channels, 
            out_channels=self.latent_dim, 
            down_block_types=down_blocks, 
            block_out_channels=down_block_out_channels,
            double_z=False)
        
        self.decoder = Decoder(
            in_channels=self.latent_dim,
            out_channels=self.out_channels,
            up_block_types=up_blocks,
            block_out_channels=up_block_out_channels
        )

        self.criterion = L1Loss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def shared_step(self, batch: KspaceUNetSample):
        input = batch.full_kspace
        input = input.permute(0,3,1,2).contiguous()
        input = self.transform(input)
        input, mean, std = norm(input)
        output = self(input)
        loss = self.criterion(input,output)
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

    def validation_step(self, batch:KspaceUNetSample, batch_idx):
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
    
    def configure_optimizers(self):
        return optim.Adam(list(self.encoder.parameters())+ list(self.decoder.parameters()), lr=1e-4)
    