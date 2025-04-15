from torch import nn, optim
from functools import reduce
import operator
from typing import Tuple  
import torch  
from modules.transforms import complex_center_crop_c_h_w, kspace_to_mri, reconstruct_kspace
from pl_modules.mri_module import MRIModule

class Simple(nn.Module):
    def __init__(self, latent_dim: int = 128, input_shape: Tuple[int, int, int, int] = (1,1,320,320)):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.input_dim = reduce(operator.mul, self.input_shape)
        self.real_encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, self.latent_dim)
        )

        self.real_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, self.input_dim),
            nn.Tanh()
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, self.input_dim),
            nn.Tanh()
        )

    def _encode(self, x, y):
        return self.real_encoder(x), self.encoder(y)
    
    def _decode(self, z, w):
        return self.real_decoder(z), self.decoder(w)
    
    def forward(self, x, y):
        x = x.flatten()
        y = y.flatten()
        z,w = self._encode(x,y)
        x_hat, y_hat = self._decode(z,w)
        return x_hat.view(self.input_shape), y_hat.view(self.input_shape)




class SimpleAutoencoder(MRIModule):
    def __init__(self, is_euler: bool = False):
        super().__init__()
        self.model = Simple()
        self.criterion = nn.MSELoss()
        self.is_euler = is_euler

    def forward(self, x, y):
        return self.model.forward(x,y)
    
    def training_step(self, batch, batch_idx):
        input = batch.kspace.permute(0,3,1,2)
        input = complex_center_crop_c_h_w(input, (320,320))
        r_in, i_in = input[:,0:1,...], input[:,1:2,...]
        r_out, i_out = self.forward(r_in, i_in)
        r_loss = self.criterion(r_in, r_out)
        i_loss = self.criterion(i_in, i_out)
        output = torch.cat([r_out, i_out], dim=1)
        if self.is_euler:
            output = reconstruct_kspace(output)
        loss = self.criterion(input, output)
        input = input.permute(0,2,3,1).contiguous()
        output = output.permute(0,2,3,1).contiguous()
        img = kspace_to_mri(input)
        rec_img = kspace_to_mri(output)
        self.log("r_loss", r_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch.kspace.shape[0])
        self.log("i_loss", i_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch.kspace.shape[0])
        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch.kspace.shape[0])
        return {
            "loss": loss,
            "input": input, 
            "reconstruction": output,
            "rec_img": rec_img,
            "target": img

        }
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)
    