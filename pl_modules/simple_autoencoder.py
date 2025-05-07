from modules.autoencoders import KSpaceAutoencoder
from modules.transforms import norm, unnorm, kspace_to_mri
from torch.nn import L1Loss
from torch import optim
import fastmri.data.transforms as fT
from pl_modules import MRIModule


class SimpleAutoencoder(MRIModule):
    def __init__(self, latent_dim = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = KSpaceAutoencoder(latent_dim=self.latent_dim)
        self.criterion = L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input = fT.complex_center_crop(batch.kspace,(320,320))
        input = input.permute(0,3,1,2).contiguous()
        input, mean, std = norm(input)
        output = self(input)
        loss = self.criterion(input, output)
        input = unnorm(input, mean, std)
        output = unnorm(output, mean,std)
        input_mri = kspace_to_mri(input)
        output_mri = kspace_to_mri(output)
        self.log_dict({
            "log_loss": loss,
            },
            on_epoch=True, on_step=True, batch_size=input.shape[0])
        return {
            "loss": loss,
            "input": input, 
            "reconstruction": output,
            "rec_img": output_mri,
            "target": input_mri

        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
