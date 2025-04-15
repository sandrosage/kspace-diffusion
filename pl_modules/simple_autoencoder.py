import pytorch_lightning as pl
from modules.autoencoders import KSpaceAutoencoder
from modules.transforms import log_normalize_kspace, preprocess_kspace, extract_phase, reconstruct_kspace
from torch.nn import L1Loss
from torch import optim


class SimpleAutoencoder(pl.LightningModule):
    def __init__(self, latent_dim = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = KSpaceAutoencoder(latent_dim=self.latent_dim)
        self.criterion = L1Loss()

    def forward(self, x):
        dec = self.model(x)

    def training_step(self, batch, batch_idx):
        input = batch.masked_kspace
        print(input.shape)
        log_input = preprocess_kspace(input).permute(0,3,1,2).contiguous()
        dec = self(log_input)
        loss = self.criterion(log_input, dec)
        kspace_loss = self.criterion(input, reconstruct_kspace(dec))
        self.log_dict({
            "log_loss": loss, 
            "kspace_loss": kspace_loss
        }
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
