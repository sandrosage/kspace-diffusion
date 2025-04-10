import pytorch_lightning as pl
from modules.simple_autoencoder import KSpaceAutoencoder
from torch import nn, optim


class SimpleAutoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = KSpaceAutoencoder()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model.forward(x)
    
    def training_step(self, batch, batch_idx):
        input = batch.kspace
        output = self.forward(input)
        loss = self.criterion(input, output)
        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch.shape[0])
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)
    