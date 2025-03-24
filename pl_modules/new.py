import pytorch_lightning as pl
import torch
from modules.autoencoders import Encoder, Decoder
from modules.distributions import DiagonalGaussianDistribution
from modules.losses import LPIPSWithDiscriminator
import torch.nn.functional as F
from torch import nn
from modules.transforms import complex_center_crop_to_smallest

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
        ddconfig,
        lossconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        lr=4.5e-6
        ):
        super().__init__()
        self.learning_rate = lr
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if monitor is not None:
            self.monitor = monitor


    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        kl = posterior.kl()
        return dec, posterior, kl

    def training_step(self, batch, batch_idx):
        input = batch.kspace.permute(0,3,1,2)
        reconstructions, _, kl = self(input)
        if input.shape[-1] != reconstructions.shape[-1]:
            input, reconstructions = complex_center_crop_to_smallest(input,reconstructions)
        rec_loss = nn.functional.mse_loss(input, reconstructions)
        elbo = (rec_loss + kl) / len(input) # we want to maximize the elbo -> so we minimize the negative of the elbo
        loss = - elbo
        self.log("train_loss", loss,  sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae
