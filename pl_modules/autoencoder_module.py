import torch
from modules.autoencoders import Encoder, Decoder
from modules.distributions import DiagonalGaussianDistribution
from torch import nn
from modules.transforms import complex_center_crop_to_smallest
import fastmri
from fastmri.data.transforms import center_crop_to_smallest
from modules.losses import ELBOLoss
from pl_modules.mri_module import MRIModule

class AutoencoderKL(MRIModule):
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
        self.ELBO = ELBOLoss()


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
        return dec, posterior

    def shared_step(self, batch):
        input = batch.masked_kspace.permute(0,3,1,2)
        reconstruction, posterior = self(input)
        if input.shape[-1] != reconstruction.shape[-1]:
            input, reconstruction = complex_center_crop_to_smallest(input,reconstruction)
        rec_img = fastmri.complex_abs(fastmri.ifft2c(reconstruction.permute(0,2,3,1)))
        target, rec_img = center_crop_to_smallest(batch.target, rec_img)
        return input, posterior, reconstruction, target, rec_img

    def training_step(self, batch, batch_idx):
        # 1: Get the output of the model
        input, posterior, reconstruction, target, rec_img = self.shared_step(batch)
        # 2: Compute the losses
        rec_loss = nn.functional.mse_loss(input, reconstruction) # MSE loss between the kspaces
        elbo_loss = self.ELBO(rec_loss, posterior.kl(), input) # ElBO for optimizing the VAE
        ssim_loss = self.SSIM(rec_img, target, data_range=batch.max_value) # SSIM on the reconstructed image and target image

        # 3: Encapsulate the metrics
        metrics = {
            "train/elbo_loss": elbo_loss,
            "train/rec_loss": rec_loss,
            "train/ssim_loss": ssim_loss
        }
        # 4: Log the metrics
        self.log_dict(metrics, sync_dist=True, on_epoch=True, on_step=True, batch_size=input.shape[0])
        return {
            "loss": elbo_loss,
            "input": input.permute(0,2,3,1).contiguous(), 
            "reconstruction": reconstruction.permute(0,2,3,1).contiguous(),
            "rec_img": rec_img
        }
        

    def validation_step(self, batch, batch_idx):
        # 1: Get the output of the model
        _, _, _, target, rec_img = self.shared_step(batch)

        # 2: Return batch information to on_validation_batch_end
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "target": target,
            "rec_img": rec_img
        }

    def test_step(self, batch, batch_idx):
        # 1: Get the output of the model
        _, _, _, target, rec_img = self.shared_step(batch)

        # 2: Return batch information to on_validation_batch_end
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "target": target,
            "rec_img": rec_img
        }
        

    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return opt
