import pytorch_lightning as pl
from modules.autoencoders import Encoder, Decoder
import torch
from modules.distributions import DiagonalGaussianDistribution
from modules.losses import LPIPSWithDiscriminator
import torch.nn.functional as F
from modules.transforms import complex_center_crop_to_smallest
from pl_modules.mri_module import MRIModule
import fastmri
from fastmri.data.transforms import center_crop_to_smallest
from fastmri.losses import SSIMLoss

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
        self.automatic_optimization = False
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = LPIPSWithDiscriminator(disc_start=10001, kl_weight=0.000001, disc_weight=0.5, disc_in_channels=1, perceptual_weight=0)
        self.SSIM = SSIMLoss()
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

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


    def training_step(self, batch, batch_idx):
        opt_0, opt_1 = self.optimizers()
        inputs = batch.masked_kspace.permute(0, 3, 1, 2).contiguous()
        reconstructions, posterior = self(inputs)
        if inputs.shape[-1] != reconstructions.shape[-1]:
            inputs, reconstructions = complex_center_crop_to_smallest(inputs,reconstructions)
        rec_img = fastmri.complex_abs(fastmri.ifft2c(reconstructions.permute(0,2,3,1).contiguous()))
        target = fastmri.complex_abs(fastmri.ifft2c(inputs.permute(0,2,3,1).contiguous()))
        

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(target.unsqueeze(0), rec_img.unsqueeze(0), posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        
        opt_0.zero_grad()
        self.manual_backward(aeloss + 100*self.SSIM(target, rec_img, batch.max_value))
        opt_0.step()

        inputs = batch.masked_kspace.permute(0, 3, 1, 2).contiguous()
        reconstructions, posterior = self(inputs)
        if inputs.shape[-1] != reconstructions.shape[-1]:
            inputs, reconstructions = complex_center_crop_to_smallest(inputs,reconstructions)
        rec_img = fastmri.complex_abs(fastmri.ifft2c(reconstructions.permute(0,2,3,1).contiguous()))
        target = fastmri.complex_abs(fastmri.ifft2c(inputs.permute(0,2,3,1).contiguous()))

        # train the discriminator
        discloss, log_dict_disc = self.loss(target.unsqueeze(0), rec_img.unsqueeze(0), posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

        
        opt_1.zero_grad()
        self.manual_backward(discloss+ 100*self.SSIM(target, rec_img, batch.max_value))
        opt_1.step()

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=inputs.shape[0])
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=inputs.shape[0])

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=inputs.shape[0])
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=inputs.shape[0])


        return {
            "input": inputs.permute(0,2,3,1).contiguous(), 
            "reconstruction": reconstructions.permute(0,2,3,1).contiguous(),
            "rec_img": rec_img,
            "target": target
        }


    def validation_step(self, batch, batch_idx):
        inputs = batch.masked_kspace.permute(0, 3, 1, 2)
        reconstructions, posterior = self(inputs)
        if inputs.shape[-1] != reconstructions.shape[-1]:
            inputs, reconstructions = complex_center_crop_to_smallest(inputs,reconstructions)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
    
        rec_img = fastmri.complex_abs(fastmri.ifft2c(reconstructions.permute(0,2,3,1)))
        target, rec_img = center_crop_to_smallest(batch.target, rec_img)

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "target": target,
            "rec_img": rec_img
        }

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
