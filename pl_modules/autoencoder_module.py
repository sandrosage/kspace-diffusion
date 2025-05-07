import torch
from modules.autoencoders import Encoder, Decoder
from modules.distributions import DiagonalGaussianDistribution
from torch import nn
from modules.transforms import complex_center_crop_to_smallest, reconstruct_kspace, kspace_to_mri, complex_center_crop_c_h_w
import fastmri
from fastmri.data.transforms import center_crop_to_smallest, complex_center_crop
from modules.losses import ELBOLoss
from pl_modules.mri_module import MRIModule
from typing import Tuple

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

    def shared_step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        input = batch.kspace
        input  = complex_center_crop(input, (320,320)).permute(0,3,1,2).contiguous()
        input,mean,std = self.norm(input)
        output, posterior = self(input)
        # if input.shape[-1] != output.shape[-1]:
        #     input_euler, output_euler = complex_center_crop_to_smallest(input,output_euler)
        loss = nn.functional.l1_loss(input,output)
        input = self.unnorm(input, mean, std)
        output = self.unnorm(output, mean, std)
        input = input.permute(0,2,3,1).contiguous()
        output = output.permute(0,2,3,1).contiguous()
        output_mri = kspace_to_mri(output)
        input_mri = kspace_to_mri(input)
        # 2: Compute the losses
        elbo_loss = self.ELBO(loss, posterior.kl(), input) # ElBO for optimizing the VAE
        ssim_loss = self.SSIM(input_mri.contiguous(), output_mri.contiguous(), data_range=batch.max_value) # SSIM on the reconstructed image and target image

        # 3: Encapsulate the metrics
        metrics = {
            "train/elbo_loss": elbo_loss,
            "train/ssim_loss": ssim_loss
        }
        # 4: Log the metrics
        self.log_dict(metrics, sync_dist=True, on_epoch=True, on_step=True, batch_size=input.shape[0])
        return {
            "loss": elbo_loss,
            "input": input, 
            "reconstruction": output,
            "rec_img": output_mri,
            "target": input_mri
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


class AutoencoderKLOriginal(MRIModule):
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
        input = batch.masked_kspace
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
        rec_loss = nn.functional.mse_loss(target.contiguous(), rec_img.contiguous()) # MSE loss between the kspaces
        elbo_loss = self.ELBO(rec_loss, posterior.kl(), input) # ElBO for optimizing the VAE
        ssim_loss = self.SSIM(target.contiguous(), rec_img.contiguous(), data_range=batch.max_value) # SSIM on the reconstructed image and target image

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
    
class AutoencoderComplex(MRIModule):
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
        self.real_encoder = Encoder(**ddconfig)
        self.real_decoder = Decoder(**ddconfig)
        self.imag_encoder = Encoder(**ddconfig)
        self.imag_decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.real_quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.real_post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.imag_quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.imag_post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if monitor is not None:
            self.monitor = monitor
        self.ELBO = ELBOLoss()

    def encode_real(self, x):
        h = self.real_encoder(x)
        moments = self.real_quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def encode_imag(self, x):
        h = self.imag_encoder(x)
        moments = self.imag_quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode_real(self, z):
        z = self.real_post_quant_conv(z)
        dec = self.real_decoder(z)
        return dec
    
    def decode_imag(self, z):
        z = self.imag_post_quant_conv(z)
        dec = self.imag_decoder(z)
        return dec

    def forward(self, input, sample_posterior=False):
        real, imag = input[:,0:1,...], input[:,1:2,...]
        real_posterior = self.encode_real(real)
        imag_posterior = self.encode_imag(imag)

        if sample_posterior:
            real_z = real_posterior.sample()
            imag_z = imag_posterior.sample()
        else:
            real_z = real_posterior.mode()
            imag_z = imag_posterior.mode()
        real_dec = self.decode_real(real_z)
        imag_dec = self.decode_imag(imag_z)
        return real_dec, imag_dec, real_posterior, imag_posterior

    def shared_step(self, batch):
        input = batch.masked_kspace.permute(0,3,1,2)
        input = complex_center_crop_c_h_w(input, (320,320))
        real_dec, imag_dec, real_posterior, imag_posterior = self(input)
        output = torch.cat([real_dec, imag_dec], dim=1)
        if input.shape[-1] != output.shape[-1]:
            input, output = complex_center_crop_to_smallest(input,output)
        rec_img = fastmri.complex_abs(fastmri.ifft2c(output.permute(0,2,3,1)))
        target, rec_img = center_crop_to_smallest(batch.target, rec_img)
        return input, output, target, rec_img

    def training_step(self, batch, batch_idx):
        # 1: Get the output of the model
        input, output, target, rec_img = self.shared_step(batch)
        # 2: Compute the losses
        rec_loss = nn.functional.l1_loss(input.contiguous(), output.contiguous()) # MSE loss between the kspaces
        # elbo_loss = self.ELBO(rec_loss, posterior.kl(), input) # ElBO for optimizing the VAE
        ssim_loss = self.SSIM(target.contiguous(), rec_img.contiguous(), data_range=batch.max_value) # SSIM on the reconstructed image and target image

        # 3: Encapsulate the metrics
        metrics = {
            "train/rec_loss": rec_loss,
            "train/ssim_loss": ssim_loss
        }
        # 4: Log the metrics
        self.log_dict(metrics, sync_dist=True, on_epoch=True, on_step=True, batch_size=input.shape[0])
        return {
            "loss": rec_loss,
            "input": input.permute(0,2,3,1).contiguous(), 
            "reconstruction": output.permute(0,2,3,1).contiguous(),
            "rec_img": rec_img,
            "target": batch.target
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
        opt = torch.optim.Adam(
            list(self.real_encoder.parameters())+
            list(self.imag_encoder.parameters())+
            list(self.real_decoder.parameters())+
            list(self.imag_decoder.parameters())+
            list(self.real_quant_conv.parameters())+
            list(self.imag_quant_conv.parameters())+
            list(self.real_post_quant_conv.parameters())+
            list(self.imag_post_quant_conv.parameters()),
            lr=self.learning_rate, betas=(0.5, 0.9))
        return opt